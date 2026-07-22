"""Explicit-grammar math parsing and bounded answer verification.

No student text is passed to ``sympy.parse_expr`` or Python evaluation.  The
small recursive-descent parser below constructs only numeric atoms, symbols,
arithmetic nodes, and an explicit function allow-list.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing
import os
import queue
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Callable, Final, Iterable, Mapping, NamedTuple, Protocol

import sympy
from pydantic import BaseModel, ConfigDict

from tutor.schemas.assessment import (
    AnswerSpec,
    AntiderivativeAnswerSpec,
    ChoiceAnswerSpec,
    FiniteSetAnswerSpec,
    IntervalSetAnswerSpec,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    SymbolicAnswerSpec,
    answer_spec_adapter,
)

MAX_INPUT_LENGTH: Final = 256
MAX_AST_NODES: Final = 128
MAX_AST_DEPTH: Final = 16
MAX_EXPONENT_MAGNITUDE: Final = 20
WORKER_START_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_VERIFIER_POOL_SIZE: Final = max(1, min(4, os.cpu_count() or 1))
DEFAULT_VERIFIER_QUEUE_CAPACITY: Final = 32
DEFAULT_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS: Final = 0.25

_RETRYABLE_OVERLOAD_CODES: Final = frozenset(
    {"verifier_saturated", "verifier_queue_timeout"}
)

_ASSIGNMENT_LHS = re.compile(
    r"^[A-Za-z][A-Za-z0-9]*(?:\([A-Za-z][A-Za-z0-9]*\))?$"
)
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_TOKEN = re.compile(
    r"""
    (?P<SPACE>\s+)
    |(?P<NUMBER>(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)
    |(?P<IDENT>[A-Za-z][A-Za-z0-9]*)
    |(?P<POW>\*\*)
    |(?P<OP>[+\-*/^(),])
    """,
    re.VERBOSE,
)
_FUNCTIONS: Final = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "cot": sympy.cot,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "sqrt": sympy.sqrt,
    "Abs": sympy.Abs,
}
_CONSTANTS: Final = {"pi": sympy.pi, "e": sympy.E}
_ALLOW_ANY_ASSIGNMENT = object()


def _bounded_rational(text: str) -> sympy.Rational:
    scientific = re.search(r"[eE]([+-]?\d+)$", text)
    if scientific is not None:
        exponent = int(scientific.group(1))
        if abs(exponent) > MAX_EXPONENT_MAGNITUDE:
            raise MathVerificationError(
                f"numeric exponent magnitude exceeds {MAX_EXPONENT_MAGNITUDE}"
            )
    try:
        return sympy.Rational(text)
    except (TypeError, ValueError) as exc:
        raise MathVerificationError(f"invalid number {text!r}") from exc


class MathVerificationError(ValueError):
    """Raised when text is outside the restricted mathematical grammar."""


class VerificationStatus(StrEnum):
    """Typed outcome used by v2 state machines."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID = "invalid"
    TIMEOUT = "timeout"


class VerificationResult(BaseModel):
    """A safe verification result; expected answers never appear in it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    code: str
    normalized_form: str | None = None

    @property
    def correct(self) -> bool:
        """Whether the result is a verified mathematical match."""
        return self.status == VerificationStatus.CORRECT

    @property
    def retryable_overload(self) -> bool:
        """Whether verification did not start because local capacity was full."""
        return self.code in _RETRYABLE_OVERLOAD_CODES


def _environment_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _environment_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class VerifierPoolSettings:
    """Bounded process-pool configuration, suitable for environment loading."""

    pool_size: int = DEFAULT_VERIFIER_POOL_SIZE
    queue_capacity: int = DEFAULT_VERIFIER_QUEUE_CAPACITY
    queue_wait_timeout_seconds: float = DEFAULT_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not 1 <= self.pool_size <= 64:
            raise ValueError("pool_size must be between 1 and 64")
        if not 0 <= self.queue_capacity <= 4096:
            raise ValueError("queue_capacity must be between 0 and 4096")
        if not 0.001 <= self.queue_wait_timeout_seconds <= 60:
            raise ValueError("queue_wait_timeout_seconds must be between 0.001 and 60")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> VerifierPoolSettings:
        source = os.environ if environ is None else environ
        return cls(
            pool_size=_environment_int(
                source,
                "TUTOR_VERIFIER_POOL_SIZE",
                DEFAULT_VERIFIER_POOL_SIZE,
                minimum=1,
                maximum=64,
            ),
            queue_capacity=_environment_int(
                source,
                "TUTOR_VERIFIER_QUEUE_CAPACITY",
                DEFAULT_VERIFIER_QUEUE_CAPACITY,
                minimum=0,
                maximum=4096,
            ),
            queue_wait_timeout_seconds=_environment_float(
                source,
                "TUTOR_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS",
                DEFAULT_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS,
                minimum=0.001,
                maximum=60,
            ),
        )


@dataclass(frozen=True, slots=True)
class VerifierMetricsSnapshot:
    """Aggregate-only verifier counters; no learner input is retained."""

    requests: int
    invalid: int
    timed_out: int
    saturated: int


class _VerifierWorker(Protocol):
    def verify(
        self,
        spec: AnswerSpec,
        given: str,
        timeout_seconds: float,
    ) -> VerificationResult: ...

    def close(self) -> None: ...


class _LexToken(NamedTuple):
    kind: str
    value: str


class _ParsedExpression(NamedTuple):
    """One safely constructed expression and its syntactic AST depth."""

    value: sympy.Expr
    depth: int


def _normalized_lhs(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _split_assignment(
    text: str,
    allowed_lhs: str | None | object,
) -> str:
    candidate = text.strip()
    if not candidate:
        raise MathVerificationError("empty math input")
    if len(candidate) > MAX_INPUT_LENGTH:
        raise MathVerificationError(
            f"math input exceeds {MAX_INPUT_LENGTH} characters"
        )
    if any(operator in candidate for operator in ("!=", "<=", ">=", "==")):
        raise MathVerificationError("relations and comparisons are not expressions")
    equals = candidate.count("=")
    if equals == 0:
        return candidate
    if equals != 1:
        raise MathVerificationError("an expression may contain at most one assignment")
    lhs, rhs = candidate.split("=", 1)
    normalized_lhs = _normalized_lhs(lhs)
    if not _ASSIGNMENT_LHS.fullmatch(normalized_lhs) or not rhs.strip():
        raise MathVerificationError("malformed assignment")
    if allowed_lhs is None:
        raise MathVerificationError("assignments are not allowed for this item")
    if (
        allowed_lhs is not _ALLOW_ANY_ASSIGNMENT
        and normalized_lhs != _normalized_lhs(str(allowed_lhs))
    ):
        raise MathVerificationError(
            f"assignment must use left-hand side {allowed_lhs!r}"
        )
    return rhs.strip()


def strip_assignment_prefix(text: str) -> str:
    """Strip one syntactically valid legacy assignment, never a comparison.

    This compatibility helper intentionally accepts any safe left-hand side.
    V2 verification instead passes the exact authored ``assignment_lhs``.
    """
    try:
        return _split_assignment(text, _ALLOW_ANY_ASSIGNMENT)
    except MathVerificationError:
        return text.strip()


def _lex(text: str) -> list[_LexToken]:
    tokens: list[_LexToken] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise MathVerificationError(
                f"unsupported character at position {position}"
            )
        position = match.end()
        if match.lastgroup != "SPACE":
            tokens.append(_LexToken(match.lastgroup or "", match.group()))
    if not tokens:
        raise MathVerificationError("empty math input")
    return tokens


class _ExpressionParser:
    """Recursive-descent parser that directly constructs safe SymPy nodes."""

    def __init__(
        self,
        tokens: list[_LexToken],
        *,
        allowed_variables: set[str] | None,
        allowed_functions: set[str],
    ) -> None:
        self._tokens = tokens
        self._position = 0
        self._allowed_variables = allowed_variables
        self._allowed_functions = allowed_functions
        self._node_count = 0

    def parse(self) -> sympy.Expr:
        expression = self._parse_sum()
        if self._peek() is not None:
            token = self._peek()
            raise MathVerificationError(
                f"unexpected token {token.value!r}" if token else "unexpected input"
            )
        return expression.value

    def _record(
        self,
        value: sympy.Expr,
        *children: _ParsedExpression,
    ) -> _ParsedExpression:
        """Record one real syntax node rather than parser call-stack depth.

        The recursive-descent grammar has several precedence methods between a
        parenthesis and an atom. Counting each method call as an AST level made
        ordinary factored quotient-rule answers exceed the depth-16 contract.
        Parentheses and precedence layers are not AST nodes; operators,
        functions, and atoms are.
        """
        depth = 1 + max((child.depth for child in children), default=0)
        if depth > MAX_AST_DEPTH:
            raise MathVerificationError(
                f"expression exceeds maximum depth {MAX_AST_DEPTH}"
            )
        self._node_count += 1
        if self._node_count > MAX_AST_NODES:
            raise MathVerificationError(
                f"expression exceeds maximum size {MAX_AST_NODES}"
            )
        return _ParsedExpression(value=value, depth=depth)

    def _peek(self, offset: int = 0) -> _LexToken | None:
        position = self._position + offset
        return self._tokens[position] if position < len(self._tokens) else None

    def _take(self, value: str | None = None) -> _LexToken:
        token = self._peek()
        if token is None:
            raise MathVerificationError("unexpected end of expression")
        if value is not None and token.value != value:
            raise MathVerificationError(f"expected {value!r}, got {token.value!r}")
        self._position += 1
        return token

    def _parse_sum(self) -> _ParsedExpression:
        result = self._parse_product()
        while (token := self._peek()) is not None and token.value in {"+", "-"}:
            operator = self._take().value
            right = self._parse_product()
            value = (
                result.value + right.value
                if operator == "+"
                else result.value - right.value
            )
            result = self._record(value, result, right)
        return result

    def _starts_implicit_factor(self, token: _LexToken | None) -> bool:
        return token is not None and (
            token.kind in {"NUMBER", "IDENT"} or token.value == "("
        )

    def _parse_product(self) -> _ParsedExpression:
        result = self._parse_unary()
        previous = self._tokens[self._position - 1]
        while (token := self._peek()) is not None:
            if token.value in {"*", "/"}:
                operator = self._take().value
                right = self._parse_unary()
            elif self._starts_implicit_factor(token):
                if previous.kind == "NUMBER" and token.kind == "NUMBER":
                    raise MathVerificationError(
                        "adjacent numbers require an explicit operator"
                    )
                operator = "*"
                right = self._parse_unary()
            else:
                break
            value = (
                result.value * right.value
                if operator == "*"
                else result.value / right.value
            )
            result = self._record(value, result, right)
            previous = self._tokens[self._position - 1]
        return result

    def _parse_unary(self) -> _ParsedExpression:
        token = self._peek()
        if token is not None and token.value in {"+", "-"}:
            operator = self._take().value
            operand = self._parse_unary()
            value = operand.value if operator == "+" else -operand.value
            return self._record(value, operand)
        return self._parse_power()

    def _parse_power(self) -> _ParsedExpression:
        base = self._parse_primary()
        token = self._peek()
        if token is not None and token.value in {"^", "**"}:
            self._take()
            exponent = self._parse_unary()
            if exponent.value.is_number:
                try:
                    numeric_exponent = float(exponent.value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise MathVerificationError("invalid exponent") from exc
                if (
                    not math.isfinite(numeric_exponent)
                    or abs(numeric_exponent) > MAX_EXPONENT_MAGNITUDE
                ):
                    raise MathVerificationError(
                        f"numeric exponent magnitude exceeds {MAX_EXPONENT_MAGNITUDE}"
                    )
            return self._record(
                sympy.Pow(base.value, exponent.value),
                base,
                exponent,
            )
        return base

    def _parse_primary(self) -> _ParsedExpression:
        token = self._take()
        if token.kind == "NUMBER":
            return self._record(_bounded_rational(token.value))
        if token.value == "(":
            expression = self._parse_sum()
            self._take(")")
            return expression
        if token.kind != "IDENT":
            raise MathVerificationError(f"unexpected token {token.value!r}")

        name = token.value
        function_power: sympy.Expr | None = None
        if (
            name in self._allowed_functions
            and (power_token := self._peek()) is not None
            and power_token.value in {"^", "**"}
        ):
            self._take()
            sign = 1
            if self._peek() is not None and self._peek().value in {"+", "-"}:
                sign = -1 if self._take().value == "-" else 1
            exponent_token = self._take()
            if exponent_token.kind != "NUMBER":
                raise MathVerificationError(
                    "a function exponent must be an explicit number"
                )
            function_power = sign * _bounded_rational(exponent_token.value)
            try:
                numeric_exponent = float(function_power)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MathVerificationError("invalid function exponent") from exc
            if (
                not math.isfinite(numeric_exponent)
                or abs(numeric_exponent) > MAX_EXPONENT_MAGNITUDE
            ):
                raise MathVerificationError(
                    f"numeric exponent magnitude exceeds {MAX_EXPONENT_MAGNITUDE}"
                )
        if name in self._allowed_functions:
            if self._peek() is None or self._peek().value != "(":
                raise MathVerificationError(f"function {name!r} requires parentheses")
            self._take("(")
            argument = self._parse_sum()
            self._take(")")
            function = self._record(_FUNCTIONS[name](argument.value), argument)
            if function_power is None:
                return function
            exponent = self._record(function_power)
            return self._record(
                sympy.Pow(function.value, exponent.value),
                function,
                exponent,
            )

        if self._peek() is not None and self._peek().value == "(":
            raise MathVerificationError(f"function {name!r} is not allowed")
        if name in _CONSTANTS:
            return self._record(_CONSTANTS[name])
        if self._allowed_variables is not None and name not in self._allowed_variables:
            raise MathVerificationError(f"variable {name!r} is not allowed")
        return self._record(sympy.Symbol(name))


def parse_restricted(
    text: str,
    *,
    allowed_variables: Iterable[str] | None = None,
    allowed_functions: Iterable[str] | None = None,
    allowed_assignment_lhs: str | None | object = _ALLOW_ANY_ASSIGNMENT,
) -> sympy.Expr:
    """Parse a bounded expression using only explicitly constructed math nodes.

    The default assignment behavior preserves the v1 parser contract.  New
    assessment code must pass ``None`` (no assignment) or the exact authored
    left-hand side.
    """
    candidate = _split_assignment(text, allowed_assignment_lhs)
    functions = set(_FUNCTIONS if allowed_functions is None else allowed_functions)
    unknown_functions = functions - set(_FUNCTIONS)
    if unknown_functions:
        raise MathVerificationError(
            f"unknown allowed functions: {sorted(unknown_functions)}"
        )
    variables = None if allowed_variables is None else set(allowed_variables)
    if variables is not None:
        invalid = [name for name in variables if _IDENTIFIER.fullmatch(name) is None]
        if invalid:
            raise MathVerificationError(f"invalid allowed variables: {sorted(invalid)}")
    return _ExpressionParser(
        _lex(candidate),
        allowed_variables=variables,
        allowed_functions=functions,
    ).parse()


def _equivalent(left: sympy.Expr, right: sympy.Expr) -> bool:
    if left == right:
        return True
    try:
        difference = sympy.simplify(left - right)
    except Exception:  # noqa: BLE001 - a symbolic failure is a safe mismatch
        return False
    return difference == 0 or difference.equals(0) is True


def _parse_symbolic(
    text: str,
    *,
    variables: Iterable[str],
    functions: Iterable[str] = (),
    assignment_lhs: str | None = None,
) -> sympy.Expr:
    expression = parse_restricted(
        text,
        allowed_variables=variables,
        allowed_functions=functions,
        allowed_assignment_lhs=assignment_lhs,
    )
    if expression.has(sympy.zoo, sympy.nan, sympy.oo):
        raise MathVerificationError("undefined or infinite expressions are not allowed")
    if expression.is_finite is False:
        raise MathVerificationError("non-finite expressions are not allowed")
    return expression


def _split_container(text: str, opening: str, closing: str) -> list[str]:
    candidate = text.strip()
    if len(candidate) > MAX_INPUT_LENGTH:
        raise MathVerificationError(
            f"math input exceeds {MAX_INPUT_LENGTH} characters"
        )
    if not candidate.startswith(opening) or not candidate.endswith(closing):
        raise MathVerificationError(f"expected {opening}...{closing}")
    body = candidate[1:-1]
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(body):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise MathVerificationError("unbalanced parentheses")
        elif character == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    if depth != 0:
        raise MathVerificationError("unbalanced parentheses")
    parts.append(body[start:].strip())
    if not parts or any(not part for part in parts):
        raise MathVerificationError("empty container entry")
    if len(parts) > MAX_AST_NODES:
        raise MathVerificationError("too many container entries")
    return parts


def _unordered_equivalent(
    expected: list[sympy.Expr],
    given: list[sympy.Expr],
) -> bool:
    expected_members = _unique_equivalent_members(expected)
    given_members = _unique_equivalent_members(given)
    if len(expected_members) != len(given_members):
        return False
    unmatched = list(given_members)
    for expected_value in expected_members:
        for index, given_value in enumerate(unmatched):
            if _equivalent(expected_value, given_value):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def _unique_equivalent_members(values: list[sympy.Expr]) -> list[sympy.Expr]:
    """Canonicalize a finite set by removing equivalent duplicate members."""
    unique: list[sympy.Expr] = []
    for value in values:
        if not any(_equivalent(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _parse_endpoint(text: str) -> sympy.Expr:
    normalized = text.strip().lower()
    if normalized in {"-inf", "-infinity"}:
        return -sympy.oo
    if normalized in {"inf", "+inf", "infinity", "+infinity"}:
        return sympy.oo
    expression = parse_restricted(
        text,
        allowed_variables=set(),
        allowed_functions=(),
        allowed_assignment_lhs=None,
    )
    if expression.free_symbols:
        raise MathVerificationError("interval endpoints must be numeric")
    return expression


def _parse_intervals(text: str) -> list[tuple[sympy.Expr, sympy.Expr, bool, bool]]:
    candidate = text.strip().replace("∪", "U")
    if len(candidate) > MAX_INPUT_LENGTH:
        raise MathVerificationError(
            f"math input exceeds {MAX_INPUT_LENGTH} characters"
        )
    chunks = [chunk.strip() for chunk in re.split(r"\s+[Uu]\s+|\s*[Uu]\s*", candidate)]
    if not chunks or any(not chunk for chunk in chunks):
        raise MathVerificationError("invalid interval union")
    intervals: list[tuple[sympy.Expr, sympy.Expr, bool, bool]] = []
    for chunk in chunks:
        if chunk[0] not in "([" or chunk[-1] not in ")]":
            raise MathVerificationError("invalid interval delimiters")
        entries = _split_container(chunk, chunk[0], chunk[-1])
        if len(entries) != 2:
            raise MathVerificationError("an interval requires two endpoints")
        intervals.append(
            (
                _parse_endpoint(entries[0]),
                _parse_endpoint(entries[1]),
                chunk[0] == "[",
                chunk[-1] == "]",
            )
        )
    return intervals


def _intervals_equivalent(
    expected: list[tuple[sympy.Expr, sympy.Expr, bool, bool]],
    given: list[tuple[sympy.Expr, sympy.Expr, bool, bool]],
) -> bool:
    def represented_set(
        intervals: list[tuple[sympy.Expr, sympy.Expr, bool, bool]],
    ) -> sympy.Set:
        parts: list[sympy.Set] = []
        for lower, upper, lower_closed, upper_closed in intervals:
            difference = sympy.simplify(lower - upper)
            if difference.is_positive is True:
                raise MathVerificationError("interval lower bound exceeds upper bound")
            if difference == 0 and not (lower_closed and upper_closed):
                raise MathVerificationError("an open zero-width interval is empty")
            parts.append(
                sympy.Interval(
                    lower,
                    upper,
                    left_open=not lower_closed,
                    right_open=not upper_closed,
                )
            )
        return sympy.Union(*parts)

    return represented_set(expected) == represented_set(given)


def _invalid(code: str) -> VerificationResult:
    return VerificationResult(status=VerificationStatus.INVALID, code=code)


def _verify_local(spec: AnswerSpec, given: str) -> VerificationResult:
    """Verify in-process. Public v2 calls supervise this in a worker."""
    try:
        if isinstance(spec, ChoiceAnswerSpec):
            normalized = given.strip()
            if normalized not in spec.option_ids:
                return _invalid("unknown_choice")
            matched = normalized == spec.expected_choice_id
        elif isinstance(spec, NumericAnswerSpec):
            expected = parse_restricted(
                spec.expected,
                allowed_variables=set(),
                allowed_functions=(),
                allowed_assignment_lhs=None,
            )
            answer = parse_restricted(
                given,
                allowed_variables=set(),
                allowed_functions=(),
                allowed_assignment_lhs=None,
            )
            if expected.free_symbols or answer.free_symbols:
                return _invalid("numeric_answer_has_variables")
            try:
                expected_number = float(expected.evalf())
                answer_number = float(answer.evalf())
            except (TypeError, ValueError, OverflowError):
                return _invalid("numeric_answer_not_real")
            if not math.isfinite(expected_number) or not math.isfinite(answer_number):
                return _invalid("numeric_answer_not_finite")
            normalized = sympy.sstr(answer)
            matched = abs(expected_number - answer_number) <= spec.tolerance
        elif isinstance(spec, SymbolicAnswerSpec):
            expected = _parse_symbolic(
                spec.expected,
                variables=spec.variables,
                functions=spec.functions,
                assignment_lhs=spec.assignment_lhs,
            )
            answer = _parse_symbolic(
                given,
                variables=spec.variables,
                functions=spec.functions,
                assignment_lhs=spec.assignment_lhs,
            )
            normalized = sympy.sstr(answer)
            matched = _equivalent(expected, answer)
        elif isinstance(spec, AntiderivativeAnswerSpec):
            variables = set(spec.variables) | {spec.variable, "C"}
            expected = _parse_symbolic(
                spec.expected,
                variables=variables,
                functions=spec.functions,
            )
            answer = _parse_symbolic(
                given,
                variables=variables,
                functions=spec.functions,
            )
            variable = sympy.Symbol(spec.variable)
            constant = sympy.Symbol("C")
            normalized = sympy.sstr(answer)
            derivative_matches = _equivalent(
                sympy.diff(expected, variable),
                sympy.diff(answer, variable),
            )
            if spec.require_explicit_constant and not _equivalent(
                sympy.diff(expected, constant), sympy.Integer(1)
            ):
                return _invalid("invalid_answer_spec")
            constant_matches = not spec.require_explicit_constant or _equivalent(
                sympy.diff(answer, constant), sympy.Integer(1)
            )
            matched = derivative_matches and constant_matches
            if derivative_matches and not constant_matches:
                return VerificationResult(
                    status=VerificationStatus.INCORRECT,
                    code="explicit_constant_required",
                    normalized_form=normalized,
                )
        elif isinstance(spec, FiniteSetAnswerSpec):
            expected = [
                _parse_symbolic(
                    value,
                    variables=spec.variables,
                    functions=spec.functions,
                )
                for value in spec.expected
            ]
            answer = [
                _parse_symbolic(
                    value,
                    variables=spec.variables,
                    functions=spec.functions,
                )
                for value in _split_container(given, "{", "}")
            ]
            normalized_members = _unique_equivalent_members(answer)
            normalized = (
                "{"
                + ", ".join(sorted(sympy.sstr(value) for value in normalized_members))
                + "}"
            )
            matched = _unordered_equivalent(expected, answer)
        elif isinstance(spec, OrderedTupleAnswerSpec):
            expected = [
                _parse_symbolic(
                    value,
                    variables=spec.variables,
                    functions=spec.functions,
                )
                for value in spec.expected
            ]
            answer = [
                _parse_symbolic(
                    value,
                    variables=spec.variables,
                    functions=spec.functions,
                )
                for value in _split_container(given, "(", ")")
            ]
            normalized = "(" + ", ".join(sympy.sstr(value) for value in answer) + ")"
            matched = len(expected) == len(answer) and all(
                _equivalent(expected_value, answer_value)
                for expected_value, answer_value in zip(expected, answer, strict=True)
            )
        elif isinstance(spec, IntervalSetAnswerSpec):
            expected = [
                (
                    _parse_endpoint(interval.lower),
                    _parse_endpoint(interval.upper),
                    interval.lower_closed,
                    interval.upper_closed,
                )
                for interval in spec.expected
            ]
            answer = _parse_intervals(given)
            normalized = given.strip()
            matched = _intervals_equivalent(expected, answer)
        else:  # pragma: no cover - discriminated union makes this unreachable
            return _invalid("unsupported_answer_spec")
    except MathVerificationError:
        return _invalid("invalid_syntax")
    except Exception:  # noqa: BLE001 - expected-content bugs cannot escape verification
        return _invalid("verification_error")
    return VerificationResult(
        status=VerificationStatus.CORRECT if matched else VerificationStatus.INCORRECT,
        code="equivalent" if matched else "not_equivalent",
        normalized_form=normalized,
    )


def _worker_main(connection: Connection) -> None:
    # SymPy lazily initializes simplification internals. Warm them before the
    # worker advertises readiness so a learner's 250 ms equivalence budget is
    # not consumed by one-time process initialization.
    try:
        warm = sympy.Symbol("_warm")
        sympy.simplify((warm + warm) - 2 * warm)
    except BaseException:  # noqa: BLE001 - a broken worker must never serve
        connection.send({"ready": False})
        return
    connection.send({"ready": True})
    while True:
        try:
            payload = connection.recv()
        except EOFError:
            return
        if payload is None:
            return
        spec_data, given = payload
        try:
            spec = answer_spec_adapter.validate_python(spec_data)
            result = _verify_local(spec, given)
            connection.send(result.model_dump(mode="json"))
        except BaseException:  # noqa: BLE001 - isolate all worker failures
            connection.send(_invalid("worker_error").model_dump(mode="json"))


class _SupervisedWorker:
    """One serialized verifier process, replaced after a crash or timeout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None

    def _start(self, timeout_seconds: float) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_worker_main, args=(child,), daemon=True)
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        try:
            ready = parent.poll(timeout_seconds) and parent.recv() == {"ready": True}
        except (EOFError, BrokenPipeError, OSError):
            ready = False
        if not ready:
            self._stop()
            raise RuntimeError("verifier worker did not become ready")

    def _stop(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            connection.close()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.1)

    def verify(
        self,
        spec: AnswerSpec,
        given: str,
        timeout_seconds: float,
    ) -> VerificationResult:
        if not self._lock.acquire(timeout=timeout_seconds):
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                code="verifier_queue_timeout",
            )
        try:
            if self._process is None or not self._process.is_alive():
                self._stop()
                try:
                    self._start(WORKER_START_TIMEOUT_SECONDS)
                except (EOFError, BrokenPipeError, OSError, RuntimeError):
                    return VerificationResult(
                        status=VerificationStatus.TIMEOUT,
                        code="worker_start_timeout",
                    )
            assert self._connection is not None
            # Cold startup and bounded pool admission have separate allowances.
            # A worker receives the full authored equivalence budget once the
            # request starts executing.
            try:
                self._connection.send((spec.model_dump(mode="json"), given))
                if not self._connection.poll(timeout_seconds):
                    self._stop()
                    return VerificationResult(
                        status=VerificationStatus.TIMEOUT,
                        code="equivalence_timeout",
                    )
                return VerificationResult.model_validate(self._connection.recv())
            except (EOFError, BrokenPipeError, OSError):
                self._stop()
                return _invalid("worker_unavailable")
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
            self._stop()


class _SupervisedVerifierPool:
    """Bounded admission and scheduling over isolated verifier processes."""

    def __init__(
        self,
        settings: VerifierPoolSettings | None = None,
        *,
        worker_factory: Callable[[], _VerifierWorker] = _SupervisedWorker,
    ) -> None:
        self.settings = settings or VerifierPoolSettings()
        self._workers = tuple(
            worker_factory() for _ in range(self.settings.pool_size)
        )
        # LIFO keeps one process warm for serial traffic while still scaling
        # out to the remaining workers under concurrent load.
        self._available: queue.LifoQueue[_VerifierWorker] = queue.LifoQueue(
            maxsize=self.settings.pool_size
        )
        for worker in self._workers:
            self._available.put_nowait(worker)
        self._admission = threading.BoundedSemaphore(
            self.settings.pool_size + self.settings.queue_capacity
        )
        self._metrics_lock = threading.Lock()
        self._requests = 0
        self._invalid = 0
        self._timed_out = 0
        self._saturated = 0
        self._lifecycle = threading.Condition()
        self._active_calls = 0
        self._closed = False

    def _record(self, result: VerificationResult) -> None:
        with self._metrics_lock:
            self._requests += 1
            if result.status == VerificationStatus.INVALID:
                self._invalid += 1
            if result.status == VerificationStatus.TIMEOUT:
                self._timed_out += 1
            if result.retryable_overload:
                self._saturated += 1

    def _recorded(self, result: VerificationResult) -> VerificationResult:
        self._record(result)
        return result

    def verify(
        self,
        spec: AnswerSpec,
        given: str,
        timeout_seconds: float,
    ) -> VerificationResult:
        with self._lifecycle:
            if self._closed:
                return self._recorded(
                    VerificationResult(
                        status=VerificationStatus.TIMEOUT,
                        code="verifier_unavailable",
                    )
                )
            admitted = self._admission.acquire(blocking=False)
            if admitted:
                self._active_calls += 1
        if not admitted:
            return self._recorded(
                VerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    code="verifier_saturated",
                )
            )

        worker: _VerifierWorker | None = None
        try:
            try:
                worker = self._available.get(
                    timeout=self.settings.queue_wait_timeout_seconds
                )
            except queue.Empty:
                return self._recorded(
                    VerificationResult(
                        status=VerificationStatus.TIMEOUT,
                        code="verifier_queue_timeout",
                    )
                )
            return self._recorded(worker.verify(spec, given, timeout_seconds))
        finally:
            if worker is not None:
                self._available.put_nowait(worker)
            self._admission.release()
            with self._lifecycle:
                self._active_calls -= 1
                self._lifecycle.notify_all()

    def metrics_snapshot(self) -> VerifierMetricsSnapshot:
        with self._metrics_lock:
            return VerifierMetricsSnapshot(
                requests=self._requests,
                invalid=self._invalid,
                timed_out=self._timed_out,
                saturated=self._saturated,
            )

    def close(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            while self._active_calls:
                self._lifecycle.wait()
        for worker in self._workers:
            worker.close()


_WORKER_POOL_LOCK = threading.Lock()
_WORKER_POOL: _SupervisedVerifierPool | None = None


def _worker_pool() -> _SupervisedVerifierPool:
    global _WORKER_POOL
    with _WORKER_POOL_LOCK:
        if _WORKER_POOL is None:
            _WORKER_POOL = _SupervisedVerifierPool(
                VerifierPoolSettings.from_environment()
            )
        return _WORKER_POOL


def verifier_metrics_snapshot() -> VerifierMetricsSnapshot:
    """Return aggregate counters without starting worker processes."""
    with _WORKER_POOL_LOCK:
        pool = _WORKER_POOL
    if pool is None:
        return VerifierMetricsSnapshot(
            requests=0,
            invalid=0,
            timed_out=0,
            saturated=0,
        )
    return pool.metrics_snapshot()


def close_verifier_pool() -> None:
    """Close every verifier process; intended for application lifespan shutdown."""
    global _WORKER_POOL
    with _WORKER_POOL_LOCK:
        pool = _WORKER_POOL
        _WORKER_POOL = None
    if pool is not None:
        pool.close()


atexit.register(close_verifier_pool)


def verify_answer(
    answer_spec: AnswerSpec,
    given: str,
    *,
    timeout_seconds: float = 0.25,
    supervised: bool = True,
) -> VerificationResult:
    """Return a typed result, supervising symbolic work with a hard timeout."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not isinstance(given, str):
        return _invalid("answer_must_be_text")
    if len(given) > MAX_INPUT_LENGTH:
        return _invalid("input_too_long")
    if supervised:
        return _worker_pool().verify(answer_spec, given, timeout_seconds)
    return _verify_local(answer_spec, given)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", strip_assignment_prefix(text)).lower()


def _legacy_variables(*texts: str) -> list[str]:
    names: set[str] = set()
    for text in texts:
        names.update(_IDENTIFIER.findall(strip_assignment_prefix(text)))
    return sorted(names - set(_FUNCTIONS) - set(_CONSTANTS))


def check_answer(
    expected: str,
    given: str,
    checker: str = "sympy_equiv",
    tolerance: float | None = None,
) -> bool:
    """Backward-compatible v1 boolean check over the new explicit parser.

    Legacy unparseable prose retains normalized string comparison.  New code
    should use ``verify_answer`` so invalid and timed-out input cannot be
    mistaken for an ordinary incorrect response.
    """
    expected_value = strip_assignment_prefix(expected)
    given_value = strip_assignment_prefix(given)
    if checker == "numeric":
        spec: AnswerSpec = NumericAnswerSpec(
            expected=expected_value,
            tolerance=tolerance if tolerance is not None else 1e-6,
        )
    else:
        spec = SymbolicAnswerSpec(
            expected=expected_value,
            variables=_legacy_variables(expected_value, given_value),
            functions=list(_FUNCTIONS),
        )
    result = _verify_local(spec, given_value)
    if result.status == VerificationStatus.INVALID:
        return _normalize(expected) == _normalize(given)
    return result.correct
