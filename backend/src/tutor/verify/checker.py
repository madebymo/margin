"""Restricted math parsing and answer checking.

Generated and student-entered math is parsed through a character whitelist
(no underscores, quotes, or brackets — which blocks dunder/attribute tricks)
and a fixed function table. Unknown names become inert SymPy symbols. This is
one validator inside the correctness gate, never arbitrary evaluation.
"""

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_SAFE = re.compile(r"^[0-9a-zA-Z+\-*/^(). ,]*$")
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
_ALLOWED_LOCALS = {
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
    "pi": sympy.pi,
    "Abs": sympy.Abs,
}


class MathVerificationError(ValueError):
    """Raised when text cannot be safely parsed as math."""


def strip_assignment_prefix(text: str) -> str:
    """Reduce ``u = x^2`` to ``x^2`` (the rightmost ``=`` wins)."""
    if "=" in text:
        return text.rsplit("=", 1)[1].strip()
    return text.strip()


def parse_restricted(text: str) -> sympy.Expr:
    """Parse text into a SymPy expression under the whitelist, or raise."""
    candidate = strip_assignment_prefix(text)
    if not candidate or not _SAFE.match(candidate):
        raise MathVerificationError(f"cannot safely parse: {text!r}")
    try:
        return parse_expr(candidate, transformations=_TRANSFORMS, local_dict=dict(_ALLOWED_LOCALS))
    except Exception as exc:  # noqa: BLE001 — any parser failure is a verification failure
        raise MathVerificationError(f"cannot parse: {text!r}") from exc


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", strip_assignment_prefix(text)).lower()


def check_answer(
    expected: str,
    given: str,
    checker: str = "sympy_equiv",
    tolerance: float | None = None,
) -> bool:
    """Judge a student answer against the expected value.

    ``sympy_equiv`` checks symbolic equivalence; ``numeric`` compares within a
    tolerance. When either side cannot be parsed safely, falls back to a
    normalized string comparison (never arbitrary evaluation).
    """
    try:
        expected_expr = parse_restricted(expected)
        given_expr = parse_restricted(given)
    except MathVerificationError:
        return _normalize(expected) == _normalize(given)

    if checker == "numeric":
        try:
            delta = abs(float(expected_expr.evalf()) - float(given_expr.evalf()))
        except (TypeError, ValueError):
            return False
        return delta <= (tolerance if tolerance is not None else 1e-6)

    try:
        difference = sympy.simplify(expected_expr - given_expr)
    except Exception:  # noqa: BLE001 — treat simplification failure as mismatch
        return False
    if difference == 0:
        return True
    return difference.equals(0) is True
