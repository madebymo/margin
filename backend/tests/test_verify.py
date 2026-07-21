"""Restricted parsing and answer checking."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tutor.schemas.assessment import (
    AntiderivativeAnswerSpec,
    ChoiceAnswerSpec,
    FiniteSetAnswerSpec,
    IntervalSetAnswerSpec,
    NumericAnswerSpec,
    OrderedTupleAnswerSpec,
    SymbolicAnswerSpec,
)
from tutor.verify.checker import (
    MathVerificationError,
    VerifierPoolSettings,
    _SupervisedWorker,
    _SupervisedVerifierPool,
    VerificationResult,
    VerificationStatus,
    check_answer,
    parse_restricted,
    verify_answer,
)


def test_symbolic_equivalence_with_caret_and_implicit_multiplication():
    assert check_answer("5x^4", "5*x**4")
    assert check_answer("2x cos(x^2)", "2*x*cos(x^2)")
    assert check_answer("1/2", "0.5")
    assert check_answer("sec^2(x)", "1/cos(x)^2")


def test_non_equivalent_rejected():
    assert not check_answer("5x^4", "4x^5")
    assert not check_answer("sin(x)", "cos(x)")


def test_numeric_checker_with_tolerance():
    assert check_answer("15/8", "1.875", checker="numeric")
    assert not check_answer("4", "4.2", checker="numeric")
    assert check_answer("4", "4.05", checker="numeric", tolerance=0.1)


def test_assignment_prefix_is_stripped():
    assert check_answer("x^2", "u = x^2")
    assert check_answer("y = x^2", "x^2")


def test_unsafe_input_raises_in_parser():
    with pytest.raises(MathVerificationError):
        parse_restricted("__import__('os').system('ls')")
    with pytest.raises(MathVerificationError):
        parse_restricted("x_1 + 2")
    with pytest.raises(MathVerificationError):
        parse_restricted("")


def test_unparseable_text_falls_back_to_string_compare():
    text = "Area under v(t) on [0, 4] gives distance traveled"
    assert check_answer(text, text)
    assert check_answer(text, "  area under v(t) on [0, 4] gives DISTANCE traveled ")
    assert not check_answer(text, "something else entirely")


@pytest.mark.parametrize(
    "payload",
    [
        "factorial(5)",
        "gamma(2)",
        "Integer(2)",
        "integrate(x)",
        "__import__(os)",
    ],
)
def test_sympy_globals_and_unknown_calls_are_not_exposed(payload):
    with pytest.raises(MathVerificationError):
        parse_restricted(payload)


def test_comparison_is_not_misread_as_assignment():
    with pytest.raises(MathVerificationError):
        parse_restricted("f(x) != 2")
    assert not check_answer("f(x) != 2", "2")


@pytest.mark.parametrize(
    "payload",
    [
        "x^21",
        "1e21",
        "1e-21",
        "1e" + "9" * 200,
        "x" * 257,
        "+".join(["x"] * 130),
    ],
)
def test_parser_resource_limits(payload):
    with pytest.raises(MathVerificationError):
        parse_restricted(payload)


def test_parser_depth_limit_measures_ast_not_precedence_call_stack():
    natural_quotient_rule_form = "(2*x*(x+2)-(x^2+1))/(x+2)^2"

    parsed = parse_restricted(
        natural_quotient_rule_form,
        allowed_variables=["x"],
        allowed_functions=[],
        allowed_assignment_lhs=None,
    )

    assert str(parsed)
    with pytest.raises(MathVerificationError, match="maximum depth 16"):
        parse_restricted("+".join(["x"] * 17))


def test_factored_quotient_rule_answer_is_accepted_by_supervised_verifier():
    spec = SymbolicAnswerSpec(
        expected="(x^2+4*x-1)/(x^2+4*x+4)",
        variables=["x"],
    )

    verdict = verify_answer(
        spec,
        "(2*x*(x+2)-(x^2+1))/(x+2)^2",
    )

    assert verdict.status == VerificationStatus.CORRECT


def test_typed_symbolic_result_and_exact_assignment_contract():
    spec = SymbolicAnswerSpec(
        expected="2*x",
        variables=["x"],
        assignment_lhs="f(x)",
    )
    assert verify_answer(spec, "f(x) = x+x", supervised=False).status == "correct"
    wrong_lhs = verify_answer(spec, "g(x) = 2*x", supervised=False)
    assert wrong_lhs.status == VerificationStatus.INVALID
    assert wrong_lhs.code == "invalid_syntax"


def test_typed_answer_specs():
    cases = [
        (NumericAnswerSpec(expected="3/2"), "1.5"),
        (FiniteSetAnswerSpec(expected=["2", "3"]), "{3, 2}"),
        (
            IntervalSetAnswerSpec(
                expected=[
                    {
                        "lower": "-inf",
                        "upper": "2",
                        "lower_closed": False,
                        "upper_closed": False,
                    }
                ]
            ),
            "(-inf, 2)",
        ),
        (
            OrderedTupleAnswerSpec(expected=["1", "sqrt(2)"], functions=["sqrt"]),
            "(1, sqrt(2))",
        ),
        (
            AntiderivativeAnswerSpec(
                expected="x^3",
                variable="x",
                variables=[],
            ),
            "x^3 + C",
        ),
        (
            ChoiceAnswerSpec(option_ids=["r1", "r2"], expected_choice_id="r2"),
            "r2",
        ),
    ]
    for spec, answer in cases:
        verdict = verify_answer(spec, answer, supervised=False)
        assert verdict.status == VerificationStatus.CORRECT, (spec, verdict)


def test_finite_set_uses_mathematical_set_semantics_for_duplicates():
    singleton = FiniteSetAnswerSpec(expected=["1"])
    duplicated_expected = FiniteSetAnswerSpec(
        expected=["x", "2*x/2"],
        variables=["x"],
    )

    duplicate_submission = verify_answer(singleton, "{1, 1}", supervised=False)
    deduplicated_submission = verify_answer(
        duplicated_expected,
        "{x}",
        supervised=False,
    )

    assert duplicate_submission.status == VerificationStatus.CORRECT
    assert duplicate_submission.normalized_form == "{1}"
    assert deduplicated_submission.status == VerificationStatus.CORRECT


def test_interval_answers_compare_the_represented_set():
    spec = IntervalSetAnswerSpec(
        expected=[
            {
                "lower": "0",
                "upper": "2",
                "lower_closed": False,
                "upper_closed": False,
            }
        ]
    )

    assert (
        verify_answer(spec, "(0,1) U [1,2)", supervised=False).status
        == VerificationStatus.CORRECT
    )
    assert (
        verify_answer(spec, "(0,2) U (0,2)", supervised=False).status
        == VerificationStatus.CORRECT
    )
    assert (
        verify_answer(spec, "(2,0)", supervised=False).status
        == VerificationStatus.INVALID
    )


def test_numeric_and_interval_specs_reject_functions_they_cannot_declare():
    numeric = verify_answer(
        NumericAnswerSpec(expected="2"),
        "sqrt(4)",
        supervised=False,
    )
    interval = verify_answer(
        IntervalSetAnswerSpec(
            expected=[
                {
                    "lower": "0",
                    "upper": "2",
                    "lower_closed": False,
                    "upper_closed": False,
                }
            ]
        ),
        "(0, sqrt(4))",
        supervised=False,
    )

    assert numeric.status == VerificationStatus.INVALID
    assert interval.status == VerificationStatus.INVALID


def test_typed_invalid_is_not_an_incorrect_mastery_observation():
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    verdict = verify_answer(spec, "x[0]", supervised=False)
    assert verdict.status == VerificationStatus.INVALID
    assert verdict.correct is False


def test_undefined_symbolic_form_is_invalid_not_merely_incorrect():
    spec = SymbolicAnswerSpec(expected="x", variables=["x"])
    verdict = verify_answer(spec, "1/0", supervised=False)

    assert verdict.status == VerificationStatus.INVALID


def test_undefined_constant_cannot_make_antiderivative_pass():
    spec = AntiderivativeAnswerSpec(
        expected="x^2",
        variable="x",
        variables=[],
    )
    verdict = verify_answer(spec, "x^2 + 1/0", supervised=False)

    assert verdict.status == VerificationStatus.INVALID


def test_supervised_verifier_process_round_trip():
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    assert verify_answer(spec, "x*x").status == VerificationStatus.CORRECT


def test_rejected_scientific_exponent_does_not_poison_the_worker():
    spec = NumericAnswerSpec(expected="1")
    rejected = verify_answer(spec, "1e" + "9" * 200)
    recovered = verify_answer(spec, "1")

    assert rejected.status == VerificationStatus.INVALID
    assert recovered.status == VerificationStatus.CORRECT


def test_input_length_limit_applies_to_choice_answers_too():
    spec = ChoiceAnswerSpec(option_ids=["a", "b"], expected_choice_id="a")
    verdict = verify_answer(spec, "a" * 257, supervised=False)

    assert verdict.status == VerificationStatus.INVALID
    assert verdict.code == "input_too_long"


def test_supervisor_returns_timeout_and_discards_stuck_worker():
    class NeverReplies:
        def send(self, payload):
            self.payload = payload

        def poll(self, timeout):
            return False

        def close(self):
            self.closed = True

    class RunningProcess:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, timeout):
            return None

        def kill(self):
            self.alive = False

    worker = _SupervisedWorker()
    worker._connection = NeverReplies()
    worker._process = RunningProcess()
    verdict = worker.verify(
        SymbolicAnswerSpec(expected="x^2", variables=["x"]),
        "x*x",
        timeout_seconds=0.001,
    )
    assert verdict.status == VerificationStatus.TIMEOUT
    assert worker._connection is None
    assert worker._process is None


def test_supervisor_bounds_time_waiting_for_a_busy_worker():
    worker = _SupervisedWorker()
    worker._lock.acquire()
    try:
        verdict = worker.verify(
            SymbolicAnswerSpec(expected="x^2", variables=["x"]),
            "x*x",
            timeout_seconds=0.001,
        )
    finally:
        worker._lock.release()

    assert verdict.status == VerificationStatus.TIMEOUT
    assert verdict.code == "verifier_queue_timeout"
    assert verdict.retryable_overload


def test_verifier_pool_settings_load_bounded_environment_values():
    settings = VerifierPoolSettings.from_environment(
        {
            "TUTOR_VERIFIER_POOL_SIZE": "3",
            "TUTOR_VERIFIER_QUEUE_CAPACITY": "17",
            "TUTOR_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS": "0.4",
        }
    )

    assert settings == VerifierPoolSettings(
        pool_size=3,
        queue_capacity=17,
        queue_wait_timeout_seconds=0.4,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TUTOR_VERIFIER_POOL_SIZE", "0"),
        ("TUTOR_VERIFIER_QUEUE_CAPACITY", "-1"),
        ("TUTOR_VERIFIER_QUEUE_WAIT_TIMEOUT_SECONDS", "forever"),
    ],
)
def test_verifier_pool_settings_reject_invalid_environment(name, value):
    with pytest.raises(ValueError, match=name):
        VerifierPoolSettings.from_environment({name: value})


class _BlockingVerifierWorker:
    def __init__(self, release: threading.Event) -> None:
        self.started = threading.Event()
        self.release = release
        self.closed = False

    def verify(self, spec, given, timeout_seconds):
        self.started.set()
        assert self.release.wait(timeout=2)
        return VerificationResult(
            status=VerificationStatus.CORRECT,
            code="equivalent",
            normalized_form="x**2",
        )

    def close(self):
        self.closed = True


def test_pool_executes_up_to_configured_worker_count_concurrently():
    release = threading.Event()
    workers: list[_BlockingVerifierWorker] = []

    def factory():
        worker = _BlockingVerifierWorker(release)
        workers.append(worker)
        return worker

    pool = _SupervisedVerifierPool(
        VerifierPoolSettings(pool_size=2, queue_capacity=0),
        worker_factory=factory,
    )
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(pool.verify, spec, "x*x", 0.25)
                for _ in range(2)
            ]
            assert all(worker.started.wait(timeout=1) for worker in workers)
            release.set()
            assert all(
                future.result(timeout=1).status == VerificationStatus.CORRECT
                for future in futures
            )
    finally:
        release.set()
        pool.close()


def test_full_pool_rejects_before_execution_with_retryable_overload():
    release = threading.Event()
    workers: list[_BlockingVerifierWorker] = []

    def factory():
        worker = _BlockingVerifierWorker(release)
        workers.append(worker)
        return worker

    pool = _SupervisedVerifierPool(
        VerifierPoolSettings(pool_size=1, queue_capacity=0),
        worker_factory=factory,
    )
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(pool.verify, spec, "x*x", 0.25)
            assert workers[0].started.wait(timeout=1)

            overloaded = pool.verify(spec, "student input is never recorded", 0.25)

            assert overloaded.status == VerificationStatus.TIMEOUT
            assert overloaded.code == "verifier_saturated"
            assert overloaded.retryable_overload
            snapshot = pool.metrics_snapshot()
            assert snapshot.requests == 1
            assert snapshot.timed_out == 1
            assert snapshot.saturated == 1
            assert "student" not in repr(snapshot)
            release.set()
            assert active.result(timeout=1).correct
    finally:
        release.set()
        pool.close()


def test_admitted_request_has_a_bounded_queue_wait():
    release = threading.Event()
    workers: list[_BlockingVerifierWorker] = []

    def factory():
        worker = _BlockingVerifierWorker(release)
        workers.append(worker)
        return worker

    pool = _SupervisedVerifierPool(
        VerifierPoolSettings(
            pool_size=1,
            queue_capacity=1,
            queue_wait_timeout_seconds=0.01,
        ),
        worker_factory=factory,
    )
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(pool.verify, spec, "x*x", 0.25)
            assert workers[0].started.wait(timeout=1)

            queued = pool.verify(spec, "x*x", 0.25)

            assert queued.status == VerificationStatus.TIMEOUT
            assert queued.code == "verifier_queue_timeout"
            assert queued.retryable_overload
            release.set()
            assert active.result(timeout=1).correct
    finally:
        release.set()
        pool.close()


def test_pool_timeout_reaps_only_the_affected_worker():
    class NeverReplies:
        def send(self, payload):
            self.payload = payload

        def poll(self, timeout):
            return False

        def close(self):
            self.closed = True

    class Replies:
        def send(self, payload):
            self.payload = payload

        def poll(self, timeout):
            return True

        def recv(self):
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                code="equivalent",
            ).model_dump(mode="json")

        def close(self):
            self.closed = True

    class RunningProcess:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, timeout):
            return None

        def kill(self):
            self.alive = False

    healthy = _SupervisedWorker()
    healthy_process = RunningProcess()
    healthy._connection = Replies()
    healthy._process = healthy_process
    stuck = _SupervisedWorker()
    stuck._connection = NeverReplies()
    stuck._process = RunningProcess()
    workers = iter((healthy, stuck))
    pool = _SupervisedVerifierPool(
        VerifierPoolSettings(pool_size=2, queue_capacity=0),
        worker_factory=lambda: next(workers),
    )
    spec = SymbolicAnswerSpec(expected="x^2", variables=["x"])
    try:
        timed_out = pool.verify(spec, "x*x", 0.001)

        assert timed_out.code == "equivalence_timeout"
        assert not timed_out.retryable_overload
        assert stuck._connection is None
        assert stuck._process is None
        assert healthy._process is healthy_process
        assert healthy_process.is_alive()

        recovered = pool.verify(spec, "x*x", 0.25)

        assert recovered.correct
        assert healthy._process is healthy_process
        assert healthy_process.is_alive()
    finally:
        pool.close()


def test_pool_crash_reaps_only_the_affected_worker():
    class CrashesOnReceive:
        def send(self, payload):
            self.payload = payload

        def poll(self, timeout):
            return True

        def recv(self):
            raise EOFError

        def close(self):
            self.closed = True

    class RunningProcess:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

        def join(self, timeout):
            return None

        def kill(self):
            self.alive = False

    healthy = _SupervisedWorker()
    healthy_process = RunningProcess()
    healthy._process = healthy_process
    crashed = _SupervisedWorker()
    crashed._connection = CrashesOnReceive()
    crashed._process = RunningProcess()
    workers = iter((healthy, crashed))
    pool = _SupervisedVerifierPool(
        VerifierPoolSettings(pool_size=2, queue_capacity=0),
        worker_factory=lambda: next(workers),
    )
    try:
        result = pool.verify(
            SymbolicAnswerSpec(expected="x^2", variables=["x"]),
            "x*x",
            0.01,
        )

        assert result.status == VerificationStatus.INVALID
        assert result.code == "worker_unavailable"
        assert crashed._connection is None
        assert crashed._process is None
        assert healthy._process is healthy_process
        assert healthy_process.is_alive()
    finally:
        pool.close()
