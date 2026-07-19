"""Restricted parsing and answer checking."""

import pytest

from tutor.verify.checker import MathVerificationError, check_answer, parse_restricted


def test_symbolic_equivalence_with_caret_and_implicit_multiplication():
    assert check_answer("5x^4", "5*x**4")
    assert check_answer("2x cos(x^2)", "2*x*cos(x^2)")
    assert check_answer("1/2", "0.5")


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
