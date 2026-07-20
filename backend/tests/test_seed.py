"""Seed data integrity: node count, id hygiene, coverage completeness, roots."""

import re

import pytest

from tutor.graph import service
from tutor.content.item_bank import load_item_bank, validate_item_bank
from tutor.schemas.kc import KC_ID_PATTERN, GraphDocument
from tutor.seed.load_seed import load_coverage, load_graph, validate_coverage


@pytest.fixture(scope="module")
def seed() -> GraphDocument:
    return load_graph()


@pytest.fixture(scope="module")
def coverage() -> dict:
    return load_coverage()


def test_seed_has_exactly_40_nodes(seed):
    assert len(seed.nodes) == 40


def test_node_ids_match_pattern(seed):
    for node in seed.nodes:
        assert re.match(KC_ID_PATTERN, node.id), node.id


def test_coverage_matrix_is_valid_and_complete(seed, coverage):
    assert validate_coverage(seed, coverage) == []


def test_every_entry_has_widget_and_fallback(coverage):
    for kc_id, entry in coverage.items():
        assert entry["widget_types"], kc_id
        assert entry["text_fallback"] is True, kc_id


def test_computational_kcs_measure_production(coverage):
    for kc_id in [
        "kc.der.power_rule",
        "kc.der.chain_rule",
        "kc.int.antiderivatives",
        "kc.int.u_substitution",
    ]:
        assert coverage[kc_id]["measures"] == "production", kc_id


def test_roots_include_expected_foundations(seed):
    expected = {
        "kc.alg.arith_fractions",
        "kc.alg.exponent_rules",
        "kc.alg.polynomial_ops",
        "kc.alg.solve_linear",
        "kc.fun.function_notation",
    }
    assert expected <= set(service.roots(seed))


def test_packaged_item_bank_is_valid_unreleased_draft(seed):
    bank = load_item_bank()

    assert bank.released_kcs == []
    assert {item.review_status.value for item in bank.items} == {"draft"}
    assert all(item.provenance.reviewed_by is None for item in bank.items)
    assert validate_item_bank(bank, seed) == []
