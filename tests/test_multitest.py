from typing import List

import numpy as np
import pytest

from ambrosia.tools import multitest as mt
from ambrosia.tools.configs import MultipleTestingCorrection

ALL_METHODS: List[str] = [
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "fdr_bh",
    "fdr_by",
    "hommel",
    "simes-hochberg",
]

# statsmodels is used purely as a cross-check oracle for the native implementation.
try:
    from statsmodels.stats.multitest import multipletests

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - statsmodels is a hard dependency in practice
    _HAS_STATSMODELS = False

requires_statsmodels = pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels is not installed")


# Sorted reference vector with hand-computed expectations (family size m = 5).
HANDP = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
HAND_EXPECTED = {
    # bonferroni: min(m * p, 1)
    "bonferroni": [0.005, 0.04, 0.195, 0.205, 1.0],
    # holm step-down: cummax of (m - i + 1) * p_(i)
    "holm": [0.005, 0.032, 0.117, 0.117, 0.9],
    # Benjamini-Hochberg step-up: cummin (from top) of (m / i) * p_(i)
    "fdr_bh": [0.005, 0.02, 0.05125, 0.05125, 0.9],
    # Hochberg step-up: cummin (from top) of (m - i + 1) * p_(i)
    "simes-hochberg": [0.005, 0.032, 0.082, 0.082, 0.9],
}

ORACLE_VECTORS = [
    np.array([0.001, 0.008, 0.039, 0.041, 0.9]),
    np.array([0.04, 0.04, 0.04, 0.04]),  # ties
    np.array([0.5]),  # singleton
    np.array([0.0, 0.0, 1.0, 1.0]),  # boundary values
    np.array([1e-9, 0.2, 0.7, 0.999, 0.5, 0.03]),
    np.random.default_rng(7).uniform(0, 1, 37),
    np.random.default_rng(8).uniform(0, 1, 128),
]


@pytest.mark.unit
@pytest.mark.parametrize("method", list(HAND_EXPECTED))
def test_adjust_matches_hand_computed(method):
    """
    Native output equals independently hand-computed values for clean methods.
    """
    got = mt.adjust_pvalues(HANDP, method)
    np.testing.assert_allclose(got, HAND_EXPECTED[method], atol=1e-9)


@requires_statsmodels
@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("vec_id", range(len(ORACLE_VECTORS)))
def test_adjust_matches_statsmodels(method, vec_id):
    """
    Native output matches statsmodels.multipletests for every method and vector.
    """
    vec = ORACLE_VECTORS[vec_id]
    # statsmodels' sidak/holm-sidak use log1p(-p), which warns at p == 1; harmless here.
    with np.errstate(divide="ignore", invalid="ignore"):
        _, expected, _, _ = multipletests(vec, method=method)
    got = mt.adjust_pvalues(vec, method)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
def test_adjusted_within_unit_interval_and_at_least_raw(method):
    """
    Adjusted p-values stay in [0, 1] and are never smaller than the raw values.
    """
    p = np.random.default_rng(123).uniform(0, 1, 60)
    adj = mt.adjust_pvalues(p, method)
    assert np.all(adj >= -1e-12)
    assert np.all(adj <= 1 + 1e-12)
    assert np.all(adj >= p - 1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
def test_monotone_in_rank(method):
    """
    Adjusted p-values are non-decreasing in the rank of the raw p-value.
    """
    p = np.random.default_rng(321).uniform(0, 1, 40)
    adj = mt.adjust_pvalues(p, method)
    order = np.argsort(p, kind="mergesort")
    assert np.all(np.diff(adj[order]) >= -1e-12)


@pytest.mark.unit
def test_method_conservativeness_ordering():
    """
    Known dominance relations between the supported procedures.

    A fixed vector with small leading p-values is used so each inequality is
    strict somewhere: a degenerate vector (everything saturating to 1.0) would
    make the relations vacuously true and could hide, e.g., a Holm step-up /
    step-down swap or a Hommel / Hochberg mix-up.
    """
    p = np.array([0.001, 0.005, 0.012, 0.025, 0.04, 0.06, 0.15, 0.35, 0.6, 0.85])
    res = {m: mt.adjust_pvalues(p, m) for m in ALL_METHODS}
    tol = 1e-12

    def dominates(smaller: np.ndarray, larger: np.ndarray) -> None:
        # ``smaller`` <= ``larger`` everywhere and strictly smaller somewhere.
        assert np.all(smaller <= larger + tol)
        assert np.any(smaller < larger - tol)

    # FWER step-down family.
    dominates(res["holm"], res["bonferroni"])
    dominates(res["sidak"], res["bonferroni"])
    dominates(res["holm-sidak"], res["holm"])
    # Step-up dominates step-down; Hommel dominates Hochberg.
    dominates(res["simes-hochberg"], res["holm"])
    dominates(res["hommel"], res["simes-hochberg"])
    # FDR control is less conservative than Holm; BY is more conservative than BH.
    dominates(res["fdr_bh"], res["holm"])
    dominates(res["fdr_bh"], res["fdr_by"])


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
def test_single_hypothesis_is_identity(method):
    """
    With one hypothesis every method returns the raw p-value unchanged.
    """
    for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
        np.testing.assert_allclose(mt.adjust_pvalues(np.array([p]), method), [p], atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
def test_nan_preserved_and_counted_in_family(method):
    """
    Entries that are NaN pass through in place but still count toward the family
    size (matching R's p.adjust); only the present p-values are ranked. The
    present positions therefore match treating the NaN slot as a neutral p = 1.
    """
    p = np.array([0.01, np.nan, 0.02, 0.04])
    adj = mt.adjust_pvalues(p, method)
    assert np.isnan(adj[1])
    expected_full = mt.adjust_pvalues(np.array([0.01, 1.0, 0.02, 0.04]), method)
    np.testing.assert_allclose(adj[[0, 2, 3]], expected_full[[0, 2, 3]], atol=1e-12)


@pytest.mark.unit
def test_all_nan_returns_all_nan():
    adj = mt.adjust_pvalues(np.array([np.nan, np.nan]), "holm")
    assert np.all(np.isnan(adj))


@pytest.mark.unit
def test_empty_input_returns_empty():
    adj = mt.adjust_pvalues(np.array([]), "fdr_bh")
    assert adj.shape == (0,)


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_METHODS)
def test_permutation_equivariance(method):
    """
    Adjusting then permuting equals permuting then adjusting.
    """
    rng = np.random.default_rng(55)
    p = rng.uniform(0, 1, 25)
    perm = rng.permutation(25)
    adj = mt.adjust_pvalues(p, method)
    adj_perm = mt.adjust_pvalues(p[perm], method)
    np.testing.assert_allclose(adj_perm, adj[perm], atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[-0.01, 0.2], [0.3, 1.5], [np.inf, 0.1]])
def test_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        mt.adjust_pvalues(np.array(bad), "holm")


@pytest.mark.unit
def test_two_dimensional_raises():
    with pytest.raises(ValueError):
        mt.adjust_pvalues(np.array([[0.1, 0.2], [0.3, 0.4]]), "holm")


@pytest.mark.unit
def test_invalid_method_raises():
    with pytest.raises(ValueError):
        mt.adjust_pvalues(np.array([0.1, 0.2]), "definitely-not-a-method")


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("benjamini-hochberg", "fdr_bh"),
        ("BH", "fdr_bh"),
        ("benjamini-yekutieli", "fdr_by"),
        ("BY", "fdr_by"),
        ("hochberg", "simes-hochberg"),
        ("  Holm  ", "holm"),
    ],
)
def test_alias_normalization(alias, canonical):
    assert mt.normalize_method(alias) == canonical
    np.testing.assert_allclose(mt.adjust_pvalues(HANDP, alias), mt.adjust_pvalues(HANDP, canonical), atol=1e-12)


@pytest.mark.unit
def test_validate_method_returns_canonical():
    assert mt.validate_method("BH") == "fdr_bh"
    with pytest.raises(ValueError):
        mt.validate_method("nope")


@pytest.mark.unit
def test_available_methods_matches_enum():
    assert set(mt.available_methods()) == {member.value for member in MultipleTestingCorrection}
    assert set(mt.available_methods()) == set(ALL_METHODS)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,correctable",
    [
        ("bonferroni", True),
        ("sidak", True),
        ("holm", False),
        ("holm-sidak", False),
        ("fdr_bh", False),
        ("fdr_by", False),
        ("hommel", False),
        ("simes-hochberg", False),
    ],
)
def test_is_ci_correctable(method, correctable):
    assert mt.is_ci_correctable(method) is correctable


@pytest.mark.unit
def test_alpha_for_confidence_interval():
    alpha = np.array([0.05, 0.1])
    m = 4
    np.testing.assert_allclose(mt.alpha_for_confidence_interval(alpha, "bonferroni", m), alpha / m)
    np.testing.assert_allclose(mt.alpha_for_confidence_interval(alpha, "sidak", m), 1 - (1 - alpha) ** (1 / m))
    for method in ["holm", "holm-sidak", "fdr_bh", "fdr_by", "hommel", "simes-hochberg"]:
        np.testing.assert_allclose(mt.alpha_for_confidence_interval(alpha, method, m), alpha)
    # Sidak's interval level sits strictly between Bonferroni and the nominal level.
    sidak_alpha = mt.alpha_for_confidence_interval(alpha, "sidak", m)
    assert np.all(sidak_alpha > alpha / m) and np.all(sidak_alpha < alpha)
