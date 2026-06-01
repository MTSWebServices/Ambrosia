#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Multiple hypothesis testing corrections.

This module provides p-value adjustment procedures and confidence-interval
significance-level rescaling used by the ``Tester`` to control error rates when
several hypotheses (metric x group-pair combinations) are evaluated together.

Two families of methods are supported:

* **alpha-scaling** methods (``bonferroni``, ``sidak``) apply a constant
  per-test transformation. Both the reported p-values and the confidence
  intervals can be adjusted consistently, so for these methods the interval is
  rebuilt at a rescaled significance level.
* **step-wise / rank-based** methods (``holm``, ``holm-sidak``, ``fdr_bh``,
  ``fdr_by``, ``hommel``, ``simes-hochberg``) adjust p-values using the whole
  ordered vector of p-values. No simple per-hypothesis multiplicity-adjusted
  confidence interval exists for them, so intervals are left at the nominal
  level and only the p-values are corrected.

The adjusted p-values match ``statsmodels.stats.multitest.multipletests`` and
R's ``stats::p.adjust`` for the corresponding method names.
"""
from typing import Dict, List

import numpy as np

from ambrosia.tools.configs import MultipleTestingCorrection

# Canonical method names.
BONFERRONI: str = "bonferroni"
SIDAK: str = "sidak"
HOLM: str = "holm"
HOLM_SIDAK: str = "holm-sidak"
FDR_BH: str = "fdr_bh"
FDR_BY: str = "fdr_by"
HOMMEL: str = "hommel"
SIMES_HOCHBERG: str = "simes-hochberg"

# Methods whose multiplicity adjustment is a constant per-test scaling and can
# therefore also be reflected in confidence intervals via a rescaled alpha.
_CI_CORRECTABLE = frozenset({BONFERRONI, SIDAK})

# User-friendly aliases mapped to canonical method names.
_ALIASES: Dict[str, str] = {
    "benjamini-hochberg": FDR_BH,
    "benjamini_hochberg": FDR_BH,
    "bh": FDR_BH,
    "benjamini-yekutieli": FDR_BY,
    "benjamini_yekutieli": FDR_BY,
    "by": FDR_BY,
    "hochberg": SIMES_HOCHBERG,
}


def available_methods() -> List[str]:
    """
    Return the list of canonical correction method names.
    """
    return MultipleTestingCorrection.get_all_enum_values()


def normalize_method(method: str) -> str:
    """
    Map a user-provided method name (possibly an alias) to its canonical form.

    Unknown names are returned lower-cased and unchanged so that a single,
    consistent validation error can be raised downstream.

    Parameters
    ----------
    method : str
        Correction method name or alias.

    Returns
    -------
    str
        Canonical method name when recognised, otherwise the normalised input.
    """
    if not isinstance(method, str):
        return method
    key: str = method.strip().lower()
    return _ALIASES.get(key, key)


def validate_method(method: str) -> str:
    """
    Normalise and validate a correction method name.

    Parameters
    ----------
    method : str
        Correction method name or alias.

    Returns
    -------
    str
        Canonical method name.

    Raises
    ------
    ValueError
        If the (normalised) method is not supported.
    """
    canonical: str = normalize_method(method)
    MultipleTestingCorrection.raise_if_value_incorrect_enum(canonical)
    return canonical


def is_ci_correctable(method: str) -> bool:
    """
    Whether confidence intervals can be multiplicity-adjusted for ``method``.
    """
    return normalize_method(method) in _CI_CORRECTABLE


def alpha_for_confidence_interval(alpha: np.ndarray, method: str, n_hypotheses: int) -> np.ndarray:
    """
    Per-test significance level to use when building confidence intervals.

    For constant per-test ("alpha-scaling") methods the level is rescaled so the
    interval is consistent with the adjusted decision rule. For all other methods
    the nominal level is returned unchanged.

    Parameters
    ----------
    alpha : np.ndarray
        Nominal first type error level(s).
    method : str
        Correction method name (canonical or alias).
    n_hypotheses : int
        Number of simultaneously tested hypotheses (family size).

    Returns
    -------
    np.ndarray
        Significance level(s) for confidence-interval construction.
    """
    canonical: str = normalize_method(method)
    alpha = np.asarray(alpha, dtype=float)
    if canonical == BONFERRONI:
        return np.asarray(alpha / n_hypotheses)
    if canonical == SIDAK:
        return np.asarray(1.0 - np.power(1.0 - alpha, 1.0 / n_hypotheses))
    return alpha


def adjust_pvalues(pvalues, method: str) -> np.ndarray:
    """
    Adjust a family of p-values for multiple testing.

    The family size equals the full length of ``pvalues``. ``NaN`` entries are
    excluded from ranking but still counted towards the family size (they stand
    for family members whose test could not be computed) and are returned in
    place as ``NaN``, matching R's ``p.adjust``. All other values must lie in
    ``[0, 1]``.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values, one per hypothesis.
    method : str
        Correction method name (canonical or alias).

    Returns
    -------
    np.ndarray
        Adjusted p-values, clipped to ``[0, 1]``, in the original order.

    Raises
    ------
    ValueError
        If ``method`` is not supported, ``pvalues`` is not one-dimensional, or
        any non-``NaN`` p-value lies outside ``[0, 1]``.
    """
    canonical: str = validate_method(method)
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1:
        raise ValueError("pvalues must be a one-dimensional array")
    # NaN entries yield False in both comparisons and are tolerated here.
    if np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    adjusted: np.ndarray = np.full(pvalues.shape, np.nan, dtype=float)
    valid: np.ndarray = ~np.isnan(pvalues)
    if valid.any():
        adjusted[valid] = _adjust_valid(pvalues[valid], canonical, pvalues.size)
    return adjusted


def _adjust_valid(pvalues: np.ndarray, method: str, n: int) -> np.ndarray:
    """
    Apply a correction to the finite (non-NaN) members of a family of size ``n``.

    ``pvalues`` holds the ``k <= n`` present (non-NaN) p-values; the remaining
    ``n - k`` members were not computed but still count towards the family size.
    """
    k: int = len(pvalues)
    if k == 0:
        return pvalues.copy()
    if method == BONFERRONI:
        return np.minimum(pvalues * n, 1.0)
    if method == SIDAK:
        # Sidak: 1 - (1 - p)^n, computed directly. The result matches statsmodels
        # (which uses the equivalent expm1/log1p form) to floating-point precision
        # while avoiding its divide-by-zero warning at p = 1.
        return np.minimum(1.0 - np.power(1.0 - pvalues, n), 1.0)

    # Rank-based procedures operate on the ascending-sorted p-value vector; the
    # ranks run over the k present values while the family size stays n.
    order: np.ndarray = np.argsort(pvalues, kind="mergesort")
    inverse: np.ndarray = np.empty(k, dtype=int)
    inverse[order] = np.arange(k)
    p_sorted: np.ndarray = pvalues[order]
    ranks: np.ndarray = np.arange(1, k + 1)  # 1-based ascending ranks

    if method == HOLM:
        adjusted = np.maximum.accumulate((n - ranks + 1) * p_sorted)
    elif method == HOLM_SIDAK:
        adjusted = np.maximum.accumulate(1.0 - np.power(1.0 - p_sorted, n - ranks + 1))
    elif method == SIMES_HOCHBERG:
        adjusted = np.minimum.accumulate((((n - ranks + 1) * p_sorted)[::-1]))[::-1]
    elif method == FDR_BH:
        adjusted = np.minimum.accumulate(((n / ranks) * p_sorted)[::-1])[::-1]
    elif method == FDR_BY:
        c_n: float = float(np.sum(1.0 / np.arange(1, n + 1)))
        adjusted = np.minimum.accumulate(((c_n * n / ranks) * p_sorted)[::-1])[::-1]
    elif method == HOMMEL:
        adjusted = _hommel_sorted(p_sorted, n)
    else:  # pragma: no cover - guarded by validate_method
        raise ValueError(f"Unsupported correction method: {method}")

    adjusted = np.minimum(adjusted, 1.0)
    return adjusted[inverse]


def _hommel_sorted(p_sorted: np.ndarray, n: int) -> np.ndarray:
    """
    Hommel (1988) adjusted p-values for an ascending-sorted p-value vector.

    Implements the closed form reproduced in Wright (1992); matches the
    ``hommel`` procedure of ``statsmodels`` and R's ``p.adjust``. When the family
    size ``n`` exceeds the number of present p-values (some members are NaN), the
    missing members are padded with the neutral value ``1.0`` (as in R) and only
    the present positions are returned.
    """
    k: int = len(p_sorted)
    full: np.ndarray = p_sorted if n == k else np.concatenate([p_sorted, np.ones(n - k)])
    adjusted: np.ndarray = full.copy()
    for size in range(n, 1, -1):
        index = np.arange(1, size + 1)
        cim = np.min(size * full[n - size :] / index)
        adjusted[n - size :] = np.maximum(adjusted[n - size :], cim)
        adjusted[: n - size] = np.maximum(adjusted[: n - size], np.minimum(size * full[: n - size], cim))
    return adjusted[:k]
