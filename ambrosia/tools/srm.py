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
Sample Ratio Mismatch (SRM) diagnostics.

A Sample Ratio Mismatch occurs when the observed experimental group sizes
deviate from the ratios planned for the assignment procedure. It usually
indicates a broken randomization, filtering or logging pipeline, and any
test results computed on such groups may be unreliable.

The check is a chi-square goodness-of-fit test of the observed group sizes
against the expected ratios. Following the industry practice, a strict
default significance level (``alpha=0.0005``) is used so that the alarm
fires only on strong evidence of a mismatch.
"""
from typing import Any, Dict, Optional

import pandas as pd
import scipy.stats as sps

from ambrosia import types

DEFAULT_SRM_ALPHA: float = 0.0005


def check_srm_from_counts(
    observed: Dict[Any, int],
    expected_ratios: Optional[Dict[Any, float]] = None,
    alpha: float = DEFAULT_SRM_ALPHA,
) -> Dict[str, Any]:
    """
    Run a Sample Ratio Mismatch check on precomputed group sizes.

    Parameters
    ----------
    observed : Dict[Any, int]
        Mapping from group label to the observed group size.
    expected_ratios : Dict[Any, float], optional
        Mapping from group label to the expected share of objects.
        Ratios may be given in any positive scale (e.g. ``{"A": 9, "B": 1}``)
        and are normalized internally. If ``None``, equal group sizes
        are expected.
    alpha : float, default: ``0.0005``
        Significance level for the chi-square goodness-of-fit test.

    Returns
    -------
    result : Dict[str, Any]
        Dictionary with ``observed`` sizes, ``expected`` sizes, chi-square
        ``pvalue``, the used ``alpha`` and the boolean ``srm_detected``.
    """
    if len(observed) < 2:
        raise ValueError("SRM check requires at least two groups")
    if any(size < 0 for size in observed.values()):
        raise ValueError("Observed group sizes must be non-negative")
    total: int = sum(observed.values())
    if total == 0:
        raise ValueError("Observed group sizes are all zero")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be from (0, 1), got {alpha}")

    if expected_ratios is None:
        ratios: Dict[Any, float] = {label: 1.0 for label in observed}
    else:
        if set(expected_ratios) != set(observed):
            raise ValueError(
                f"Labels of expected_ratios {sorted(map(str, expected_ratios))} "
                f"must match the group labels {sorted(map(str, observed))}"
            )
        if any(ratio <= 0 for ratio in expected_ratios.values()):
            raise ValueError("Expected ratios must be positive")
        ratios = expected_ratios
    ratios_sum: float = sum(ratios.values())

    labels = list(observed)
    observed_sizes = [observed[label] for label in labels]
    expected_sizes = [total * ratios[label] / ratios_sum for label in labels]
    _, pvalue = sps.chisquare(f_obs=observed_sizes, f_exp=expected_sizes)
    return {
        "observed": dict(observed),
        "expected": dict(zip(labels, expected_sizes)),
        "pvalue": float(pvalue),
        "alpha": alpha,
        "srm_detected": bool(pvalue < alpha),
    }


def group_size(dataframe: types.SparkOrPandas) -> int:
    """
    Return the number of rows in a pandas or Spark dataframe.
    """
    if isinstance(dataframe, pd.DataFrame):
        return len(dataframe)
    return dataframe.count()


def check_srm(
    dataframe: types.SparkOrPandas,
    column_groups: types.ColumnNameType,
    expected_ratios: Optional[Dict[Any, float]] = None,
    alpha: float = DEFAULT_SRM_ALPHA,
) -> Dict[str, Any]:
    """
    Run a Sample Ratio Mismatch check on experiment data.

    Examples
    --------
    >>> result = check_srm(df, column_groups="group")
    >>> result["srm_detected"]
    False
    >>> check_srm(df, "group", expected_ratios={"A": 0.9, "B": 0.1})["pvalue"]
    0.83

    Parameters
    ----------
    dataframe : SparkOrPandas
        Dataframe with experiment objects, one row per object.
    column_groups : ColumnNameType
        Column containing the group label of each object.
    expected_ratios : Dict[Any, float], optional
        Mapping from group label to the expected share of objects,
        normalized internally. If ``None``, equal group sizes are expected.
    alpha : float, default: ``0.0005``
        Significance level for the chi-square goodness-of-fit test.

    Returns
    -------
    result : Dict[str, Any]
        Dictionary with ``observed`` sizes, ``expected`` sizes, chi-square
        ``pvalue``, the used ``alpha`` and the boolean ``srm_detected``.
    """
    if isinstance(dataframe, pd.DataFrame):
        if column_groups not in dataframe.columns:
            raise ValueError(f"Column {column_groups} is not in dataframe columns")
        observed = dataframe[column_groups].value_counts().to_dict()
    else:
        rows = dataframe.groupBy(column_groups).count().collect()
        observed = {row[column_groups]: row["count"] for row in rows}
    return check_srm_from_counts(observed, expected_ratios=expected_ratios, alpha=alpha)
