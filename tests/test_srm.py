import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps

from ambrosia.tester import Tester, test
from ambrosia.tools.srm import check_srm, check_srm_from_counts


def make_groups_frame(size_a: int, size_b: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "group": ["A"] * size_a + ["B"] * size_b,
            "metric": rng.normal(0, 1, size_a + size_b),
        }
    )


@pytest.mark.unit
def test_pvalue_matches_scipy():
    """
    The reported p-value equals a manual scipy chi-square computation.
    """
    observed = {"A": 5200, "B": 4800}
    result = check_srm_from_counts(observed)
    _, expected_pvalue = sps.chisquare(f_obs=[5200, 4800], f_exp=[5000.0, 5000.0])
    assert result["pvalue"] == pytest.approx(expected_pvalue, abs=1e-12)
    assert result["observed"] == observed
    assert result["expected"] == {"A": 5000.0, "B": 5000.0}
    assert result["alpha"] == 0.0005


@pytest.mark.unit
def test_balanced_split_passes():
    result = check_srm_from_counts({"A": 5000, "B": 4980})
    assert not result["srm_detected"]
    assert result["pvalue"] > 0.0005


@pytest.mark.unit
def test_skewed_split_detected():
    result = check_srm_from_counts({"A": 5000, "B": 4500})
    assert result["srm_detected"]
    assert result["pvalue"] < 0.0005


@pytest.mark.unit
def test_multigroup_skew_detected():
    balanced = check_srm_from_counts({"A": 3000, "B": 3010, "C": 2990})
    skewed = check_srm_from_counts({"A": 3000, "B": 3000, "C": 2500})
    assert not balanced["srm_detected"]
    assert skewed["srm_detected"]


@pytest.mark.unit
def test_custom_expected_ratios():
    """
    A deliberate 90/10 split passes with matching ratios and fails without them.
    """
    observed = {"A": 9000, "B": 1020}
    with_ratios = check_srm_from_counts(observed, expected_ratios={"A": 0.9, "B": 0.1})
    without_ratios = check_srm_from_counts(observed)
    assert not with_ratios["srm_detected"]
    assert without_ratios["srm_detected"]


@pytest.mark.unit
def test_ratios_are_normalized():
    observed = {"A": 9000, "B": 1020}
    fractions = check_srm_from_counts(observed, expected_ratios={"A": 0.9, "B": 0.1})
    weights = check_srm_from_counts(observed, expected_ratios={"A": 9, "B": 1})
    assert fractions["pvalue"] == pytest.approx(weights["pvalue"], abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "observed, expected_ratios, alpha",
    [
        ({"A": 100}, None, 0.0005),
        ({"A": 100, "B": -1}, None, 0.0005),
        ({"A": 0, "B": 0}, None, 0.0005),
        ({"A": 100, "B": 100}, {"A": 1.0}, 0.0005),
        ({"A": 100, "B": 100}, {"A": 1.0, "B": 1.0, "C": 1.0}, 0.0005),
        ({"A": 100, "B": 100}, {"A": 1.0, "B": 0.0}, 0.0005),
        ({"A": 100, "B": 100}, None, 0.0),
        ({"A": 100, "B": 100}, None, 1.0),
    ],
)
def test_invalid_inputs_raise(observed, expected_ratios, alpha):
    with pytest.raises(ValueError):
        check_srm_from_counts(observed, expected_ratios=expected_ratios, alpha=alpha)


@pytest.mark.unit
def test_dataframe_wrapper_matches_counts():
    frame = make_groups_frame(5000, 4500)
    from_frame = check_srm(frame, column_groups="group")
    from_counts = check_srm_from_counts({"A": 5000, "B": 4500})
    assert from_frame["pvalue"] == pytest.approx(from_counts["pvalue"], abs=1e-12)
    assert from_frame["srm_detected"]


@pytest.mark.unit
def test_dataframe_wrapper_missing_column():
    with pytest.raises(ValueError, match="is not in dataframe columns"):
        check_srm(make_groups_frame(10, 10), column_groups="no_such_column")


@pytest.mark.unit
def test_tester_warns_on_srm():
    tester = Tester(dataframe=make_groups_frame(5000, 4500), column_groups="group", metrics=["metric"])
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        tester.run("absolute", method="theory", as_table=False)


@pytest.mark.unit
def test_tester_silent_on_balanced_split():
    tester = Tester(dataframe=make_groups_frame(5000, 4980), column_groups="group", metrics=["metric"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tester.run("absolute", method="theory", as_table=False)


@pytest.mark.unit
def test_tester_srm_check_can_be_disabled():
    tester = Tester(dataframe=make_groups_frame(5000, 4500), column_groups="group", metrics=["metric"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tester.run("absolute", method="theory", as_table=False, check_srm=False)


@pytest.mark.unit
def test_tester_respects_expected_ratios():
    """
    An intentional 90/10 split passes once the expected ratios are provided.
    """
    frame = make_groups_frame(9000, 1020)
    tester = Tester(dataframe=frame, column_groups="group", metrics=["metric"])
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        tester.run("absolute", method="theory", as_table=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tester.run(
            "absolute",
            method="theory",
            as_table=False,
            srm_expected_ratios={"A": 0.9, "B": 0.1},
        )


@pytest.mark.unit
def test_standalone_test_function_passthrough():
    frame = make_groups_frame(5000, 4500)
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        test("absolute", dataframe=frame, column_groups="group", metrics="metric", as_table=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test(
            "absolute",
            dataframe=frame,
            column_groups="group",
            metrics="metric",
            as_table=False,
            check_srm=False,
        )


@pytest.mark.unit
def test_check_srm_spark(local_spark_session):
    """
    The dataframe wrapper computes group sizes for Spark tables as well.
    """
    frame = make_groups_frame(500, 430)
    spark_frame = local_spark_session.createDataFrame(frame)
    from_spark = check_srm(spark_frame, column_groups="group")
    from_pandas = check_srm(frame, column_groups="group")
    assert from_spark["observed"] == from_pandas["observed"]
    assert from_spark["pvalue"] == pytest.approx(from_pandas["pvalue"], abs=1e-12)
