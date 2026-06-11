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


def collect_srm_warnings(run_callable) -> list:
    """
    Run the callable and return the emitted Sample Ratio Mismatch warnings.
    """
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        run_callable()
    return [record for record in records if "Sample Ratio Mismatch" in str(record.message)]


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
def test_alpha_threshold_band():
    """
    A split with a p-value between the strict default alpha and 0.05 is NOT
    flagged by default, but is flagged at a looser alpha: guards the actual
    detection threshold rather than a hardcoded 0.05.
    """
    observed = {"A": 5110, "B": 4890}
    default_result = check_srm_from_counts(observed)
    loose_result = check_srm_from_counts(observed, alpha=0.05)
    assert 0.0005 < default_result["pvalue"] < 0.05
    assert not default_result["srm_detected"]
    assert loose_result["srm_detected"]


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
        tester.run("absolute", method="theory", as_table=False, check_srm=True)


@pytest.mark.unit
def test_tester_silent_on_balanced_split():
    tester = Tester(dataframe=make_groups_frame(5000, 4980), column_groups="group", metrics=["metric"])
    assert not collect_srm_warnings(lambda: tester.run("absolute", method="theory", as_table=False, check_srm=True))


@pytest.mark.unit
def test_tester_srm_off_by_default():
    """
    The check is opt-in: a skewed split produces no warning unless enabled.
    """
    tester = Tester(dataframe=make_groups_frame(5000, 4500), column_groups="group", metrics=["metric"])
    assert not collect_srm_warnings(lambda: tester.run("absolute", method="theory", as_table=False))


@pytest.mark.unit
def test_tester_explicit_false_wins_over_ratios():
    tester = Tester(dataframe=make_groups_frame(5000, 4500), column_groups="group", metrics=["metric"])
    assert not collect_srm_warnings(
        lambda: tester.run(
            "absolute",
            method="theory",
            as_table=False,
            check_srm=False,
            srm_expected_ratios={"A": 0.5, "B": 0.5},
        )
    )


@pytest.mark.unit
def test_ratios_alone_enable_check():
    """
    Providing srm_expected_ratios opts into the check automatically.
    """
    tester = Tester(dataframe=make_groups_frame(5000, 4500), column_groups="group", metrics=["metric"])
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        tester.run("absolute", method="theory", as_table=False, srm_expected_ratios={"A": 0.5, "B": 0.5})


@pytest.mark.unit
def test_tester_respects_expected_ratios():
    """
    An intentional 90/10 split passes once the expected ratios are provided.
    """
    frame = make_groups_frame(9000, 1020)
    tester = Tester(dataframe=frame, column_groups="group", metrics=["metric"])
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        tester.run("absolute", method="theory", as_table=False, check_srm=True)
    assert not collect_srm_warnings(
        lambda: tester.run(
            "absolute",
            method="theory",
            as_table=False,
            srm_expected_ratios={"A": 0.9, "B": 0.1},
        )
    )


@pytest.mark.unit
def test_standalone_test_function_passthrough():
    frame = make_groups_frame(5000, 4500)
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        test("absolute", dataframe=frame, column_groups="group", metrics="metric", as_table=False, check_srm=True)
    assert not collect_srm_warnings(
        lambda: test(
            "absolute",
            dataframe=frame,
            column_groups="group",
            metrics="metric",
            as_table=False,
        )
    )


@pytest.mark.unit
def test_standalone_test_function_default_alpha():
    """
    The standalone test function works without an explicit first_type_errors
    and reports the documented 0.05 default.
    """
    frame = make_groups_frame(1000, 1000)
    result = test("absolute", dataframe=frame, column_groups="group", metrics="metric", as_table=False)
    assert result[0]["first_type_error"] == pytest.approx(0.05)


@pytest.mark.unit
def test_tester_experiment_results_dict_mode():
    """
    The SRM check also covers the experiment_results dict input mode.
    """
    frame = make_groups_frame(5000, 4500)
    experiment_results = {
        "A": frame[frame["group"] == "A"],
        "B": frame[frame["group"] == "B"],
    }
    tester = Tester(experiment_results=experiment_results, metrics=["metric"])
    with pytest.warns(UserWarning, match="Sample Ratio Mismatch detected"):
        tester.run("absolute", method="theory", as_table=False, check_srm=True)


@pytest.mark.unit
def test_tester_experiment_results_arrays_supported():
    """
    Array-valued experiment_results (used with metric_funcs) keep working
    with the SRM check enabled: group sizes are taken via len().
    """
    rng = np.random.default_rng(5)
    experiment_results = {"A": rng.normal(0, 1, 1000), "B": rng.normal(0, 1, 1000)}
    result = test(
        "absolute",
        method="empiric",
        experiment_results=experiment_results,
        metrics="metric",
        metric_funcs={"metric": lambda values: values},
        as_table=False,
        random_seed=7,
        check_srm=True,
    )
    assert "pvalue" in result[0]


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
