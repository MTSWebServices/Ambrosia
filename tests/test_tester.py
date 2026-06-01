from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from ambrosia.tester import Tester, test
from ambrosia.tools import multitest as mt
from ambrosia.tools.stat_criteria import TtestIndCriterion, TtestRelCriterion


def check_eq(a: float, b: float, eps: float = 1e-5) -> bool:
    if a == np.inf and b == np.inf:
        return True
    if a == -np.inf and b == -np.inf:
        return True
    return abs(a - b) < eps


def check_eq_int(i1, i2) -> bool:
    return check_eq(i1[0], i2[0]) and check_eq(i1[1], i2[1])


@pytest.mark.smoke
def test_instance():
    """
    Check that simple instance without args work
    """
    tester = Tester()


@pytest.mark.smoke
def test_constructors(results_ltv_retention_conversions):
    """
    Test different constructors
    """
    # Only table
    tester = Tester(dataframe=results_ltv_retention_conversions, column_groups="group")
    # Use metrics
    tester = Tester(
        dataframe=results_ltv_retention_conversions, metrics=["retention", "conversions"], column_groups="group"
    )
    tester = Tester(metrics="ltv")


@pytest.mark.smoke
@pytest.mark.parametrize("effect_type", ["relative", "absolute"])
@pytest.mark.parametrize("as_table", [False, True])
def test_correct_type(effect_type, as_table, tester_on_ltv_retention):
    """
    Check, that method run is callable and return correct type
    """
    types = [List, pd.DataFrame]
    assert isinstance(tester_on_ltv_retention.run(effect_type, as_table=as_table), types[as_table])


@pytest.mark.unit
@pytest.mark.parametrize("effect_type", ["relative", "absolute"])
@pytest.mark.parametrize("method", ["theory", "empiric"])
def test_every_type_run(effect_type, method, tester_on_ltv_retention):
    """
    Use cortesian product of all params to check, that all posible combinations are working
    """
    result = tester_on_ltv_retention.run(effect_type=effect_type, method=method, as_table=False)
    assert result[0]["effect"] > 0
    assert result[2]["effect"] < 0


def check_pvalue_for_interval(interval: Tuple, pvalue: float, alpha: float, check_value: float = 0) -> bool:
    """
    Check, that check_value in interval <=> pvalue <= alpha
    """
    if interval[0] <= check_value and interval[1] >= check_value and pvalue <= alpha:
        return False
    elif (interval[0] > check_value or interval[1] < check_value) and pvalue > alpha:
        return False
    else:
        return True


@pytest.mark.unit
@pytest.mark.parametrize("method", ["theory", "binary", "empiric"])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
def test_coinf_interval_absolute(method, alpha, metrics, criterion, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue < alpha
    """
    result = tester_on_ltv_retention.run(
        "absolute", method=method, criterion=criterion, first_type_errors=alpha, metrics=metrics, as_table=False
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha, 0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["theory", "empiric"])
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions", "ltv"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_relative(method, alpha, metrics, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 1 <=> pvalue <= alpha
    """
    result = tester_on_ltv_retention.run(
        "relative",
        method=method,
        first_type_errors=alpha,
        metrics=metrics,
        as_table=False,
        alternative=alternative,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha, 0)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("interval_type", ["wald", "yule", "newcombe", "yule_modif", "jeffrey", "recenter"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_bin_abs(alpha, metrics, interval_type, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue <= alpha
    For binary method and different interval approaches
    For absolute effect
    """
    result = tester_on_ltv_retention.run(
        "absolute",
        method="binary",
        first_type_errors=alpha,
        metrics=metrics,
        interval_type=interval_type,
        alternative=alternative,
        as_table=False,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_coinf_interval_bin_rel(alpha, metrics, alternative, tester_on_ltv_retention):
    """
    Test that confidence interval contains 0 <=> pvalue <= alpha
    For binary method and different interval approaches
    For relative effect
    """
    result = tester_on_ltv_retention.run(
        "relative",
        method="binary",
        first_type_errors=alpha,
        metrics=metrics,
        alternative=alternative,
        as_table=False,
    )[0]
    interval = result["confidence_interval"]
    pvalue = result["pvalue"]
    assert check_pvalue_for_interval(interval, pvalue, alpha)


@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("method", ["theory", "binary"])
@pytest.mark.parametrize("alpha", [0.01, 0.05])
@pytest.mark.parametrize("metrics", ["retention", "conversions"])
def test_standalone_test_function(
    criterion, effect_type, method, alpha, metrics, tester_on_ltv_retention, results_ltv_retention_conversions
):
    """
    Test standalone test function gives same result as Tester class.
    """
    if method == "binary" and effect_type == "relative":
        return

    function_result = test(
        effect_type,
        method,
        dataframe=results_ltv_retention_conversions,
        metrics=metrics,
        criterion=criterion,
        column_groups="group",
        first_type_errors=alpha,
        as_table=False,
    )
    class_result = tester_on_ltv_retention.run(
        effect_type, method, metrics=metrics, first_type_errors=alpha, criterion=criterion, as_table=False
    )
    assert function_result == class_result


@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
def test_criteria_ttest_different(effect_type):
    """
    Test criteria classes
    """
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([2, 3, 4, 7, 10])
    ttest_ind = TtestIndCriterion()
    ttest_rel = TtestRelCriterion()
    assert ttest_ind.calculate_pvalue(group_a, group_b, effect_type=effect_type) != ttest_rel.calculate_pvalue(
        group_a, group_b, effect_type=effect_type
    )
    assert ttest_ind.calculate_conf_interval(
        group_a, group_b, effect_type=effect_type
    ) != ttest_rel.calculate_conf_interval(group_a, group_b, effect_type=effect_type)


@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel", "mw", "wilcoxon"])
@pytest.mark.parametrize("metrics, alternative", [("retention", "greater"), ("conversions", "less"), ("ltv", "less")])
def test_kwargs_passing_theory(criterion, metrics, alternative, tester_on_ltv_retention):
    """
    Test passing key word argument to run method for theoretical approach.
    """
    old_pvalue = tester_on_ltv_retention.run(criterion=criterion, metrics=metrics, as_table=False)[0]["pvalue"]
    alternative_pvalue = tester_on_ltv_retention.run(
        criterion=criterion, metrics=metrics, as_table=False, alternative=alternative
    )[0]["pvalue"]
    assert old_pvalue >= alternative_pvalue


@pytest.mark.parametrize("metrics, alternative", [("retention", "greater"), ("conversions", "less")])
def test_kwargs_passing_empiric(metrics, alternative, tester_on_ltv_retention):
    """
    Test passing key word argument to run method for empirical approach.
    """
    random_seed: int = 33
    old_pvalue = tester_on_ltv_retention.run(
        method="empiric",
        metrics=metrics,
        random_seed=random_seed,
        as_table=False,
    )[0]["pvalue"]
    alternative_pvalue = tester_on_ltv_retention.run(
        method="empiric",
        metrics=metrics,
        as_table=False,
        random_seed=random_seed,
        alternative=alternative,
    )[0]["pvalue"]
    assert old_pvalue >= alternative_pvalue


@pytest.mark.parametrize("interval_type", ["yule", "yule_modif", "newcombe", "jeffrey", "agresti"])
def test_kwargs_passing_binary(interval_type, tester_on_ltv_retention):
    """
    Test passing key word argument to run method for binary metrics.
    """
    wald_interval = tester_on_ltv_retention.run("absolute", "binary", metrics="retention", as_table=False)[0][
        "confidence_interval"
    ]
    other_interval = tester_on_ltv_retention.run(
        "absolute", "binary", metrics="retention", interval_type=interval_type, as_table=False
    )[0]["confidence_interval"]
    assert wald_interval != other_interval


def get_ci_pvalue(tester_on_ltv_retention, alternative: str, idx: int = 0, **run_kwargs):
    """
    Get pvalue and confidence intervals for alternative
    """
    res_table = tester_on_ltv_retention.run(alternative=alternative, **run_kwargs)
    pvalue = res_table[idx]["pvalue"]
    confidence_interval = res_table[idx]["confidence_interval"]
    return pvalue, confidence_interval


def calc_intervals_pvalue(tester_on_ltv_retention, idx: int = 0, **run_kwargs) -> bool:
    """
    Calc pvalue and intervals
    """
    pvalue_center, int_center = get_ci_pvalue(tester_on_ltv_retention, "two-sided", idx, **run_kwargs)
    pvalue_gr, int_gr = get_ci_pvalue(tester_on_ltv_retention, "greater", idx, **run_kwargs)
    pvalue_less, int_less = get_ci_pvalue(tester_on_ltv_retention, "less", idx, **run_kwargs)
    return pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less


def check_bound_intervals(int_center, int_less, int_gr, left_bound: float = -np.inf, right_bound: float = np.inf):
    """
    Check bound of intervals for different alternatives
    """
    assert int_less[0] == left_bound
    assert int_gr[1] == right_bound
    assert int_gr[0] > int_center[0]
    assert int_less[1] < int_center[1]


@pytest.mark.parametrize("effect_type", ["absolute"])
@pytest.mark.parametrize("interval_type", ["wald", "yule", "newcombe", "yule_modif", "jeffrey", "recenter"])
def test_alternative_change_binary(effect_type, interval_type, tester_on_ltv_retention):
    """
    Test changes in pvalue and confidence interval for binary method
    """
    pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less = calc_intervals_pvalue(
        tester_on_ltv_retention, effect_type=effect_type, method="binary", metrics="retention", as_table=False
    )
    # mean retention A - 0.303
    # mean retention B - 0.399
    assert pvalue_less > pvalue_center
    assert pvalue_center > pvalue_gr
    # Check intervals
    check_bound_intervals(int_center, int_less, int_gr, -1, 1)


@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("criterion", ["ttest", "ttest_rel"])
def test_alternative_change_th(effect_type, criterion, tester_on_ltv_retention):
    """
    Test changes in pvalue and confidence interval for theory method
    """
    pvalue_center, int_center, pvalue_gr, int_gr, pvalue_less, int_less = calc_intervals_pvalue(
        tester_on_ltv_retention,
        effect_type=effect_type,
        criterion=criterion,
        method="theory",
        metrics="ltv",
        as_table=False,
    )
    # Mean(group_a) > Mean(group_b) in this table
    assert pvalue_less < pvalue_center
    assert pvalue_center < pvalue_gr
    # Check intervals
    check_bound_intervals(int_center, int_less, int_gr)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
def test_spark_tester(tester_spark_ltv_ret, tester_on_ltv_retention, alternative: str, effect_type: str):
    """
    Test the Tester results for Spark and Pandas dataframe for equivalence.
    """
    res_pandas = tester_on_ltv_retention.run(
        effect_type, "theory", correction_method=None, as_table=False, alternative=alternative
    )
    res_spark = tester_spark_ltv_ret.run(
        effect_type, "theory", correction_method=None, as_table=False, alternative=alternative
    )
    for j in range(len(res_pandas)):
        assert check_eq(res_pandas[j]["pvalue"], res_spark[j]["pvalue"])
        assert check_eq_int(res_pandas[j]["confidence_interval"], res_spark[j]["confidence_interval"])


@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
def test_paired_bootstrap(effect_type, alternative):
    """
    Compare pvalues and confidence intervals between paired and regular bootstrap
    for generated dependent groups
    """
    sample_size: Tuple = (1000,)
    metrics: str = "metric"
    column_groups: str = "group"
    random_seed: int = 9
    rng = np.random.default_rng(random_seed)

    data_a = pd.DataFrame({metrics: rng.normal(loc=2.0, size=sample_size), column_groups: "A"})
    data_b = data_a.copy()
    data_b[metrics] += 0.1 + rng.normal(size=sample_size)
    data_b[column_groups] = "B"
    test_data = pd.concat([data_a, data_b])

    tester = Tester(dataframe=test_data, metrics=metrics, column_groups=column_groups)
    test_results_ind = tester.run(
        effect_type=effect_type,
        method="empiric",
        paired=False,
        alternative=alternative,
        random_seed=random_seed,
        as_table=False,
    )
    test_results_dep = tester.run(
        effect_type=effect_type,
        method="empiric",
        paired=True,
        alternative=alternative,
        random_seed=random_seed,
        as_table=False,
    )
    assert test_results_dep[0]["pvalue"] < test_results_ind[0]["pvalue"]
    assert test_results_dep[0]["confidence_interval"][0] > test_results_ind[0]["confidence_interval"][0]


@pytest.mark.unit
def test_metric_func_constructor(results_ltv_retention_conversions):
    """
    Test that metric_funcs passed to constructor are used when metric name matches.
    """
    # ratio metric: ltv / retention (arbitrary, just to test callable path)
    ratio_func = lambda df: (df["ltv"] / (df["retention"] + 1e-6)).values
    tester = Tester(
        dataframe=results_ltv_retention_conversions,
        column_groups="group",
        metrics=["ratio_metric"],
        metric_funcs={"ratio_metric": ratio_func},
    )
    result = tester.run(as_table=False)
    assert len(result) == 1
    assert "pvalue" in result[0]


@pytest.mark.unit
@pytest.mark.parametrize("method", ["theory", "empiric"])
def test_metric_func_run(method, results_ltv_retention_conversions):
    """
    Test that metric_funcs passed to run() work for theory and empiric methods.
    """
    double_ltv = lambda df: (df["ltv"] * 2).values
    tester = Tester(
        dataframe=results_ltv_retention_conversions,
        column_groups="group",
        metrics=["ltv"],
    )
    result_normal = tester.run(method=method, metrics=["ltv"], as_table=False)
    result_func = tester.run(
        method=method,
        metrics=["custom"],
        metric_funcs={"custom": double_ltv},
        as_table=False,
    )
    # Doubling values doesn't change pvalue for ttest (same scale), but effect should be doubled
    assert abs(result_func[0]["effect"]) == pytest.approx(abs(result_normal[0]["effect"]) * 2, rel=1e-4)


@pytest.mark.unit
def test_metric_func_overrides_constructor(results_ltv_retention_conversions):
    """
    Test that metric_funcs in run() override those set in constructor.
    """
    func_a = lambda df: df["ltv"].values
    func_b = lambda df: (df["ltv"] * 3).values
    tester = Tester(
        dataframe=results_ltv_retention_conversions,
        column_groups="group",
        metric_funcs={"my_metric": func_a},
    )
    result_a = tester.run(metrics=["my_metric"], as_table=False)
    result_b = tester.run(metrics=["my_metric"], metric_funcs={"my_metric": func_b}, as_table=False)
    assert abs(result_b[0]["effect"]) == pytest.approx(abs(result_a[0]["effect"]) * 3, rel=1e-4)


@pytest.mark.unit
def test_metric_func_bootstrap(results_ltv_retention_conversions):
    """
    Test that metric_funcs work with empiric (bootstrap) method.
    """
    double_ltv = lambda df: (df["ltv"] * 2).values
    tester = Tester(
        dataframe=results_ltv_retention_conversions,
        column_groups="group",
        metrics=["custom"],
        metric_funcs={"custom": double_ltv},
    )
    result = tester.run(method="empiric", as_table=False)
    assert len(result) == 1
    assert "pvalue" in result[0]
    assert "effect" in result[0]
    assert "confidence_interval" in result[0]


def _two_group_frame() -> pd.DataFrame:
    """
    Deterministic two-group, two-metric frame with UNEQUAL within-group variances.

    The unequal spread makes the pinned p-values specific to the Welch t-test
    (the default criterion); they would change if the variance handling regressed
    to a pooled/Student t-test.
    """
    return pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "m1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "m2": [5, 5, 5, 5, 5, 6, 6, 6, 6, 6] + [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )


def _three_group_frame() -> pd.DataFrame:
    """
    Deterministic three-group, one-metric frame. The family size becomes
    m = C(3, 2) * 1 = 3, exercising both the pair enumeration and the p-value
    clip-at-1 boundary (the A-vs-B pair has raw p * 3 > 1).
    """
    return pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            + [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        }
    )


@pytest.mark.unit
def test_bonferroni_backward_compat_characterization():
    """
    Lock the EXACT current Bonferroni output (absolute effect) before the multitest
    refactor. With two metrics and two groups the family size is m = C(2, 2) * 2 = 2:
    p-values are multiplied by 2 (clipped at 1), confidence intervals are built at
    alpha / 2, and the reported first type error is restored to the nominal level.
    The default ``correction_method`` is ``"bonferroni"``, so default == explicit.
    Backward-compatibility guard: must keep passing unchanged after the refactor.
    """
    tester = Tester(
        dataframe=_two_group_frame(),
        column_groups="group",
        metrics=["m1", "m2"],
        first_type_errors=0.05,
    )
    res_bonf = tester.run("absolute", method="theory", correction_method="bonferroni", as_table=False)
    res_none = tester.run("absolute", method="theory", correction_method=None, as_table=False)
    res_default = tester.run("absolute", method="theory", as_table=False)
    bonf = {r["metric name"]: r for r in res_bonf}
    none = {r["metric name"]: r for r in res_none}
    default = {r["metric name"]: r for r in res_default}

    # Exact uncorrected Welch p-values (differ from Student's t for this frame).
    assert none["m1"]["pvalue"] == pytest.approx(0.0230712838, abs=1e-9)
    assert none["m2"]["pvalue"] == pytest.approx(0.0679387323, abs=1e-9)
    # Exact Bonferroni-corrected p-values.
    assert bonf["m1"]["pvalue"] == pytest.approx(0.0461425675, abs=1e-9)
    assert bonf["m2"]["pvalue"] == pytest.approx(0.1358774645, abs=1e-9)

    for metric in ["m1", "m2"]:
        # The default correction_method is "bonferroni": default == explicit.
        assert default[metric]["pvalue"] == pytest.approx(bonf[metric]["pvalue"], abs=1e-12)
        # Bonferroni p-value == min(raw * m, 1).
        assert bonf[metric]["pvalue"] == pytest.approx(min(none[metric]["pvalue"] * 2, 1.0), abs=1e-12)
        # Reported first type error is restored to the nominal level.
        assert float(np.ravel(bonf[metric]["first_type_error"])[0]) == pytest.approx(0.05)
        # Corrected confidence interval is strictly wider (built at alpha / m).
        assert bonf[metric]["confidence_interval"][0] < none[metric]["confidence_interval"][0]
        assert bonf[metric]["confidence_interval"][1] > none[metric]["confidence_interval"][1]


@pytest.mark.unit
def test_bonferroni_characterization_as_table():
    """
    Lock the ``as_table=True`` output (the primary path; the list-of-dicts output is
    derived from it): the column set and the restored nominal first_type_error column.
    """
    tester = Tester(
        dataframe=_two_group_frame(),
        column_groups="group",
        metrics=["m1", "m2"],
        first_type_errors=0.05,
    )
    table = tester.run("absolute", method="theory", correction_method="bonferroni", as_table=True)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "first_type_error",
        "pvalue",
        "effect",
        "confidence_interval",
        "metric name",
        "group A label",
        "group B label",
    ]
    assert list(table["first_type_error"]) == [0.05, 0.05]
    pvals = dict(zip(table["metric name"], table["pvalue"]))
    assert pvals["m1"] == pytest.approx(0.0461425675, abs=1e-9)
    assert pvals["m2"] == pytest.approx(0.1358774645, abs=1e-9)


@pytest.mark.unit
def test_bonferroni_characterization_relative():
    """
    Lock current Bonferroni behavior for the relative effect type (delta-method
    p-value path). Family size m = 2, so p-values double (clipped at 1).
    """
    tester = Tester(
        dataframe=_two_group_frame(),
        column_groups="group",
        metrics=["m1", "m2"],
        first_type_errors=0.05,
    )
    none = {r["metric name"]: r for r in tester.run("relative", "theory", correction_method=None, as_table=False)}
    bonf = {
        r["metric name"]: r for r in tester.run("relative", "theory", correction_method="bonferroni", as_table=False)
    }
    assert none["m1"]["pvalue"] == pytest.approx(0.0239676798, abs=1e-9)
    assert none["m2"]["pvalue"] == pytest.approx(0.0316317147, abs=1e-9)
    for metric in ["m1", "m2"]:
        assert bonf[metric]["pvalue"] == pytest.approx(min(none[metric]["pvalue"] * 2, 1.0), abs=1e-12)


@pytest.mark.unit
def test_bonferroni_three_groups_clip_characterization():
    """
    Three groups, one metric => m = C(3, 2) = 3 hypotheses. Locks the pair
    enumeration, the exact 3x scaling for non-clipping pairs, and the clip-at-1
    boundary (the A-vs-B pair has raw p * 3 > 1).
    """
    tester = Tester(
        dataframe=_three_group_frame(),
        column_groups="group",
        metrics=["x"],
        first_type_errors=0.05,
    )
    none = tester.run("absolute", "theory", correction_method=None, as_table=False)
    bonf = tester.run("absolute", "theory", correction_method="bonferroni", as_table=False)
    assert len(none) == 3 and len(bonf) == 3
    none_by = {(r["group A label"], r["group B label"]): r["pvalue"] for r in none}
    bonf_by = {(r["group A label"], r["group B label"]): r["pvalue"] for r in bonf}
    assert set(none_by) == {("A", "B"), ("A", "C"), ("B", "C")}
    # Each corrected p-value equals min(raw * 3, 1).
    for pair, raw_p in none_by.items():
        assert bonf_by[pair] == pytest.approx(min(raw_p * 3, 1.0), abs=1e-12)
    # A-vs-B clips to 1.0; the well-separated pairs do not.
    assert bonf_by[("A", "B")] == pytest.approx(1.0, abs=1e-12)
    assert bonf_by[("A", "C")] < 1.0
    assert bonf_by[("B", "C")] < 1.0


@pytest.mark.unit
def test_single_metric_no_correction_characterization():
    """
    With a single hypothesis (one metric, two groups) no correction is applied,
    so Bonferroni, the default and ``None`` must all yield identical p-values.
    """
    tester = Tester(dataframe=_two_group_frame(), column_groups="group", metrics=["m1"], first_type_errors=0.05)
    p_bonf = tester.run("absolute", method="theory", correction_method="bonferroni", as_table=False)[0]["pvalue"]
    p_none = tester.run("absolute", method="theory", correction_method=None, as_table=False)[0]["pvalue"]
    assert p_bonf == pytest.approx(p_none, abs=1e-12)
    assert p_none == pytest.approx(0.0230712838, abs=1e-9)


NEW_CORRECTION_METHODS = ["sidak", "holm", "holm-sidak", "fdr_bh", "fdr_by", "hommel", "simes-hochberg"]
ALL_CORRECTION_METHODS = ["bonferroni"] + NEW_CORRECTION_METHODS


def _three_group_two_metric_frame() -> pd.DataFrame:
    """
    Deterministic three-group, two-metric frame (family size m = C(3, 2) * 2 = 6)
    with a spread of effect sizes, so the corrected p-values vary across methods.
    """
    rng = np.random.default_rng(2024)
    n = 150
    return pd.DataFrame(
        {
            "group": ["A"] * n + ["B"] * n + ["C"] * n,
            "x": np.r_[rng.normal(0.0, 1.0, n), rng.normal(0.3, 1.0, n), rng.normal(0.15, 1.0, n)],
            "y": np.r_[rng.normal(10.0, 3.0, n), rng.normal(10.4, 3.0, n), rng.normal(9.7, 3.0, n)],
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize("effect_type", ["absolute", "relative"])
@pytest.mark.parametrize("method", ALL_CORRECTION_METHODS)
def test_correction_method_wires_multitest(method, effect_type):
    """
    The Tester applies the multitest module to the family of raw p-values:
    corrected p-values match the module output, the reported first type error
    stays nominal, and only the constant-scaling methods (Bonferroni, Sidak)
    widen the confidence intervals while step-wise methods leave them nominal.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )
    raw = tester.run(effect_type, method="theory", correction_method=None, as_table=True)
    corrected = tester.run(effect_type, method="theory", correction_method=method, as_table=True)
    np.testing.assert_allclose(corrected["pvalue"].values, mt.adjust_pvalues(raw["pvalue"].values, method), atol=1e-12)
    np.testing.assert_allclose(corrected["first_type_error"].values, raw["first_type_error"].values)
    if mt.is_ci_correctable(method):
        assert list(corrected["confidence_interval"]) != list(raw["confidence_interval"])
    else:
        assert list(corrected["confidence_interval"]) == list(raw["confidence_interval"])


@pytest.mark.unit
@pytest.mark.parametrize("method", ALL_CORRECTION_METHODS)
def test_correction_is_more_conservative_than_raw(method):
    """
    Every supported correction yields p-values no smaller than the uncorrected ones.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )
    raw = tester.run("absolute", method="theory", correction_method=None, as_table=True)["pvalue"].values
    corrected = tester.run("absolute", method="theory", correction_method=method, as_table=True)["pvalue"].values
    assert np.all(corrected >= raw - 1e-12)


@pytest.mark.unit
def test_correction_relative_ordering():
    """
    At the Tester level: raw <= Holm <= Bonferroni and Benjamini-Hochberg <= Holm
    (results across runs share the same row order, so they compare element-wise).
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )

    def pvalues(correction):
        return tester.run("absolute", method="theory", correction_method=correction, as_table=True)["pvalue"].values

    raw, bonf, holm, bh = pvalues(None), pvalues("bonferroni"), pvalues("holm"), pvalues("fdr_bh")
    tol = 1e-12
    assert np.all(raw <= holm + tol)
    assert np.all(holm <= bonf + tol)
    assert np.all(bh <= holm + tol)


@pytest.mark.unit
@pytest.mark.parametrize("method", NEW_CORRECTION_METHODS)
def test_single_hypothesis_correction_is_noop(method):
    """
    With a single hypothesis every method equals the uncorrected result.
    """
    tester = Tester(dataframe=_two_group_frame(), column_groups="group", metrics=["m1"], first_type_errors=0.05)
    p_none = tester.run("absolute", method="theory", correction_method=None, as_table=False)[0]["pvalue"]
    p_corr = tester.run("absolute", method="theory", correction_method=method, as_table=False)[0]["pvalue"]
    assert p_corr == pytest.approx(p_none, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["holm", "fdr_bh"])
def test_correction_as_table_matches_records(method):
    """
    The as_table=True and as_table=False outputs carry the same corrected p-values.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )
    table = tester.run("absolute", method="theory", correction_method=method, as_table=True)
    records = tester.run("absolute", method="theory", correction_method=method, as_table=False)
    np.testing.assert_allclose(table["pvalue"].values, [record["pvalue"] for record in records], atol=1e-12)


@pytest.mark.unit
def test_invalid_correction_method_raises():
    """
    An unknown correction method raises ValueError when correction is applied.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )
    with pytest.raises(ValueError):
        tester.run("absolute", method="theory", correction_method="not-a-method")


@pytest.mark.unit
def test_correction_with_nan_pvalue_counts_full_family():
    """
    A degenerate (constant) metric yields a NaN p-value. It still counts toward
    the family size, so the surviving metric is corrected by the full family of 2
    (matching the pre-refactor Bonferroni behavior), and the NaN passes through.
    """
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "m_ok": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "m_const": [5] * 20,
        }
    )
    tester = Tester(dataframe=frame, column_groups="group", metrics=["m_ok", "m_const"], first_type_errors=0.05)
    raw = {
        r["metric name"]: r["pvalue"] for r in tester.run("absolute", "theory", correction_method=None, as_table=False)
    }
    bonf = {
        r["metric name"]: r["pvalue"]
        for r in tester.run("absolute", "theory", correction_method="bonferroni", as_table=False)
    }
    assert np.isnan(raw["m_const"]) and np.isnan(bonf["m_const"])
    assert bonf["m_ok"] == pytest.approx(min(raw["m_ok"] * 2, 1.0), abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["holm", "fdr_bh"])
def test_correction_with_alpha_vector_uses_hypothesis_family(method):
    """
    With several first type error levels the result table repeats each hypothesis
    once per level, but a step-wise correction must use the hypothesis family
    (size 6 here), not the flattened 12-row table. The corrected p-value of each
    hypothesis is therefore the family-6 adjustment, shared across the levels.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=[0.05, 0.1],
    )
    raw_family = tester.run("absolute", "theory", first_type_errors=0.05, correction_method=None, as_table=True)
    corrected = tester.run("absolute", "theory", correction_method=method, as_table=True)
    assert len(raw_family) == 6
    assert len(corrected) == 12
    assert sorted(np.unique(corrected["first_type_error"]).tolist()) == [0.05, 0.1]
    expected = mt.adjust_pvalues(raw_family["pvalue"].values, method)  # family size 6
    unique_corrected = corrected.drop_duplicates(subset=["group A label", "group B label", "metric name"])
    np.testing.assert_allclose(unique_corrected["pvalue"].values, expected, atol=1e-12)


@pytest.mark.unit
def test_correction_alias_through_tester():
    """
    A correction alias passed to the Tester resolves to its canonical method.
    """
    tester = Tester(
        dataframe=_three_group_two_metric_frame(),
        column_groups="group",
        metrics=["x", "y"],
        first_type_errors=0.05,
    )
    alias = tester.run("absolute", "theory", correction_method="bh", as_table=True)["pvalue"].values
    canonical = tester.run("absolute", "theory", correction_method="fdr_bh", as_table=True)["pvalue"].values
    np.testing.assert_allclose(alias, canonical, atol=1e-12)
