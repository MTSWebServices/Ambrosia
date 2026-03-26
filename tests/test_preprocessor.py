import os

import numpy as np
import pandas as pd
import pytest

from ambrosia.preprocessing import Preprocessor

store_path: str = "tests/configs/preprocessor_config.json"


@pytest.mark.smoke()
def test_init(data_nonlin_var):
    """
    Instantiation of preprocessor class
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    preprocessor.data()
    preprocessor.data(copy=True)


@pytest.mark.smoke()
def test_cuped_sequential(data_nonlin_var):
    """
    Test sequential cuped + robust
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed: pd.DataFrame = (
        preprocessor.robust("target", alpha=0.005)
        .cuped("target", "feature_1", transformed_name="target_1")
        .cuped("target_1", "feature_2", transformed_name="target_2")
        .cuped("target_2", "feature_3", transformed_name="target_3")
        .data()
    )


@pytest.mark.smoke()
def test_full_sequential(data_nonlin_var):
    """
    Test available transformations sequentially.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_2", "feature_3"])
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
    )


@pytest.mark.unit()
def test_load_store_methods(data_nonlin_var):
    """
    Test load and store methods of Preprocessor for the number of transformations.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
    )
    preprocessor.store_transformations(store_path)
    loaded_preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    loaded_preprocessor.load_transformations(store_path)
    os.remove(store_path)
    for transformer, loaded_transformer in zip(preprocessor.transformations(), loaded_preprocessor.transformations()):
        assert transformer.get_params_dict() == loaded_transformer.get_params_dict()


@pytest.mark.unit()
def test_transform_from_config(data_nonlin_var):
    """
    Test store and transform from config method of Preprocessor for the number of transformations.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed: pd.DataFrame = (
        preprocessor.robust("feature_1", alpha=0.01, tail="right")
        .iqr(["feature_2", "feature_3"])
        .iqr(["feature_1"])
        .log("feature_1")
        .boxcox(["feature_2", "feature_3"])
        .cuped("target", "feature_3", transformed_name="target_cuped_1")
        .cuped("target", "feature_2", transformed_name="target_cuped_2")
        .multicuped("target", ["feature_1", "feature_2"], transformed_name="target_multicuped")
        .data()
    )
    preprocessor.store_transformations(store_path)
    loaded_preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    transformed_by_config: pd.DataFrame = loaded_preprocessor.transform_from_config(store_path)
    os.remove(store_path)
    assert (transformed == transformed_by_config).all(None)


@pytest.mark.unit()
def test_store_load_config(data_for_agg):
    """
    Test load, store, apply methods of Preprocessor for the number of transformations.
    """
    preprocessor = Preprocessor(data_for_agg, verbose=False)
    transformed: pd.DataFrame = (
        preprocessor.aggregate(
            groupby_columns="id",
            agg_params={"watched": "sum", "sessions": "max", "gender": "simple", "platform": "mode"},
        )
        .robust(["watched", "sessions"], alpha=0.01)
        .cuped("watched", by="sessions", transformed_name="watched_cuped")
        .data()
    )
    preprocessor.store_transformations(store_path)
    loaded_preprocessor = Preprocessor(data_for_agg, verbose=False)
    loaded_preprocessor.load_transformations(store_path)
    transformed_by_config: pd.DataFrame = loaded_preprocessor.apply_transformations()
    os.remove(store_path)
    assert (transformed == transformed_by_config).all(None)


@pytest.mark.smoke()
def test_linearize_basic(data_nonlin_var):
    """
    Test that linearize creates new column and returns self.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    result = preprocessor.linearize("target", "feature_1", transformed_name="target_lin")
    assert result is preprocessor  # method chaining
    assert "target_lin" in preprocessor.data().columns


@pytest.mark.unit()
def test_linearize_formula(data_nonlin_var):
    """
    Test that linearized values satisfy: linearized = num - ratio * denom.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    preprocessor.linearize("target", "feature_1", transformed_name="target_lin")
    df = preprocessor.data()
    transformer = preprocessor.transformations()[-1]
    ratio = transformer.ratio
    expected = data_nonlin_var["target"] - ratio * data_nonlin_var["feature_1"]
    np.testing.assert_allclose(df["target_lin"].values, expected.values, rtol=1e-10)


@pytest.mark.unit()
def test_linearize_in_chain(data_nonlin_var):
    """
    Test linearize as part of a preprocessing chain.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    result = (
        preprocessor.robust("feature_1", alpha=0.01)
        .linearize("target", "feature_1", transformed_name="target_lin")
        .data()
    )
    assert "target_lin" in result.columns


@pytest.mark.unit()
def test_linearize_load_store(data_nonlin_var):
    """
    Test that linearization transformer can be serialized and replayed.
    """
    store_path = "tests/configs/linearize_config.json"
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    preprocessor.linearize("target", "feature_1", transformed_name="target_lin")
    preprocessor.store_transformations(store_path)

    loaded_preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    loaded_preprocessor.load_transformations(store_path)

    os.remove(store_path)

    for t, lt in zip(preprocessor.transformations(), loaded_preprocessor.transformations()):
        assert t.get_params_dict() == lt.get_params_dict()


@pytest.mark.unit()
def test_linearize_default_name(data_nonlin_var):
    """
    Test that default transformed_name is '{numerator}_lin'.
    """
    preprocessor = Preprocessor(data_nonlin_var, verbose=False)
    preprocessor.linearize("target", "feature_1")
    assert "target_lin" in preprocessor.data().columns
