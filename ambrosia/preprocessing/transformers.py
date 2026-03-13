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
Module contains tools for metrics transformations during a
preprocessing task.
"""
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as sps

from ambrosia import types
from ambrosia.tools.ab_abstract_component import AbstractFittableTransformer
from ambrosia.tools.back_tools import wrap_cols


class BoxCoxTransformer(AbstractFittableTransformer):
    """
    Unit for a Box-Cox transformation of the pandas data.

    A Box Cox transformation helps to transform non-normal dependent variables
    into a normal shape. All variables values must be positive.

    Optimal transformation lambdas are selected automatically during
    the transformer fit process.


    Attributes
    ----------
    column_names : List
        Names of column which will be selected for the transformation.
    lambda_ : np.ndarray
        Array of parameters using during the transformation of the
        selected columns.
    fitted : bool
        Fit flag.

    Examples
    --------
    >>> boxcox = BoxCoxTransformer()
    >>> boxcox.fit(dataframe, ['column1', 'column2'])
    >>> boxcox.transform(dataframe, inplace=True)

    """

    def __str__(self) -> str:
        return "Box-Cox transformation"

    def __init__(
        self,
    ) -> None:
        """
        BoxCoxTransformer class constructor.
        """
        self.column_names = None
        self.lambda_ = None
        super().__init__()

    def __calculate_lambda_(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        columns_num: int = len(self.column_names)
        self.lambda_ = np.zeros(columns_num)
        X: np.ndarray = dataframe[self.column_names].values
        for num in range(columns_num):
            self.lambda_[num] = sps.boxcox(X[:, num])[1]

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.

        Returns
        -------
        params : Dict
            Dictionary with fitted params.
        """
        self._check_fitted()
        return {
            "column_names": self.column_names,
            "lambda_": self.lambda_.tolist(),
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load instance parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if "column_names" in params:
            self.column_names = params["column_names"]
        else:
            raise TypeError(f"params argument must contain: {'column_names'}")
        if "lambda_" in params:
            self.lambda_ = np.array(params["lambda_"])
        else:
            raise TypeError(f"params argument must contain: {'lambda_'}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ):
        """
        Fit to calculate transformation parameters for the selected columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to calculate optimal transformation parameters.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.

        Returns
        -------
        self : object
            Instance object.
        """
        self.column_names = wrap_cols(column_names)
        self._check_cols(dataframe, self.column_names)
        self.__calculate_lambda_(dataframe)
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Apply Box-Cox transformation for the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        X: np.ndarray = transformed[self.column_names].values
        for num in range(len(self.column_names)):
            if self.lambda_[num] == 0:
                X[:, num] = np.log(X[:, num])
            else:
                X[:, num] = (X[:, num] ** self.lambda_[num] - 1) / self.lambda_[num]
        transformed[self.column_names] = X
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit transformer parameters using given dataframe and transform it.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe for calculation of optimal parameters and further
            transformation.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)

    def inverse_transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Apply inverse Box-Cox transformation for the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to inverse transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        X_tr: np.ndarray = transformed[self.column_names].values
        for num in range(len(self.column_names)):
            if self.lambda_[num] == 0:
                X_tr[:, num] = np.exp(X_tr[:, num])
            else:
                X_tr[:, num] = (X_tr[:, num] * self.lambda_[num] + 1) ** (1 / self.lambda_[num])
        transformed[self.column_names] = X_tr
        return None if inplace else transformed


class LogTransformer(AbstractFittableTransformer):
    """
    Unit for a logarithmic transformation of the pandas data.

    A logarithmic transformation helps to transform some metrics distributions
    into a more normal shape and reduce the variance.
    All metrics values must be positive.


    Attributes
    ----------
    column_names : List
        Names of column which will be selected for the transformation.
    fitted : bool
        Fit flag.

    Examples
    --------
    >>> log = LogTransformer()
    >>> log.fit(dataframe, ['column1', 'column2'])
    >>> log.transform(dataframe, inplace=True)

    """

    def __str__(self) -> str:
        return "Logarithmic transformation"

    def __init__(self) -> None:
        """
        LogTransformer class constructor.
        """
        self.column_names = None
        super().__init__()

    def get_params_dict(self) -> Dict:
        """
        Returns a dictionary with params.
        """
        self._check_fitted()
        return {
            "column_names": self.column_names,
        }

    def load_params_dict(self, params: Dict) -> None:
        """
        Load instance parameters from the dictionary.

        Parameters
        ----------
        params : Dict
            Dictionary with params.
        """
        if "column_names" in params:
            self.column_names = params["column_names"]
        else:
            raise TypeError(f"params argument must contain: {'column_names'}")
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
    ):
        """
        Fit names of the selected columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with metrics.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.

        Returns
        -------
        self : object
            Instance object.
        """
        self.column_names = wrap_cols(column_names)
        self._check_cols(dataframe, self.column_names)
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Apply log transformation for the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        if (transformed[self.column_names] > 0).all(axis=None):
            transformed[self.column_names] = np.log(transformed[self.column_names].values)
        else:
            raise ValueError(f"All values in columns {self.column_names} must be positive")
        return None if inplace else transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        column_names: types.ColumnNamesType,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit transformer parameters using given dataframe and transform it.

        Only column names are fittable.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        column_names : ColumnNamesType
            One or number of columns in the dataframe.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self.fit(dataframe, column_names)
        return self.transform(dataframe, inplace)

    def inverse_transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Apply inverse log transformation for the data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to inverse transform.
        inplace : bool, default: ``False``
            If ``True`` transforms the given dataframe, otherwise copy and
            returns an another one.

        Returns
        -------
        df : Union[pd.DataFrame, None]
            Transformed dataframe or None
        """
        self._check_fitted()
        self._check_cols(dataframe, self.column_names)
        transformed: pd.DataFrame = dataframe if inplace else dataframe.copy()
        transformed[self.column_names] = np.exp(transformed[self.column_names].values)
        return None if inplace else transformed


class LinearizationTransformer(AbstractFittableTransformer):
    """
    Linearization transformer for ratio metrics.

    Converts a ratio metric (numerator / denominator) into a per-unit linearized
    metric that is approximately normally distributed, enabling correct t-test usage:

        linearized_i = numerator_i - ratio * denominator_i

    where ratio = mean(numerator) / mean(denominator), estimated on the reference
    (control group / historical) data passed to fit().

    Parameters
    ----------
    numerator : str
        Column name of the ratio numerator (e.g. "revenue").
    denominator : str
        Column name of the ratio denominator (e.g. "orders").
    transformed_name : str, optional
        Name for the new column. Defaults to ``"{numerator}_lin"``.

    Examples
    --------
    >>> transformer = LinearizationTransformer()
    >>> transformer.fit(control_df, "revenue", "orders", "arpu_lin")
    >>> transformer.transform(experiment_df, inplace=True)
    """

    def __str__(self) -> str:
        return "Linearization transformation"

    def __init__(self) -> None:
        self.numerator: Optional[str] = None
        self.denominator: Optional[str] = None
        self.transformed_name: Optional[str] = None
        self.ratio: Optional[float] = None
        super().__init__()

    def get_params_dict(self) -> Dict:
        self._check_fitted()
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "transformed_name": self.transformed_name,
            "ratio": self.ratio,
        }

    def load_params_dict(self, params: Dict) -> None:
        for key in ("numerator", "denominator", "transformed_name", "ratio"):
            if key not in params:
                raise TypeError(f"params argument must contain: {key}")
            setattr(self, key, params[key])
        self.fitted = True

    def fit(
        self,
        dataframe: pd.DataFrame,
        numerator: str,
        denominator: str,
        transformed_name: Optional[str] = None,
    ):
        """
        Estimate ratio = mean(numerator) / mean(denominator) on reference data.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Reference dataframe (typically control group or historical data).
        numerator : str
            Column name of the ratio numerator.
        denominator : str
            Column name of the ratio denominator.
        transformed_name : str, optional
            Name for the linearized column. Defaults to ``"{numerator}_lin"``.
        """
        self._check_cols(dataframe, [numerator, denominator])
        denom_mean = dataframe[denominator].mean()
        if denom_mean == 0:
            raise ValueError(f"Mean of denominator column '{denominator}' is zero; cannot compute ratio.")
        self.numerator = numerator
        self.denominator = denominator
        self.transformed_name = transformed_name if transformed_name is not None else f"{numerator}_lin"
        self.ratio = dataframe[numerator].mean() / denom_mean
        self.fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame, inplace: bool = False) -> Union[pd.DataFrame, None]:
        """
        Apply linearization: transformed = numerator - ratio * denominator.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to transform.
        inplace : bool, default: ``False``
            If ``True`` modifies dataframe in place, otherwise returns a copy.
        """
        self._check_fitted()
        self._check_cols(dataframe, [self.numerator, self.denominator])
        df = dataframe if inplace else dataframe.copy()
        df[self.transformed_name] = df[self.numerator] - self.ratio * df[self.denominator]
        return None if inplace else df

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        numerator: str,
        denominator: str,
        transformed_name: Optional[str] = None,
        inplace: bool = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Fit and transform in one step.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Reference dataframe for fitting and transformation.
        numerator : str
            Column name of the ratio numerator.
        denominator : str
            Column name of the ratio denominator.
        transformed_name : str, optional
            Name for the linearized column.
        inplace : bool, default: ``False``
            If ``True`` modifies dataframe in place.
        """
        self.fit(dataframe, numerator, denominator, transformed_name)
        return self.transform(dataframe, inplace)
