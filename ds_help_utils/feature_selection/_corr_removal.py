import numpy as np
import pandas as pd
from typing import Callable, Union
from sklearn.base import BaseEstimator, TransformerMixin

class HighCorrRemoval(BaseEstimator, TransformerMixin):
    """Selector to remove highly correlated features.

    Parameters
    ----------
    corr_threshold : float, default=0.9
        Correlation threshold. 
        
    method : str, default="pearson"
        Method of correlation:

        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        * callable: callable with input two 1d ndarrays
            and returning a float. Note that the returned matrix from corr
            will have 1 along the diagonals and will be symmetric
            regardless of the callable's behavior.
    
    Attributes
    ----------
    selected_features_ : List[str]
        List of selected features.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from ds_help_utils.feature_selection import HighCorrRemoval
    >>> X, y = make_classification(n_samples=5, n_features=5, n_informative=3, random_state=42)
    >>> X = pd.DataFrame(X, index = [f"feature_{i}" for i in range(X.shape[1])])
    >>> X["feature_x"] = 1
    >>> print(X.corr())
                feature_0  feature_1  feature_2  feature_3  feature_4  feature_x
    feature_0   1.000000   0.946395  -0.885197   0.631472   0.319387        NaN
    feature_1   0.946395   1.000000  -0.786504   0.484044   0.069043        NaN
    feature_2  -0.885197  -0.786504   1.000000  -0.916994  -0.128859        NaN
    feature_3   0.631472   0.484044  -0.916994   1.000000   0.031427        NaN
    feature_4   0.319387   0.069043  -0.128859   0.031427   1.000000        NaN
    feature_x        NaN        NaN        NaN        NaN        NaN        NaN
    >>> removal = HighCorrRemoval(corr_threshold=0.9, method="pearson")
    >>> removal.fit(X, y)
    HighCorrRemoval()
    >>> print(removal.transform(X))
       feature_1  feature_3  feature_4
    0  -0.663121  -0.964093   0.492927
    1  -1.304917  -0.646420  -0.020722
    2  -1.215860   0.736990   0.540227
    3   0.440901   1.056313   0.258791
    4  -0.769279   0.168486  -0.674272
    """
    def __init__(self, corr_threshold: float = 0.9, method: Union[str, Callable]="pearson") -> None:
        self.corr_threshold = corr_threshold
        self.method = method

    def fit(self, X: pd.DataFrame, y=None):
        """Select features with non-high correlation.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Data frame with features.
        y : None
            Ignored.
        
        Returns
        -------
        self : object
            Fitted removal.
        """
        corr = X.corr(method=self.method)

        sl = np.triu(corr.abs() > self.corr_threshold, k=1)
        grid_x, grid_y = np.meshgrid(np.arange(len(sl)), np.arange(len(sl)))

        removed = []

        for x, y in zip(grid_x[sl], grid_y[sl]):
            if x not in removed:
                removed.append(y)

        const = np.arange(len(corr))[np.isnan(np.diagonal(corr))]

        removed += [*const]

        self.selected_features_ = [feature for (idx, feature) 
                                   in enumerate(X.columns) 
                                   if idx not in removed]
        
        return self
        
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Return features with non-high correlation.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Data frame with features.
        y : None
            Ignored.
        
        Returns
        -------
        X_selected : pandas.core.frame.DataFrame
            Selected data frame.
        """
        return X[self.selected_features_]