import numpy as np
from typing import Tuple

def check_and_convert_array(y_true, y_score, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """An auxiliary function for converting input arrays to np.array and checks.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (0 or 1).

    y_score : array-like of shape (n_samples,)
        Model's predictions.

    K : int
        The first K ranked elements.

    Returns
    -------
    y_true : ndarray of shape (n_samples,)
        True class labels (0 or 1).

    y_score : ndarray of shape (n_samples,)
        Model's predictions.
    """
    
    y_true, y_score = np.array(y_true), np.array(y_score)
    if not (y_true.shape == y_score.shape) :
        raise ValueError("The dimensions of the source arrays must match. " +\
                        f"Dimension y_true = {y_true.shape}, dimension y_score = {y_score.shape}.")
    if not (y_true.ndim == 1):
        raise ValueError(f"The dimension of the source arrays must be equal to 1, the dimension of the source arrays = {y_true.ndim}.")
    if not (len(y_true) > 0):
        raise ValueError("The length of the source arrays must be greater than 0.")
    if not (K > 0):
        raise ValueError("K must be greater than 0.")
    if not (K <= len(y_true)):
        raise ValueError("K cannot be greater than the length of the original arrays. " +\
                        f"The length of the source arrays = {len(y_true)}, K = {K}.")
    return (y_true, y_score)
    
def rank_target(y_true: np.ndarray, y_score: np.ndarray, K: int) -> np.ndarray:
    """An auxiliary function for ranking the true class labels in descending order of the values of the Model's predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (0 or 1).

    y_score : array-like of shape (n_samples,)
        Model's predictions.
        
    K : int
        The first K ranked elements.

    Returns
    -------
    y_ranked : ndarray of shape (n_samples,)
        The first K labels after ranking.
    """

    idx_sort = y_score.argsort()[::-1]
    return y_true[idx_sort][:K]