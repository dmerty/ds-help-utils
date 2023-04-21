import numpy as np
from ..utils import check_and_convert_array, rank_target

def precision_at_k(y_true, y_score, K: int) -> np.float64:
    """Calculation Precision@K.

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
    Precision@K : float

    Examples
    --------
    >>> import numpy as np
    >>> from ds_help_utils.metrics import precision_at_k
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> K = 3
    >>> precision_at_k(y_true, y_scores, 3)
    0.67...
    """

    y_true, y_score = check_and_convert_array(y_true, y_score, K)
    y_ranked = rank_target(y_true, y_score, K)
    return y_ranked.mean()

def recall_at_k(y_true, y_score, K: int) -> np.float64:
    """Calculation Recall@K.

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
    Recall@K : float

    Examples
    --------
    >>> import numpy as np
    >>> from ds_help_utils.metrics import recall_at_k
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> K = 3
    >>> recall_at_k(y_true, y_scores, 3)
    1.0
    """

    y_true, y_score = check_and_convert_array(y_true, y_score, K)
    y_ranked = rank_target(y_true, y_score, K)
    return y_ranked.sum() / y_true.sum() if y_true.sum() else 0

def average_precision_at_k(y_true, y_score, K: int) -> np.float64:
    """Calculation AP@K.

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
    AP@K : float

    Examples
    --------
    >>> import numpy as np
    >>> from ds_help_utils.metrics import average_precision_at_k
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> K = 3
    >>> average_precision_at_k(y_true, y_scores, 3)
    0.56...
    """

    y_true, y_score = check_and_convert_array(y_true, y_score, K)
    y_ranked = rank_target(y_true, y_score, K)
    prediction_list = list(map(lambda k: y_ranked[:k + 1].mean(), range(K)))
    return (y_ranked * prediction_list).mean()

def discounted_cumulative_gain_at_k(y_true, y_score, K: int) -> np.float64:
    """Calculation DCG@K. Sourse : https://en.wikipedia.org/wiki/Discounted_cumulative_gain.

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
    DCG@K : float

    Examples
    --------
    >>> import numpy as np
    >>> from ds_help_utils.metrics import discounted_cumulative_gain_at_k
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> K = 3
    >>> discounted_cumulative_gain_at_k(y_true, y_scores, 3)
    1.5
    """

    y_true, y_score = check_and_convert_array(y_true, y_score, K)
    y_ranked = rank_target(y_true, y_score, K)
    return ((2 ** y_ranked - 1) / np.log2(np.arange(K) + 2)).sum()

def normalized_discounted_cumulative_gain_at_k(y_true, y_score, K: int) -> np.float64:
    """Calculation nDCG@K.

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
    nDCG@K : float

    Examples
    --------
    >>> import numpy as np
    >>> from ds_help_utils.metrics import normalized_discounted_cumulative_gain_at_k
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> K = 3
    >>> normalized_discounted_cumulative_gain_at_k(y_true, y_scores, 3)
    0.7...
    """

    DSG_K = discounted_cumulative_gain_at_k(y_true, y_score, K)
    IDSG_K = discounted_cumulative_gain_at_k(np.ones_like(y_true), y_score, K)
    return DSG_K / IDSG_K