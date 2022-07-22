import numpy as np
from ..utils import check_and_convert_array
from ..utils import rank_target

def precision_at_k(y_true, y_pred, K: int) -> np.float64:
    """
    Расчет Precision@K.
    Parameters
    ----------
        y_true : Истинные метки классов (0 или 1).
        y_pred : Предсказания модели (вероятности).
        K : K первых ранжированных элементов.
    """
    y_true, y_pred = check_and_convert_array(y_true, y_pred, K)
    y_ranked = rank_target(y_true, y_pred, K)
    return y_ranked.mean()

def recall_at_k(y_true, y_pred, K: int) -> np.float64:
    """
    Расчет Recall@K.

    Parameters
    ----------
    y_true : Истинные метки классов (0 или 1).
    y_pred : Предсказания модели (вероятности).
    K : K первых ранжированных элементов.
    """
    y_true, y_pred = check_and_convert_array(y_true, y_pred, K)
    y_ranked = rank_target(y_true, y_pred, K)
    return y_ranked.sum() / y_true.sum() if y_true.sum() else 0

def average_precision_at_k(y_true, y_pred, K: int) -> np.float64:
    """
    Расчет AP@K.

    Parameters
    ----------
    y_true : Истинные метки классов (0 или 1).
    y_pred : Предсказания модели (вероятности).
    K : K первых ранжированных элементов.
    """
    y_true, y_pred = check_and_convert_array(y_true, y_pred, K)
    y_ranked = rank_target(y_true, y_pred, K)
    prediction_list = list(map(lambda k: y_ranked[:k + 1].mean(), range(K)))
    return (y_ranked * prediction_list).mean()

def discounted_cumulative_gain_at_k(y_true, y_pred, K: int) -> np.float64:
    """
    Расчет DCG@K. Sourse : https://en.wikipedia.org/wiki/Discounted_cumulative_gain.

    Parameters
    ----------
    y_true : Истинные метки классов (0 или 1).
    y_pred : Предсказания модели (вероятности).
    K : K первых ранжированных элементов.
    """
    y_true, y_pred = check_and_convert_array(y_true, y_pred, K)
    y_ranked = rank_target(y_true, y_pred, K)
    return ((2 ** y_ranked - 1) / np.log2(np.arange(K) + 2)).sum()

def normalized_discounted_cumulative_gain_at_k(y_true, y_pred, K: int) -> np.float64:
    """
    Расчет nDCG@K.

    Parameters
    ----------
    y_true : Истинные метки классов (0 или 1).
    y_pred : Предсказания модели (вероятности).
    K : K первых отранжированных элементов.
    """
    DSG_K = discounted_cumulative_gain_at_k(y_true, y_pred, K)
    IDSG_K = discounted_cumulative_gain_at_k(np.ones_like(y_true), y_pred, K)
    return DSG_K / IDSG_K