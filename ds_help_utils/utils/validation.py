import numpy as np
from typing import Tuple

def check_and_convert_array(y_true, y_pred, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вспомогательная функция для преобразования входных массивов в np.array и проверок.
    Parameters
    ----------
        y_true : Истинные метки классов (0 или 1).
        y_pred : Предсказания модели (вероятности).
        K : K первых ранжированных элементов.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if not (y_true.shape == y_pred.shape) :
        raise ValueError(f"Размерности исходных массивов должны совпадать. " +\
                         "Размерность y_true = {y_true.shape}, размерность y_pred = {y_pred.shape}.")
    if not (y_true.ndim == 1):
        raise ValueError(f"Размерность исходных массивов должна быть равна 1, размерность исходных массивов = {y_true.ndim}.")
    if not (len(y_true) > 0):
        raise ValueError("Длина исходных массивов должна быть больше 0.")
    if not (K > 0):
        raise ValueError("K должен быть больше 0.")
    if not (K <= len(y_true)):
        raise ValueError(f"K не может быть больше длины исходных массивов. Длина исходных массивов = {len(y_true)}, K = {K}.")
    return (y_true, y_pred)
    
def rank_target(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    """
    Вспомогательная функция для ранжирования истинных меток классов по убыванию значений предсказаний модели.
    Parameters
    ----------
        y_true : Истинные метки классов (0 или 1).
        y_pred : Предсказания модели (вероятности).
        K : K первых ранжированных элементов.
    """
    idx_sort = y_pred.argsort()[::-1]
    return y_true[idx_sort][:K]