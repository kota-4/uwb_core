# 測位結果における測位誤差を評価するための関数群
# 平均測位誤差
# RMSE (Root Mean Square Error)
# MAE (Mean Absolute Error)

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 平均測位誤差 (従来のichiさん方式のまま)
def calculate_Localization_Accuracy(predictions, labels):
    """平均測位誤差を求める"""
    # どのような型の入力でも、安全にNumPy配列(float64)に変換する
    pred_arr = np.array(predictions, dtype=np.float64)
    lab_arr = np.array(labels, dtype=np.float64)
    return np.mean(np.sqrt(np.sum((pred_arr - lab_arr)**2, axis=1)))

# 平均測位誤差 50cm1グリッド入力版
def calculate_Localization_Accuracy_50cm(predictions, labels):
    """平均測位誤差を求める (50cmグリッド入力版)"""
    # どのような型の入力でも、安全にNumPy配列(float64)に変換する
    pred_arr = np.array(predictions, dtype=np.float64)
    lab_arr = np.array(labels, dtype=np.float64)
    return np.mean(np.sqrt(np.sum((pred_arr*50 - lab_arr*50)**2, axis=1)))

# RMSE (Root Mean Square Error)
def calculate_RMSE(predictions, labels):
    """RMSEを求める"""
    # どのような型の入力でも、安全にNumPy配列(float64)に変換する
    pred_arr = np.array(predictions, dtype=np.float64)
    lab_arr = np.array(labels, dtype=np.float64)
    return np.sqrt(mean_squared_error(lab_arr, pred_arr))

# MAE (Mean Absolute Error)
def calculate_MAE(predictions, labels):
    """MAEを求める"""
    # どのような型の入力でも、安全にNumPy配列(float64)に変換する
    pred_arr = np.array(predictions, dtype=np.float64)
    lab_arr = np.array(labels, dtype=np.float64)
    return mean_absolute_error(lab_arr, pred_arr)