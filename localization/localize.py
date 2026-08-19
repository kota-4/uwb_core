# 測位手法をまとめたモジュール
# SVR回帰
# MLP回帰
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# SVM回帰によるFP測位
# C=1000, gamma=0.01 はsmall_scale_envでの最適値
def SVR_localization(train_data, train_labels, test_data, C=1000, gamma=0.01):
    clf_x = SVR(C=C, gamma=gamma)
    clf_y = SVR(C=C, gamma=gamma)
    # .iloc[:, 0] で「全ての行(:)、0番目の列」を抽出
    y_x = train_labels.iloc[:, 0]
    # .iloc[:, 1] で「全ての行(:)、1番目の列」を抽出
    y_y = train_labels.iloc[:, 1]

    # 抽出したデータを使って学習
    clf_x.fit(train_data, y_x)
    clf_y.fit(train_data, y_y)
    
    # 予測
    test_x = clf_x.predict(test_data)
    test_y = clf_y.predict(test_data)
    predictions = np.column_stack((test_x, test_y))

    # 結果をDataFrameに変換
    predictions_df = pd.DataFrame(predictions, columns=['X', 'Y'])

    return predictions_df

# MLP回帰によるFP測位
def MLP_localization(train_data, train_labels, test_data):
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, solver='adam', activation='relu')
    mlp_regressor.fit(train_data, train_labels)

    # テストデータでの予測
    mlp_predictions = mlp_regressor.predict(test_data)

    return mlp_predictions