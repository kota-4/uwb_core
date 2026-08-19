import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union, Optional
import joblib
from pathlib import Path

class DataScaler:
    """
    データの正規化・標準化を行うラッパークラス。
    MinMaxScaler (正規化) や StandardScaler (標準化) を切り替えて使用可能。
    """
    def __init__(self, method: str = 'minmax'):
        """
        Args:
            method (str): 
                - 'minmax': 最小0, 最大1 (既存コードの動作)
                - 'standard': 平均0, 分散1 (SVM等で一般的に推奨)
                - 'robust': 中央値と四分位範囲を使用 (外れ値に強い)
        """
        self.method = method
        self.scaler = None
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        スケーラーをデータに合わせて学習(fit)させる。
        """
        self.scaler.fit(data)

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        学習済みスケーラーを使ってデータを変換(transform)する。
        DataFrameが渡された場合、カラム名とインデックスを保持して返す。
        """
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")

        # dataがDataFrameでない場合、values属性を持たないので注意
        if isinstance(data, pd.DataFrame):
            scaled_values = self.scaler.transform(data)
            return pd.DataFrame(
                scaled_values, 
                columns=data.columns, 
                index=data.index
            )
        else:
            # numpy arrayの場合はそのまま変換してDataFrame化（カラム名なし）
            scaled_values = self.scaler.transform(data)
            return pd.DataFrame(scaled_values)

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """学習と変換を一度に行う"""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        正規化されたデータを元のスケールに戻す。
        """
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        original_values = self.scaler.inverse_transform(data)
        
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(
                original_values, 
                columns=data.columns, 
                index=data.index
            )
        return pd.DataFrame(original_values)

    def save(self, path: Union[str, Path]):
        """スケーラーをファイル保存 (推論時用に便利)"""
        joblib.dump(self.scaler, path)

    def load(self, path: Union[str, Path]):
        """スケーラーを読み込み"""
        self.scaler = joblib.load(path)