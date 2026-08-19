# パラメータ郡を一元管理するクラス

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ExperimentConfig:
    """
    実験のパラメータ設定を一元管理するクラス
    """
    # --- 必須入力パラメータ ---
    who: str
    env: str
    source_rp: str
    target_rp: str
    tag_number: int
    ap_ids: Dict[str, int]
    
    # --- 自動設定パラメータ (init=False にして自動生成させる) ---
    AP_location: Dict[str, List[float]] = field(init=False)
    wall_location: List[float] = field(init=False)
    RP: List[List[int]] = field(init=False)
    Pattern_name: str = field(init=False)
    save_label: str = field(init=False)
    C: float = field(init=False, default=1000)  # SVRのCパラメータのデフォルト値
    gamma: float = field(init=False, default=0.01)  # SVRのgammaパラメータのデフォルト値
    
    def __post_init__(self):
        # パターン名の生成
        self.Pattern_name = f"{self.source_rp}_to_{self.target_rp}"
        # 保存ラベルの生成
        if self.env == "small_scale_env":
            env_name = "small"
        elif self.env == "big_scale_env":
            env_name = "big"
        else:
            env_name = "unknown"
        self.save_label = f"{self.who}_{env_name}_{self.source_rp[:3]}_{self.target_rp[:3]}"

        # 環境ごとのパラメータ設定
        if self.env == "small_scale_env":
            self._set_small_scale_params()
        elif self.env == "big_scale_env":
            self._set_big_scale_params()
        else:
            raise ValueError(f"Unknown environment: {self.env}")

    def _set_small_scale_params(self):
        """小規模環境用のパラメータ設定"""
        self.AP_location = {
            'AP5307': [0, 6], 
            'AP4524': [0, -3], 
            'AP1805': [4, -3]
        }
        self.wall_location = [11.46, -7.76, -6, 19]
        # range(4) -> 0,1,2,3 の 4x4 グリッド
        self.RP = [[i, j] for i in range(4) for j in range(4)]
        self.C = 1000
        self.gamma = 0.01

    def _set_big_scale_params(self):
        """大規模環境用のパラメータ設定"""
        self.AP_location = {
            'AP5307': [0, 15], 
            'AP4524': [0, -9], 
            'AP1805': [8, 9]
        }
        self.wall_location = [
        -16.4, # (0,0)からホワイトボード側の壁までの距離 (x軸)
        21.88, # (0,0)からホワイトボードの反対側の壁までの距離 (x軸)
        -8.2, # (0,0)からパーテーションまでの距離 (y軸)
        17.28 # (0,0)からパーテーションの反対側までの距離 (y軸)
        # -7.4, # パーテーション側の柱と0列目までの距離 (y軸)
        # 15.8 # 窓側真ん中柱と0列目までの距離 (y軸)
        ]
        # range(8) -> 0,1,...,7 の 8x8 グリッド
        self.RP = [[i, j] for i in range(8) for j in range(8)]
        self.C = 1000
        self.gamma = 0.1 # 大規模環境ではgammaを0.1に設定 (小規模環境では0.01)