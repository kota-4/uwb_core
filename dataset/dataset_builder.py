from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
from scipy.signal import find_peaks

# 型ヒントのためにのみインポート (実行時の依存回避)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from uwb_core.environment_config.experiment_config import ExperimentConfig

class UwbDatasetBuilder:
    """
    CSVファイル群から学習用・テスト用のデータセットを構築するクラス。
    Configへの依存をメソッド注入型に変更し、生データ読み込み時の循環参照を回避。
    """

    def __init__(self):
        # Configはインスタンス保持しない（メソッドごとに必要な場合に渡す）
        pass

    # ---------------------------------------------------------
    # Helper: 10進数/16進数 混在対応の変換関数
    # ---------------------------------------------------------
    def _safe_hex_or_int_conversion(self, series: pd.Series) -> pd.Series:
        """
        シリーズ内の値を数値に変換する。
        1. まず10進数として変換
        2. 失敗した箇所(NaN)について、16進数文字列とみなして変換
        """
        # 標準的な数値変換 (10進数) を試みる
        # "1234" -> 1234, "error" -> NaN, "0x1234" -> NaN (これだと困るケースを救う)
        nums = pd.to_numeric(series, errors='coerce')
        mask_nan = nums.isna()
        
        if mask_nan.any():
            def convert_hex(x):
                try:
                    # 文字列にして空白除去
                    s = str(x).strip()
                    # '0x' があってもなくても int(s, 16) は処理可能
                    # ただし 'nan' や 空文字 はエラーになるので除外
                    if s.lower() == 'nan' or s == '':
                        return np.nan
                    return int(s, 16)
                except (ValueError, TypeError):
                    return np.nan

            # NaNだった行の元の値に対してのみ apply をかける (全行にかけるより高速)
            hex_converted = series[mask_nan].apply(convert_hex)
            nums[mask_nan] = hex_converted

        return nums

    # ---------------------------------------------------------
    # 1. Raw Data Loading
    # ---------------------------------------------------------
    def read_raw_data(self, dir_path: str, env_label: str) -> pd.DataFrame:
        """
        指定ディレクトリ内のCSVファイルを読み込み、結合する。
        
        Args:
            dir_path: データディレクトリのパス
            env_label: 'big_scale_env' などの環境ラベル (フィルタリング用)
        """
        path_obj = Path(dir_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        df_list = []
        csv_files = sorted(list(path_obj.glob("*.csv")))
        
        if not csv_files:
            print(f"Warning: No csv files found in {dir_path}")
            return pd.DataFrame()
        # 型変換が必要なカラムの定義
        # 1. アドレス系 (16進数と10進数が混在する可能性があるもの)
        addr_cols = ['saddr', 'daddr', 'taddr']
        # 2. 数値系 (float64として扱うべきもの)
        val_cols = ['rng_rng', 'fsl', 'rsl', 'rng_raw', 'rpc', 'rng_fp', 'r_pwr', 'f_pwr'] # 'rng_fp', 'r_pwr', 'f_pwr'は今後の拡張に備えて追加

        for file_path in csv_files:
            try:
                # 座標取得
                coords = get_coordinates_from_filename(str(file_path))
                if len(coords) < 2:
                    continue
                
                # 文字列として読み込み (Hex対応のため)
                # dtype=str を指定することで、勝手にfloatにされたりするのを防ぐ
                temp_df = pd.read_csv(file_path, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=str)
                
                # 座標カラム追加
                temp_df.insert(0, 'X', coords[0])
                temp_df.insert(1, 'Y', coords[1])
                
                # --- 網羅的な型変換処理 ---
                # アドレス系の変換 (saddr, daddr, taddr)
                for col in addr_cols:
                    if col in temp_df.columns:
                        temp_df[col] = self._safe_hex_or_int_conversion(temp_df[col])
                
                # 数値系の変換 (rng_rng, fsl, rsl, rng_raw, rpc 等)
                for col in val_cols:
                    if col in temp_df.columns:
                        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

                # 必須データの欠損行を削除 (特にdaddr, taddr, rslがNaNの行は以降の処理に支障が出るため)
                temp_df.dropna(subset=['daddr', 'taddr', 'rsl'], inplace=True)
                
                # アドレス系を int64 へキャスト
                # (to_numericを適用した直後は float64 になっているため、整数に戻す)
                for col in addr_cols:
                    if col in temp_df.columns:
                        temp_df[col] = temp_df[col].astype(int)
                
                # Big Scale環境の特例 (taddr=0の除外)
                if env_label == 'big_scale_env':
                    temp_df = temp_df[temp_df['taddr'] != 0]

                df_list.append(temp_df)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue

        if not df_list:
            return pd.DataFrame()

        return pd.concat(df_list, ignore_index=True)

    # ---------------------------------------------------------
    # 2. Feature Extraction
    # ---------------------------------------------------------
    def create_wide_feature_dataframe(self, raw_df: pd.DataFrame, config: 'ExperimentConfig') -> pd.DataFrame:
        """
        生データからAPごとの特徴量を抽出し、横持ち(Wide)形式に変換する
        """
        # ConfigからID取得
        tag_id = config.tag_number
        target_ap_ids = config.ap_ids

        # タグIDフィルタ
        tag_df = raw_df[(raw_df['daddr'] == tag_id) | (raw_df['taddr'] == tag_id)].copy()

        grouped = tag_df.groupby(['X', 'Y'])
        all_complete_rows = []

        for (loc_x, loc_y), group_df in grouped:
            ap_features_list = []
            
            for ap_name, ap_id in target_ap_ids.items():
                # AP IDフィルタ
                per_ap_df = group_df[(group_df['daddr'] == ap_id) | (group_df['taddr'] == ap_id)].copy()
                per_ap_df = per_ap_df[per_ap_df['rsl'] >= -1000]

                if per_ap_df.empty:
                    continue

                per_ap_df.reset_index(drop=True, inplace=True)

                # CIR整形
                per_ap_df['cirData'] = per_ap_df['cirData'].apply(
                    lambda x: (x[2:-2] + '0' * (1024 - len(x[2:-2]))) 
                    if isinstance(x, str) and len(x) < 1024 else (x[2:-2] if isinstance(x, str) else x)
                )

                # 特徴量抽出
                # 優先順位: rng_raw > rng_rng どちらを使うか明示的に判定する
                # 一時的変更: rng_raw → rng_rng
                # if 'rng_raw' in per_ap_df.columns:
                #     target_rng_col = 'rng_raw'
                # elif 'rng_rng' in per_ap_df.columns:
                #     target_rng_col = 'rng_rng'
                if 'rng_rng' in per_ap_df.columns:
                    target_rng_col = 'rng_rng'
                elif 'rng_raw' in per_ap_df.columns:
                    target_rng_col = 'rng_raw'
                else:
                    # 両方ない場合はスキップ（またはエラー）
                    print(f"Warning: neither rng_raw nor rng_rng found in AP {ap_name}")
                    continue
                rssi_cols = per_ap_df[[target_rng_col, 'rsl']].copy()
                
                # CIR特徴量
                amplitudes = hex_to_amplitude_list(per_ap_df['cirData'])
                features_df = extract_features_from_amplitude(amplitudes)
                
                features_df.index = per_ap_df.index
                combined_ap_df = pd.concat([rssi_cols, features_df], axis=1)
                
                # カラム名付与
                combined_ap_df.columns = [f"{ap_name}_{col}" for col in combined_ap_df.columns]
                ap_features_list.append(combined_ap_df)

            # 全AP揃っているか確認して結合
            if len(ap_features_list) == len(target_ap_ids):
                location_wide_df = pd.concat(ap_features_list, axis=1)
                location_wide_df.dropna(how='any', inplace=True)
                
                if not location_wide_df.empty:
                    location_wide_df['X'] = loc_x
                    location_wide_df['Y'] = loc_y
                    all_complete_rows.append(location_wide_df)

        if not all_complete_rows:
            return pd.DataFrame()

        final_df = pd.concat(all_complete_rows, ignore_index=True)
        cols = ['X', 'Y'] + [c for c in final_df.columns if c not in ['X', 'Y']]
        return final_df[cols]

    # ---------------------------------------------------------
    # 3. Geometric Features
    # ---------------------------------------------------------
    def add_geometric_features(self, 
                               base_df: pd.DataFrame, 
                               predictions_df: pd.DataFrame,
                               config: 'ExperimentConfig') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        幾何学的特徴量の追加
        Args:
            base_df: create_wide_feature_dataframeで作成した特徴量DF
            predictions_df: 'prex', 'prey' カラムを持つ推定座標DF
        """
        # 結合する全パーツを確実に 0 からの連番にする
        base_df = base_df.reset_index(drop=True)
        predictions_df = predictions_df.reset_index(drop=True)

        if len(base_df) != len(predictions_df):
            raise ValueError("Shape mismatch between base_df and predictions")

        ap_locations_dict = config.AP_location
        ap_names = list(config.ap_ids.keys())
        
        # 予測座標の整形
        pred_coords_df = predictions_df.iloc[:, :2].copy()
        pred_coords_df.columns = [0, 1] # Phase1関数用

        # 1. 距離・角度・壁距離計算 (Phase1モジュール利用)
        pred_dist_df = calculate_distance_to_aps(pred_coords_df, ap_locations_dict)
        pred_dist_df.columns = [f'pre_dist_{name}' for name in ap_names]

        pred_rad_df = calculate_azimuth_to_aps(pred_coords_df, ap_locations_dict)
        pred_rad_df.columns = [f'pred_azimuth_{name}' for name in ap_names]

        # ★ Configを渡して壁距離計算
        pred_wall_df = calculate_distance_to_walls(pred_coords_df, config)
        # --- 修正版：1 & 2. 距離・角度・差分・比率をAPごとに一括計算 ---
        dist_list = []
        azimuth_list = []
        diff_list = []
        ratio_list = []

        # 座標計算用の配列
        pred_np = pred_coords_df.values # (N, 2)

        for name in ap_names:
            # A. そのAPの座標を確実に取得
            ap_coord = np.array(config.AP_location[name]) # [x, y]
            
            # B. 予測地点との差分ベクトル (N, 2)
            # ブロードキャストを利用：全データから同じAP座標を一気に引く
            diff_vec = pred_np - ap_coord
            
            # C. 距離計算 (2026-08-19修正: 誤った再現係数0.25を0.5に訂正。
            #    詳細はEAUFM_AutoUpdate進捗ログ§13参照)
            predicted_dist = np.linalg.norm(diff_vec, axis=1) * 0.5
            dist_list.append(pd.Series(predicted_dist, name=f'pre_dist_{name}'))
            
            # D. 角度計算 (旧コードの arctan2(y_diff, x_diff) を再現)
            # diff_vec[:, 1] が y, diff_vec[:, 0] が x
            angles = np.arctan2(diff_vec[:, 1], diff_vec[:, 0]) * 180 / np.pi
            azimuth_list.append(pd.Series(angles, name=f'pred_azimuth_{name}'))
            
            # E. 実測値(measured)との比較
            col_rng = f"{name}_rng_rng"
            col_raw = f"{name}_rng_raw"
            col_meas = col_rng if col_rng in base_df.columns else col_raw
            
            measured = base_df[col_meas].values
            
            # 差分 (measured - predicted)
            diff_list.append(pd.Series(measured - predicted_dist, name=f'diff_dist_{name}'))
            
            # 比率 (measured / predicted)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = measured / predicted_dist
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e9, neginf=-1e9)
            ratio_list.append(pd.Series(ratio, name=f'ratio_dist_{name}'))

        # 各リストを横に結合
        pred_dist_df = pd.concat(dist_list, axis=1)
        pred_rad_df = pd.concat(azimuth_list, axis=1)
        diff_df = pd.concat(diff_list, axis=1)
        ratio_df = pd.concat(ratio_list, axis=1)

        # 3. Wall Difference / Ratio (ベクトル化して計算)
        # pred_wall_df: (N, num_walls)
        # measured_distances: (N, num_aps)
        
        # rng_raw と rng_rng のどちらを使うか明示的に判定する
        if f"{ap_names[0]}_rng_rng" in base_df.columns:
            measured_dist_df = base_df[[f"{name}_rng_rng" for name in ap_names]] # rng_rng を優先
        elif f"{ap_names[0]}_rng_raw" in base_df.columns:
            measured_dist_df = base_df[[f"{name}_rng_raw" for name in ap_names]] # または rng_raw
        else:
            # どちらの列も見つからない場合の安全弁
            raise ValueError("base_df に必要な rng_rng または rng_raw のカラムが存在しません。")
        
        wall_diff_dfs = []
        wall_ratio_dfs = []

        for ap_idx, ap_name in enumerate(ap_names):
            meas_vec = measured_dist_df.iloc[:, ap_idx].values
            
            # Wall Diff
            for w_idx in range(pred_wall_df.shape[1]):
                wall_vec = pred_wall_df.iloc[:, w_idx].values
                col_name = f'wall_diff_{ap_name}_Wall{w_idx}'
                wall_diff_dfs.append(pd.Series(wall_vec - meas_vec, name=col_name))
                
                # Wall Ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = wall_vec / meas_vec
                r[np.isinf(r)] = 0.0 # 元コード準拠
                r = np.nan_to_num(r, nan=0.0)
                
                col_name_r = f'wall_ratio_{ap_name}_Wall{w_idx}'
                wall_ratio_dfs.append(pd.Series(r, name=col_name_r))

        wall_diff_df = pd.concat(wall_diff_dfs, axis=1)
        wall_ratio_df = pd.concat(wall_ratio_dfs, axis=1)
        
        # 全結合
        final_df = pd.concat([ # 次元数: 66 = 2((X,Y)) + 24(観測特徴量) + 40(幾何特徴量)
            base_df.reset_index(drop=True),
            pred_dist_df,
            diff_df,
            ratio_df,
            pred_wall_df,
            wall_diff_df,
            wall_ratio_df,
            pred_rad_df
        ], axis=1)

        # FP (Fingerprint) として各RPごとの平均値を計算 (X, Yでグループ化)
        avg_df = final_df.groupby(['X', 'Y']).mean()
        averaged_data = avg_df.values

        return final_df, averaged_data

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def split_data_stratified(self, df: pd.DataFrame, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        位置(X, Y)ごとに層化サンプリングを行い、Train/Testに分割する。
        時系列性を考慮し、Shuffle=Falseで分割する。
        """
        grouped = df.groupby(['X', 'Y'])
        train_idx, test_idx = [], []
        
        for _, group in grouped:
            if len(group) < test_size + 1:
                raise ValueError(f"Insufficient samples for location ({group['X'].iloc[0]}, {group['Y'].iloc[0]}). Need at least {test_size + 1} samples.")
            tr, te = train_test_split(group.index, test_size=test_size, random_state=42, shuffle=False)
            train_idx.extend(tr)
            test_idx.extend(te)
            
        return df.loc[train_idx].sort_index(), df.loc[test_idx].sort_index()

    # ---------------------------------------------------------
    # EAU_FM phase1.py 関数を複製
    # ---------------------------------------------------------
def calculate_distance_to_walls(predictions: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """
    測位地点と壁の距離計算
    Config.env の値によって、柱(pillar)の考慮ロジックを切り替える
    """
    pred_np = predictions.values
    walls = config.wall_location # list[float]
    
    # 結果格納用
    pred_dist_list = []

    # Small Scale Env 特有の「柱」ロジック判定
    is_small_env = (config.env == 'small_scale_env')
    
    # Note: Phase1.pyのロジックでは、wallsのindex 0,1がY軸方向(Top/Bottom)、2,3がX軸方向(Left/Right)と想定されている
    
    for i in range(len(pred_np)):
        x, y = pred_np[i, 0], pred_np[i, 1]
        dist_row = []
        
        for j, wall_pos in enumerate(walls):
            d_val = 0.0
            
            # --- Small Scale (柱あり) ロジック ---
            if is_small_env and (-0.5 <= x < 0.8):
                # 柱エリア
                if j == 1: # 特定の壁(Bottom側?)の場合、柱(+5.7)との距離
                    # 注意: 5.7 というマジックナンバーはConfig管理推奨だが、
                    # 現状は互換性維持のためここに記述するか、Configに追加を検討してください。
                    d_val = abs(y + 5.7)
                elif j == 0:
                    d_val = abs(y - wall_pos)
                else:
                    d_val = abs(x - wall_pos)
            
            # --- 通常ロジック (Big Scale or Smallの柱エリア外) ---
            else:
                if j == 0 or j == 1: # Y軸方向
                    d_val = abs(y - wall_pos)
                else: # X軸方向
                    d_val = abs(x - wall_pos)
            
            # ======================= ⚠️ 後修正必須 ======================= #
            # ★再現ポイント: 旧コードの「* 0.5」に合わせて 0.5 を掛ける
            dist_row.append(d_val * 0.5)
            # dist_row.append(d_val)
            # ======================= ⚠️ 後修正必須 ======================= #
        
        pred_dist_list.append(dist_row)

    # カラム名
    cols = [f'wall_dist_{k}' for k in range(len(walls))]
    return pd.DataFrame(pred_dist_list, columns=cols)

def get_coordinates_from_filename(file_path: str) -> List[float]:
    """ファイル名から座標を取得 (例: -0.5_2.5.csv -> [-0.5, 2.5])"""
    base_name = Path(file_path).stem
    parts = base_name.split('_')
    try:
        return [float(p) for p in parts]
    except ValueError:
        # 座標形式でないファイル名の場合はNoneを返す等のエラーハンドリングが必要なら追加
        return []

def hex_to_amplitude_list(cir_series: pd.Series) -> List[List[float]]:
    """CIR(HEX文字列)のSeriesを振幅リストのリストに変換"""
    amplitudes = []
    for data in cir_series:
        hex_list = [data[i:i+4] for i in range(0, len(data), 4)]
        dec_list = hex_to_decimal(hex_list)
        amplitude = calculate_amplitude(dec_list)
        amplitudes.append(amplitude)
    return amplitudes

def extract_features_from_amplitude(amplitudes: List[List[float]]) -> pd.DataFrame:
    """CIR振幅から特徴量を抽出"""
    results = []
    threshold = 300  # ノイズレベルの閾値

    for amplitude in amplitudes:
        peaks, _ = find_peaks(amplitude, height=threshold)
        delays = np.array(peaks)

        first_path_index = peaks[0] if len(peaks) > 0 else np.nan
        first_path_amplitude = amplitude[first_path_index] if len(peaks) > 0 else np.nan
        second_path_index = peaks[1] if len(peaks) > 1 else np.nan
        second_path_amplitude = amplitude[second_path_index] if len(peaks) > 1 else np.nan

        # 新しい特徴量を計算
        avg_delay = np.mean(delays) if len(delays) > 0 else np.nan
        
        results.append([
            first_path_index, 
            first_path_amplitude,
            second_path_index, 
            second_path_amplitude,
            len(peaks),
            avg_delay
        ])

    columns = [
        'First Path Index', 'First Path Amplitude',
        'Second Path Index', 'Second Path Amplitude',
        'Number of Paths', 'Avg Delay'
    ]
    return pd.DataFrame(results, columns=columns)

def calculate_distance_to_aps(predictions: pd.DataFrame, ap_locations: Dict[str, List[float]]) -> pd.DataFrame:
    """
    測位地点(predictions)と各APの距離を計算
    ConfigのAP辞書のキー順序に基づいてカラムを生成する
    """
    pred_np = predictions.values # (N, 2)
    ap_keys = list(ap_locations.keys())
    
    # 全APの座標を (M, 2) の配列にする
    ap_coords = np.array([ap_locations[k] for k in ap_keys])
    
    # 放送(Broadcasting)を使って一括計算
    # predictions[:, None, :] -> (N, 1, 2)
    # ap_coords[None, :, :]   -> (1, M, 2)
    # diff -> (N, M, 2)
    diff = pred_np[:, None, :] - ap_coords[None, :, :]
    
    # ノルム計算 (axis=2) -> (N, M)
    dists = np.linalg.norm(diff, axis=2)

    # 2026-08-19修正: phase1.pyと同じ誤りを訂正(このモジュール内では未使用の
    # 複製関数だが、将来の混乱を防ぐため揃えておく)
    dists_repro = dists * 0.5
    
    # カラム名作成
    cols = [f'pre_dist_{i}' for i in range(len(ap_keys))]
    
    return pd.DataFrame(dists_repro, columns=cols)

def calculate_azimuth_to_aps(predictions: pd.DataFrame, ap_locations: Dict[str, List[float]]) -> pd.DataFrame:
    """
    測位地点とAPの角度を計算 (高速版 ver3ロジック採用)
    """
    pred_np = predictions.values
    ap_keys = list(ap_locations.keys())
    ap_coords = np.array([ap_locations[k] for k in ap_keys]) # (M, 2)
    
    # (N, 1, 2) - (1, M, 2) = (N, M, 2)
    diff = pred_np[:, None, :] - ap_coords[None, :, :]
    
    diff_x = diff[:, :, 0] # (N, M)
    diff_y = diff[:, :, 1] # (N, M)
    
    angles = np.arctan2(diff_y, diff_x) * 180 / np.pi
    
    # カラム名作成 (必要に応じて命名規則を調整)
    cols = [f'pre_angle_{i}' for i in range(len(ap_keys))]
    
    return pd.DataFrame(angles, columns=cols)

def hex_to_decimal(hex_list: List[str]) -> List[int]:
    """16進数リストを10進数に変換"""
    dec_list = []
    for hex_value in hex_list:
        dec_value = int(hex_value, 16)
        if dec_value >= 0x8000:
            dec_value -= 0x10000
        dec_list.append(dec_value)
    return dec_list

def calculate_amplitude(dec_list: List[int]) -> List[float]:
    """複素数の振幅を計算"""
    amplitude = []
    for i in range(0, len(dec_list), 2):
        real_part = dec_list[i]
        imag_part = dec_list[i + 1] if i + 1 < len(dec_list) else 0
        amp = np.sqrt(real_part**2 + imag_part**2)
        amplitude.append(amp)
    return amplitude