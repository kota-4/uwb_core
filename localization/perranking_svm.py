# 既存perrankingコードによる指定ランキング順にFPを更新し, SVMで測位
# 測位手法: SVM回帰
# 使いまわしできるように関数化

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils_lib.common import Localize, Evaluate_localization_error, Experiment_config
from create_dataset import main as create_dataset_main # ← 要修正: uwb_core版に

def wide_format_perranking_preprocessing(wide_dict_data: dict, k_split: int = 3):
    """
    wide_format形式でperrankingによるFP更新を行うためのデータ前処理関数
    Args:
        wide_dict_data (dict): ワイドフォーマットデータ辞書 (source, target)
        k_split (int): 層化サンプリングの分割数 (デフォルト: 3)
    Returns:

    """

    # ======= データ前処理 =======
    # データ分割
    split_data_dict = create_dataset_main.split_data_k(wide_dict_data, k_split)

    # --- これから使用するデータプールを準備 ---
    # 更新元となる「source環境の訓練データ」全体
    source_train_full_df = split_data_dict['source_train']
    # 更新先となる「target環境の訓練データ」全体
    target_train_full_df = split_data_dict['target_train']
    # 評価に使う「テストデータ」 (これはループ全体で不変)
    test_df = split_data_dict['target_test']
    # データを座標ごとに高速にアクセスできるよう、辞書化しておく
    source_grouped = {tuple(name): group for name, group in source_train_full_df.groupby(['X', 'Y'])}
    target_grouped = {tuple(name): group for name, group in target_train_full_df.groupby(['X', 'Y'])}

    return source_train_full_df, test_df, source_grouped, target_grouped

def existing_perranking_wideformat_SVM(
        ranking: list,
        wide_dict_data: dict,
        config: Experiment_config.ExperimentConfig,
        k_split: float = 0.3,
        svm_parameters: dict = {'C': 1000, 'gamma': 0.01}
        ):
    """
    既存のperranking手法に基づき, 指定されたランキング順にFPを更新しながらSVMで測位を行う関数
    Args:
        ranking (list): 更新するRPのランキングリスト
        wide_dict_data (dict): ワイドフォーマットのデータセットを含む辞書
        config (ExperimentConfig): 実験設定オブジェクト
        k_split (float): クロスバリデーションの分割比率（デフォルトは0.3）
        svm_parameters (dict): SVMのハイパーパラメータ
    Returns:
        aveacc_list (list): 各ステップでの測位精度リスト
    """
    # データ前処理
    source_train_full_df, test_df, source_grouped, target_grouped = wide_format_perranking_preprocessing(wide_dict_data, k_split)
    test_data = test_df.drop(columns=['X', 'Y'])
    test_labels = test_df[['X', 'Y']]

    # ======================================================================
    # FP更新シミュレーションのメインループ ---
    # ======================================================================
    # 複数のランキングを全て更新するように
    all_rank_list = [
        ranking
    ]
    all_method_name = [
        config.Pattern_name
    ]

    for rank_list, method in zip(all_rank_list, all_method_name):
        print("\n=========================================")
        print("新しいランキングでのFP更新シミュレーションを開始します")
        print("手法:", method)
        print("ランキング:", rank_list)
        print("=========================================")

        # ランキングに基づいて、更新するRPのインデックスを取得
        # ここで、rank_listは1始まりなので、0始まりに変換
        rank_list = [r - 1 for r in rank_list]
        # 昇順にソートしたときの元のインデックスを取得

        sorted_with_index = sorted(enumerate(rank_list), key=lambda x: x[1])
        original_indices = [x[0] for x in sorted_with_index]
        
        update_location_list = []
        aveacc_list = []

        # --- ステップ0: 更新前の初期精度を測定 ---
        print("\n--- ステップ0: 更新前の精度を測定 ---")
        initial_train_data = source_train_full_df.drop(columns=['X', 'Y'])
        initial_train_labels = source_train_full_df[['X', 'Y']]
        
        # 正規化
        scaler = MinMaxScaler()
        initial_train_data_scaled = pd.DataFrame(scaler.fit_transform(initial_train_data), columns=initial_train_data.columns)
        test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
        
        # SVMで評価
        predictions_df = Localize.SVR_localization(
            initial_train_data_scaled, initial_train_labels,
            test_data_scaled,
            **svm_parameters)
        initial_acc = Evaluate_localization_error.calculate_Localization_Accuracy_50cm(
            predictions_df, test_labels
        )
        print(f"初期測位精度 (更新前): {initial_acc} m")
        aveacc_list.append(initial_acc)

        # --- メインループ ---
        for i in range(len(original_indices)):
            # 更新対象の座標を取得
            update_rp_index = original_indices[i]
            update_location = config.RP[update_rp_index]
            update_location_list.append(update_location)
            
            print(f"\n--- ステップ{i+1}: 地点 {update_location} を更新 ---")

            # --- a. 現在の訓練データセットをメモリ上で構築 ---
            current_train_df_list = []
            for rp_coords in config.RP:
                if rp_coords in update_location_list:
                    # 更新対象の座標は、targetのデータプールから取得
                    current_train_df_list.append(target_grouped[tuple(rp_coords)])
                else:
                    # それ以外の座標は、sourceのデータプールから取得
                    current_train_df_list.append(source_grouped[tuple(rp_coords)])
            
            # DataFrameのリストを結合して、このステップの訓練DFを完成
            current_train_df = pd.concat(current_train_df_list).sort_index()

            # --- b. データを分割し、正規化 ---
            current_train_data = current_train_df.drop(columns=['X', 'Y'])
            current_train_labels = current_train_df[['X', 'Y']]

            scaler = MinMaxScaler()
            current_train_data_scaled = pd.DataFrame(scaler.fit_transform(current_train_data), columns=current_train_data.columns)
            test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
            
            # --- c. SVMで評価 ---
            predictions_df = Localize.SVR_localization(
                current_train_data_scaled, current_train_labels,
                test_data_scaled,
                **svm_parameters)
            try_acc = Evaluate_localization_error.calculate_Localization_Accuracy_50cm(
                predictions_df, test_labels
            )
            print(f"更新後の測位精度: {try_acc} m")
            aveacc_list.append(try_acc)

        print("\n--- 全ての更新が完了しました ---")
        print(f"{method}_error: {aveacc_list}")
    return aveacc_list