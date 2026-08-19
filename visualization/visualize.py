# 測位結果をプロットするための関数群

import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd


# 全データポイントの測位結果をプロット
def plot_localization_all_points(predictions,
                                true_locations,
                                error,
                                positioning_model,
                                used_features,
                                save_flg=False,
                                save_path=None,
                                Pattern_name=None,
                                add_save_file_name=None):
    # predictions と true_locations を npに変換
    if isinstance(predictions, pd.DataFrame):
        # DataFrameなら .values を使ってNumPy配列に変換
        predictions = predictions.values
    else:
        # DataFrameでなければ、NumPy配列（またはリストなど）として扱う
        predictions = np.array(predictions)
    if isinstance(true_locations, pd.DataFrame):
        true_locations = true_locations.values
    else:
        true_locations = np.array(true_locations)

    plt.figure(figsize=(10, 8))
    # 真の位置をプロット
    plt.scatter(true_locations[:, 0], true_locations[:, 1], color='blue', label='true locations', alpha=0.6)

    # 推定位置をプロット
    plt.scatter(predictions[:, 0], predictions[:, 1], color='red', label='predicted locations', alpha=0.6)

    # 各真の位置と推定位置を線で結ぶ（誤差を視覚化）
    for i in range(len(true_locations)):
        plt.plot([true_locations[i, 0], predictions[i, 0]],
                [true_locations[i, 1], predictions[i, 1]],
                color='lightgray', linestyle='--', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    title = f"Model: {positioning_model}_Features: {used_features}_all_points_localization_results (error: {error:.2f} m)"
    if add_save_file_name is not None:
        title += f"_{add_save_file_name}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_flg and save_path != None and Pattern_name != None:
        save_file_name = f"{Pattern_name}_uwb_positioning_results_all_points.pdf"
        if add_save_file_name is not None:
            save_file_name = f"{save_file_name[:-4]}_{add_save_file_name}.pdf"
        save_path = save_path / save_file_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"測位結果のプロット (全データポイント) を '{save_file_name}' として保存しました。")
    else:
        print("save_flgがFalse、またはsave_path/Pattern_nameが指定されていないため、プロットを保存しません。")
    plt.show()


# 各RPごとの測位結果の平均結果をプロット
def plot_localization_mean_per_RP(predictions,
                                true_locations,
                                error, positioning_model,
                                used_features,
                                save_flg=False,
                                save_path=None,
                                Pattern_name=None,
                                add_save_file_name=None):
    # predictions と true_locations を npに変換
    if isinstance(predictions, pd.DataFrame):
        # DataFrameなら .values を使ってNumPy配列に変換
        predictions = predictions.values
    else:
        # DataFrameでなければ、NumPy配列（またはリストなど）として扱う
        predictions = np.array(predictions)
    if isinstance(true_locations, pd.DataFrame):
        true_locations = true_locations.values
    else:
        true_locations = np.array(true_locations)
    results_df = pd.DataFrame(true_locations, columns=['X', 'Y'])
    results_df.rename(columns={'X': 'true_x', 'Y': 'true_y'}, inplace=True)
    results_df['pred_x'] = predictions[:, 0]
    results_df['pred_y'] = predictions[:, 1]

    # 'true_x', 'true_y' でグループ化し、予測の平均を計算
    average_predictions = results_df.groupby(['true_x', 'true_y']).agg(
        avg_pred_x=('pred_x', 'mean'),
        avg_pred_y=('pred_y', 'mean')
    ).reset_index()

    plt.figure(figsize=(10, 8))

    # ユニークな真の位置（平均化されたリファレンスポイント）をプロット
    plt.scatter(average_predictions['true_x'], average_predictions['true_y'],
                color='blue', marker='o', s=100, label='true locations', zorder=5)

    # 各真の位置に対応する平均推定位置をプロット
    plt.scatter(average_predictions['avg_pred_x'], average_predictions['avg_pred_y'],
                color='red', marker='x', s=100, label='average predicted locations', zorder=5)

    # 真の位置と平均推定位置を線で結ぶ
    for i in range(len(average_predictions)):
        plt.plot([average_predictions['true_x'].iloc[i], average_predictions['avg_pred_x'].iloc[i]],
                [average_predictions['true_y'].iloc[i], average_predictions['avg_pred_y'].iloc[i]],
                color='lightgray', linestyle='-', linewidth=1, alpha=0.7, zorder=1)

    # plt.title('各正解座標ごとの平均測位結果')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    title = f"Model: {positioning_model}_Features: {used_features}_average_localization_results_per_RP (error: {error:.2f} m)"
    if add_save_file_name is not None:
        title += f"_{add_save_file_name}"
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を等しくして、歪みをなくす
    plt.tight_layout()
    if save_flg and save_path != None and Pattern_name != None:
        save_file_name = f"{Pattern_name}_uwb_positioning_average_results.pdf"
        if add_save_file_name is not None:
            save_file_name = f"{save_file_name[:-4]}_{add_save_file_name}.pdf"
        save_path = save_path / save_file_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"測位結果のプロット (各RPごとの平均結果) を '{save_file_name}' として保存しました。")
    else:
        print("save_flgがFalse、またはsave_path/Pattern_nameが指定されていないため、プロットを保存しません。")
    plt.show()


# 各RPごとの測位誤差ヒートマップをプロット
def plot_localization_error_heatmap(df, value_col, vmin=None, vmax=None, save_flg=False, save_path=None, env=None):
    """
    指定されたデータフレームの特定カラムの値をヒートマップとしてプロット・保存する単機能関数。
    
    Parameters:
    ----------
    df : pandas.DataFrame
        'X', 'Y' カラムと、プロットしたい値のカラム(value_col)を含むデータフレーム。
    value_col : str
        ヒートマップの色として表示したいカラム名（例: 'error'）。
    vmin : float, optional
        ヒートマップのカラーバーの最小値。指定しない場合はデータの最小値が使用される。
    vmax : float, optional
        ヒートマップのカラーバーの最大値。指定しない場合はデータの最大値が使用される。
    """
    
    # データに必要なカラムがあるか確認
    if value_col not in df.columns:
        print(f"エラー: データフレームに '{value_col}' というカラムが見つかりません。")
        return

    plt.figure(figsize=(8, 6))
    
    try:
        # 1. データをヒートマップ用に整形 (Pivot)
        # 同じ座標(X, Y)に複数のデータがある場合は平均値(mean)をとる
        df_pivot = df.pivot_table(index='Y', columns='X', values=value_col, aggfunc='mean')
        
        # Y軸をグラフの見た目（上が大きい）に合わせて反転
        df_pivot.sort_index(ascending=False, inplace=True)
        
        # 2. ヒートマップ描画
        sns.heatmap(
            df_pivot,
            annot=True,      # 各セルに数値を表示
            fmt=".2f",       # 小数点以下2桁
            cmap="Reds",     # 赤色（誤差の表現に適している）
            vmin=vmin,       # カラーバーの最小値
            vmax=vmax,       # カラーバーの最大値
            cbar_kws={'label': value_col}
        )
        
        # タイトルと軸ラベルの設定
        plt.title(f"Heatmap of {value_col} per {env}")
        plt.xlabel("X Coordinate [m]")
        plt.ylabel("Y Coordinate [m]")
        plt.tight_layout()
        
        # 3. 保存
        if save_flg and save_path != None and env != None:
            save_file_name = f"{env}_positioning_error_heatmap.pdf"
            p = Path(save_path)
            p = p / save_file_name
            # ディレクトリがない場合は作成する
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300, bbox_inches='tight')
            print(f"測位結果のプロット (各RPごとの平均結果) を '{env}_positioning_error_heatmap.pdf' として保存しました。")
        else:
            print("save_flgがFalse、またはsave_path/Pattern_nameが指定されていないため、プロットを保存しません。")
        # 4. 表示
        plt.show()

    except Exception as e:
        print(f"ヒートマップ作成中にエラーが発生しました: {e}")


# 指定更新順による測位誤差推移をプロット
def plot_localization_error_over_updates(errors, env_config, method, save_flg=False, save_path=None):
    plt.figure(figsize=(10, 6))
    
    list_length = len(errors)

    # 条件分岐
    if list_length == 17:
        x_values = list(range(0, 17))
        check_index = [0, 1, 3, 5, 10]
        print("Mode: 17 elements (Short)")
    elif list_length == 65:
        x_values = list(range(0, 65))
        check_index = [0, 1, 3, 5, 10, 30, 50]
        print("Mode: 65 elements (Long)")
    else:
        raise ValueError(f"Unexpected list length: {list_length}")

    # check用出力
    for idx in check_index:
        print(f"Index {idx}: {errors[idx]}")

    # Plotting the error curves
    plt.plot(x_values, errors , marker='o', linestyle='-', color='blue', label=method)



    # Adding labels and title
    plt.xlabel('Number of updates', fontsize=20)
    plt.ylabel('Error (cm) ', fontsize=20)
    # plt.xlabel('更新数', fontsize=20)
    # plt.ylabel('平均測位誤差 (cm) ', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)

    # 目盛りのフォントサイズを大きく
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 保存
    if save_flg and save_path != None:
        save_file_name = f"{env_config['source_rp']}_{env_config['target_rp']}_localization_error_over_updates.pdf"
        p = Path(save_path)
        p = p / save_file_name
        # ディレクトリがない場合は作成する
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"測位誤差推移のプロットを '{env_config['source_rp']}_{env_config['target_rp']}_localization_error_over_updates.pdf' として保存しました。")
    plt.show()


# 指定順位での更新による測位誤差推移の平均と標準偏差をプロット
def plot_localization_error_over_updates_with_std(errors_mean, errors_std, env_config, method, save_flg=False, save_path=None):
    plt.figure(figsize=(10, 6))
    list_length = len(errors_mean)

    # 条件分岐
    if list_length == 17:
        x_values = list(range(0, 17))
        check_index = [0, 1, 3, 5, 10]
        print("Mode: 17 elements (Short)")
    elif list_length == 65:
        x_values = list(range(0, 65))
        check_index = [0, 1, 3, 5, 10, 30, 50]
        print("Mode: 65 elements (Long)")
    else:
        raise ValueError(f"Unexpected list length: {list_length}")

    # check用出力
    for idx in check_index:
        print(f"Index {idx}: mean={errors_mean[idx]}, std={errors_std[idx]}")

    # Plotting the error curves with standard deviation as shaded area
    plt.plot(x_values, errors_mean , marker='o', linestyle='-', color='blue', label=method)
    plt.fill_between(x_values,
                     np.array(errors_mean) - np.array(errors_std),
                     np.array(errors_mean) + np.array(errors_std),
                     color='blue', alpha=0.2)

    # Adding labels and title
    plt.xlabel('Number of updates', fontsize=20)
    plt.ylabel('Error (cm) ', fontsize=20)
    # plt.xlabel('更新数', fontsize=20)
    # plt.ylabel('平均測位誤差 (cm) ', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)

    # 目盛りのフォントサイズを大きく
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # 保存
    if save_flg and save_path != None:
        save_file_name = f"{env_config['source_rp']}_{env_config['target_rp']}_localization_error_over_updates_with_std.pdf"
        p = Path(save_path)
        p = p / save_file_name
        # ディレクトリがない場合は作成する
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"測位誤差推移のプロットを '{env_config['source_rp']}_{env_config['target_rp']}_localization_error_over_updates_with_std.pdf' として保存しました。")