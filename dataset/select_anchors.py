# 指定したアンカーidsのみのdfを返す
# 例): アンカー数5個＋タグ1個の場合, アンカー数を3個に減らしたdfを返す

import pandas as pd

def filter_anchors_in_df(df: pd.DataFrame, anchor_ids: dict, tag_id: int) -> pd.DataFrame:
    """
    指定したアンカーIDsのみを含むデータフレームを返す
    @param df: 元のデータフレーム
    @param anchor_ids: フィルタリングに使用するアンカーIDの辞書
    @return: フィルタリングされたデータフレーム
    """
    if not anchor_ids:
        print("Warning: No anchor IDs provided for filtering.")
        return df

    # アンカーIDの値リストを取得
    anchor_id_values = set(anchor_ids.values())
    if tag_id is not None:
        anchor_id_values.add(tag_id)  # タグIDも含める
    # daddr, taddr列でフィルタリング
    filtered_df = df[
        df['daddr'].isin(anchor_id_values) & df['taddr'].isin(anchor_id_values)
    ]

    return filtered_df