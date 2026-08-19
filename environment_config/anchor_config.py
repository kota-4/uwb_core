# Anchorの設定を管理するモジュール
import pandas as pd

class AnchorManager:
    def __init__(self, tag_id=None):
        """
        初期化メソッド
        @param tag_id: タグIDが既知の場合はここで指定可能にする（指定なければNone）
        """
        self.anchor_ids = {}
        self.tag_id = tag_id
        # anchor配置情報
        self.small_env_anchor_locations = {
            'AP5307': [0, 6], 'AP4250': [1.5, 6], 'AP910': [3, 6], 
            'AP37051': [6, 3], 'AP23196': [6, 1.5], 'AP7091': [6, 0], 
            'AP4524': [0, -3], 'AP17057': [1.5, -3], 'AP1805': [3, -3], 
            'AP36794': [-3, 0], 'AP37045': [-1.5, 1.5], 'AP37248': [-3, 3]
        }
        self.large_env_anchor_locations = {
            'AP5307': [0, 15], 'AP4524': [0, -9], 'AP1805': [8, -9]
        }

    def generate_anchor_ids_from_data(self, df):
        """
        データフレームからアンカーIDを抽出し、自身の状態を更新する
        @return: 抽出されたアンカー辞書
        """
        unique_ids = set()

        # tag_id が未指定 かつ saddr列がある場合のみ抽出
        if self.tag_id is None and 'saddr' in df.columns:
            saddr_values = df['saddr'].dropna().unique()
            if len(saddr_values) > 0:
                self.tag_id = saddr_values[0] # 配列の先頭を取得

        # 探索するカラム名
        search_cols = ['daddr', 'taddr']
        # カラムが存在すればユニークなIDを取得
        for col in search_cols:
            if col in df.columns:
                # dropna() で欠損を除き、unique() でIDリストを取得
                ids = df[col].dropna().unique()
                unique_ids.update(ids)

        # 型を合わせた除外処理
        if self.tag_id is not None:
            # self.tag_id が numpy型かもしれないので、一旦セット内の存在確認をする
            if self.tag_id in unique_ids:
                unique_ids.remove(self.tag_id)
        
        # 辞書生成
        ap_ids_dict = {}
        for uid in sorted(unique_ids):
            try:
                val = int(uid)
                # 除外漏れ防止: 万が一抽出したIDがタグIDと同じならスキップ
                if self.tag_id is not None and val == int(self.tag_id):
                    continue
                
                ap_ids_dict[f"AP{val}"] = val
            except (ValueError, TypeError):
                print(f"Warning: Non-integer Anchor ID found: {uid}")
        
        # 自分の状態(self)を更新
        self.anchor_ids = ap_ids_dict
        
        # Docstringに合わせて return する
        return self.anchor_ids
    
    def select_anchor_ids(self, anchor_id_list):
        """
        指定されたアンカーIDリストに基づいて、保持しているアンカーIDをフィルタリングする
        @param anchor_id_list: フィルタリングに使用するアンカーIDのリスト
        @return: フィルタリング後のアンカー辞書
        """
        filtered_anchors = {
            key: value
            for key, value in self.anchor_ids.items()
            if value in anchor_id_list
        }
        self.anchor_ids = filtered_anchors
        return self.anchor_ids


    # anchorに基づいてanchor_locationsを返す
    def get_anchor_locations(self, env):
        """
        アンカーIDに基づいて、アンカーの位置情報を返す
        @param anchor_locations_dict: アンカーIDをキー、位置情報を値とする辞書
        @return: フィルタリング後のアンカー位置情報辞書
        """
        if env == 'small_scale_env':
            target_ids = self.anchor_ids.keys()
            filtered_locations = {
                key: self.small_env_anchor_locations[key]
                for key in target_ids 
                if key in self.small_env_anchor_locations
                }
        elif env == 'big_scale_env':
            target_ids = self.anchor_ids.keys()
            filtered_locations = {
                key: self.large_env_anchor_locations[key]
                for key in target_ids 
                if key in self.large_env_anchor_locations
                }
        else:
            print(f"Error: Unknown environment '{env}' for anchor locations.")
            return {}
        return filtered_locations
    
    def get_anchor_ids(self):
        """ 現在保持しているアンカーリストを返す """
        return self.anchor_ids

    def __repr__(self):
        """ print(クラスインスタンス) したときの表示を見やすくするおまじない """
        return f"<AnchorManager tag_id={self.tag_id}, anchors={list(self.anchor_ids.keys())}>"