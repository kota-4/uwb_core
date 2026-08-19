import sys
from pathlib import Path
from typing import Tuple, Dict, List

class DatasetSelector:
    """
    対話形式でデータセットのディレクトリパスを選択するためだけのクラス。
    ConfigやAnchorManagerへの依存は持たない。
    """
    
    def __init__(self, base_dir_name: str = "UWB_data"):
        # プロジェクトルートからデータフォルダを探す
        self.root_path = self._find_root_path(base_dir_name)

    def _find_root_path(self, target_folder_name: str) -> Path:
        """データフォルダのパスを解決する"""
        current_path = Path(__file__).resolve()
        # 親ディレクトリを遡って探す
        for parent in [current_path] + list(current_path.parents):
            target = parent / target_folder_name
            if target.is_dir():
                return target
        
        # カレントディレクトリ直下も探す
        possible = Path.cwd() / target_folder_name
        if possible.is_dir():
            return possible
            
        raise FileNotFoundError(f"'{target_folder_name}' directory not found.")

    def _get_user_choice(self, prompt: str, options: List[str]) -> str:
        """ユーザー入力を受け付ける汎用メソッド"""
        print(f"--- {prompt} ---")
        for i, option in enumerate(options, 1):
            print(f"{i}: {option}")
        
        while True:
            try:
                choice_num = input(f"番号を選択してください (1-{len(options)}): ")
                choice_idx = int(choice_num) - 1
                if 0 <= choice_idx < len(options):
                    return options[choice_idx]
                else:
                    print("エラー: 選択肢の範囲外です。")
            except ValueError:
                print("エラー: 数値を入力してください。")

    def _get_subdirectories(self, path: Path) -> List[str]:
        """指定パス直下のディレクトリ名を取得 (隠しファイル除外)"""
        if not path.is_dir():
            return []
        try:
            return sorted([
                p.name for p in path.iterdir() 
                if p.is_dir() and not p.name.startswith('.') and p.name != '__pycache__'
            ])
        except OSError as e:
            print(f"エラー: {e}")
            return []

    def select_dataset(self) -> Tuple[Path, Path, Dict[str, str]]:
        """
        対話形式でソース・ターゲットのディレクトリを選択する。
        
        Returns:
            source_dir (Path): 選択されたソースデータのディレクトリパス
            target_dir (Path): 選択されたターゲットデータのディレクトリパス
            meta_info (Dict): 選択された環境情報 (who, env, source_rp, target_rp等)
        """
        # 1. Who (誰のデータか)
        who_opts = [n.replace('_data', '') for n in self._get_subdirectories(self.root_path) if n.endswith('_data')]
        if not who_opts: raise FileNotFoundError("No user data found in UWB_data.")
        who = self._get_user_choice("データ所有者を選択", who_opts)
        
        # 2. Env (規模)
        path_who = self.root_path / f"{who}_data"
        env_opts = self._get_subdirectories(path_who)
        env = self._get_user_choice("環境規模を選択", env_opts)
        
        # 3. Source Condition (環境条件)
        path_env = path_who / env
        cond_opts = self._get_subdirectories(path_env)
        source_cond = self._get_user_choice("ソース環境(条件)を選択", cond_opts)
        
        # 4. Source RP (座標パターン)
        path_source_root = path_env / source_cond
        rp_opts_source = self._get_subdirectories(path_source_root)
        source_rp = self._get_user_choice("ソースのRP状況を選択", rp_opts_source)
        
        # 5. Target Condition
        # 同じ env 内の条件から選ぶ
        target_cond = self._get_user_choice("ターゲット環境(条件)を選択", cond_opts)
        
        # 6. Target RP
        path_target_root = path_env / target_cond
        rp_opts_target = self._get_subdirectories(path_target_root)
        target_rp = self._get_user_choice("ターゲットのRP状況を選択", rp_opts_target)

        # パス確定
        source_dir = path_source_root / source_rp
        target_dir = path_target_root / target_rp
        
        # メタ情報作成 (Configの初期化に必要な文字列情報)
        meta_info = {
            "who": who,
            "env": env,
            "source_condition": source_cond, # 必要なら使う
            "target_condition": target_cond, # 必要なら使う
            "source_rp": source_rp,
            "target_rp": target_rp
        }

        print("\n--- 選択完了 ---")
        print(f"Source: {source_dir}")
        print(f"Target: {target_dir}")
        
        return source_dir, target_dir, meta_info
