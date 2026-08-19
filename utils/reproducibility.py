# uwb_core/utils/reproducibility.py
import random
import os
import numpy as np
import tensorflow as tf
# import torch # もしPyTorchを使う予定があれば追加

def set_seed(seed: int = 42):
    """
    実験の再現性を確保するために、乱数シードを固定する。
    
    Args:
        seed (int): 固定するシード値
    """
    # Python標準のrandom
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # OS環境変数 (ハッシュ化のランダム性抑制)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TensorFlowを使っている場合 (Autoencoder用)
    tf.random.set_seed(seed)
    
    # # PyTorchを使う場合 (将来的な拡張)
    # if 'torch' in globals():
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False