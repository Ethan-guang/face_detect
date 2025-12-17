import numpy as np

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """对特征向量进行 L2 归一化"""
    return x / (np.linalg.norm(x) + 1e-12)

def round_list(xs, ndigits=4):
    """保留列表小数位"""
    return [round(float(v), ndigits) for v in xs]