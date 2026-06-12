"""
快速最远点采样 (Fast FPS)
用预计算范数 + 点积优化 L2 距离计算
"""
import numpy as np
from config import *


class FastFPSSelector:
    """优化的 FPS，预计算 ||x||^2 减少重复计算"""
    def __init__(self, target_ratio: float = CORESET_RATIO, seed: int = 42):
        self.target_ratio = target_ratio
        self.seed = seed
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        rng = np.random.RandomState(self.seed)
        selected = [int(rng.randint(n))]
        
        # 预计算 ||x||^2
        feat_sq = np.sum(features ** 2, axis=1)
        min_dists = np.full(n, np.inf)
        min_dists[selected[0]] = -1.0
        
        for step in range(target_size - 1):
            last_idx = selected[-1]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
            dists_sq = feat_sq + feat_sq[last_idx] - 2.0 * features.dot(features[last_idx])
            dists_sq = np.maximum(dists_sq, 0.0)
            dists = np.sqrt(dists_sq)
            
            update_mask = dists < min_dists
            min_dists[update_mask] = dists[update_mask]
            
            min_dists[selected] = -1.0
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists[next_idx] = -1.0
            
            if (step + 2) % 500 == 0 or (step + 2) == target_size:
                print(f"  [Fast FPS] Progress: {step + 2}/{target_size}")
        
        return np.array(selected, dtype=np.int64)
