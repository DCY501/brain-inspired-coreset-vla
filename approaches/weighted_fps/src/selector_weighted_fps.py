"""
加权最远点采样 (Weighted Farthest Point Sampling)

思路：在标准 FPS 的基础上，给每帧引入"质量权重"。
得分 = 到已选集合的最小距离 x 质量权重
既保证空间覆盖，又优先选高质量帧。

权重策略：
  - feature_norm: 视觉特征 L2 范数（激活强度）
  - action_delta: 动作变化率（时序突变强度）
  - combined: feature_norm * action_delta（组合权重）

与标准 FPS 的关系：
  当权重 uniform 时，退化为标准 FPS。
"""
import numpy as np
from config import *


class WeightedFPSSelector:
    """
    加权最远点采样选择器
    得分 = min_dist(到已选集合) x weight(质量权重)
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 weight_strategy: str = 'feature_norm',
                 seed: int = 42):
        self.target_ratio = target_ratio
        self.weight_strategy = weight_strategy
        self.seed = seed
    
    def compute_weights(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """计算每帧的质量权重 [0, 1]"""
        n = len(features)
        
        if self.weight_strategy == 'uniform':
            weights = np.ones(n)
        
        elif self.weight_strategy == 'feature_norm':
            # 视觉特征 L2 范数 = 激活强度
            weights = np.linalg.norm(features, axis=1)
        
        elif self.weight_strategy == 'action_delta':
            # 动作变化率（跨 episode 边界重置）
            weights = np.zeros(n)
            for ep in np.unique(episode_indices):
                mask = episode_indices == ep
                idx = np.where(mask)[0]
                if len(idx) > 1:
                    ep_actions = actions[idx]
                    d = np.zeros(len(idx))
                    d[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
                    weights[idx] = d
        
        elif self.weight_strategy == 'combined':
            w1 = np.linalg.norm(features, axis=1)
            w2 = np.zeros(n)
            for ep in np.unique(episode_indices):
                mask = episode_indices == ep
                idx = np.where(mask)[0]
                if len(idx) > 1:
                    ep_actions = actions[idx]
                    d = np.zeros(len(idx))
                    d[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
                    w2[idx] = d
            w1 = (w1 - w1.min()) / (w1.max() - w1.min() + 1e-8)
            w2 = (w2 - w2.min()) / (w2.max() - w2.min() + 1e-8)
            weights = w1 * w2
        
        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")
        
        # 归一化到 (0, 1]
        weights = weights + 1e-8
        weights = weights / weights.max()
        return weights
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """执行加权最远点采样"""
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        rng = np.random.RandomState(self.seed)
        
        # 计算质量权重
        weights = self.compute_weights(features, actions, episode_indices)
        
        print(f"[Weighted FPS] Strategy={self.weight_strategy}")
        print(f"[Weighted FPS] Weight stats: max={weights.max():.4f}, "
              f"min={weights.min():.4f}, mean={weights.mean():.4f}, "
              f"median={np.median(weights):.4f}")
        
        # 初始帧：选权重最高的帧（而不是随机）
        initial_idx = int(np.argmax(weights))
        selected = [initial_idx]
        
        # 维护每个未选帧到已选集合的最小距离
        min_dists = np.full(n, np.inf)
        min_dists[initial_idx] = -1.0
        
        # 得分 = min_dist x weight
        scores = min_dists * weights
        scores[initial_idx] = -1.0
        
        print(f"[Weighted FPS] Selecting {target_size} frames...")
        
        for step in range(target_size - 1):
            last_idx = selected[-1]
            
            # 计算所有帧到最新加入帧的欧氏距离
            dists = np.linalg.norm(features - features[last_idx], axis=1)
            
            # 更新最小距离
            update_mask = dists < min_dists
            min_dists[update_mask] = dists[update_mask]
            
            # 已选的排除
            min_dists[selected] = -1.0
            
            # 得分 = min_dist x weight，选最高分
            scores = min_dists * weights
            scores[selected] = -1.0
            
            next_idx = int(np.argmax(scores))
            selected.append(next_idx)
            
            if (step + 2) % 200 == 0 or (step + 2) == target_size:
                print(f"  [Weighted FPS] Progress: {step + 2}/{target_size} frames selected")
        
        selected = np.array(selected, dtype=np.int64)
        
        print(f"[Weighted FPS] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Weighted FPS] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
