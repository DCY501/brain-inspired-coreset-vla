"""
最远点采样 (Farthest Point Sampling, FPS)

思路：不聚类，直接贪心选"离已选集合最远的点"。每次迭代选特征空间中
距离当前核心集最远的帧，直到选够 target_size。

理论保证：FPS 是 NP-hard 覆盖问题的贪心近似算法，能保证最大最小距离
下界为最优解的 1/2。

与冗余分布思路(v1.1)的正交性：
  - v1.1: K-Means 间接保证覆盖（稀疏簇优先）
  - FPS: 直接优化覆盖（最大最小距离），无需预设簇数
"""
import numpy as np
from config import *


class FarthestPointSamplingSelector:
    """
    最远点采样选择器
    基于贪心最大最小距离策略筛选高价值数据子集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO, seed: int = 42):
        self.target_ratio = target_ratio
        self.seed = seed
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行最远点采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量（本思路不使用，保留接口一致性）
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        rng = np.random.RandomState(self.seed)
        
        # Step 1: 随机选初始帧
        selected = [int(rng.randint(n))]
        
        # Step 2: 维护每个未选帧到已选集合的最小距离
        # min_dists[i] = min(distance(i, selected_j)) for all selected_j
        min_dists = np.full(n, np.inf)
        min_dists[selected[0]] = -1.0  # 标记为已选
        
        print(f"[FPS] Selecting {target_size} frames from {n} via farthest point sampling...")
        
        for step in range(target_size - 1):
            last_idx = selected[-1]
            
            # 计算所有帧到最新加入帧的欧氏距离
            dists = np.linalg.norm(features - features[last_idx], axis=1)
            
            # 更新最小距离：对每个未选帧，取到已选集合的最近距离
            update_mask = dists < min_dists
            min_dists[update_mask] = dists[update_mask]
            
            # 选最小距离最大的帧（最远点）
            min_dists[selected] = -1.0  # 已选的排除
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists[next_idx] = -1.0
            
            if (step + 2) % 200 == 0 or (step + 2) == target_size:
                print(f"  [FPS] Progress: {step + 2}/{target_size} frames selected")
        
        selected = np.array(selected, dtype=np.int64)
        
        print(f"[FPS] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[FPS] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
