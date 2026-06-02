"""
密度过滤最远点采样 (Density-Filtered Farthest Point Sampling)

思路：标准 FPS 容易选到离群噪声点（虽然"远"但教学价值低）。
后处理阶段剔除局部密度极低的孤立点，用全局中密度较高的帧替换。

机制：
  1. 标准 FPS 选出 target_size 帧
  2. 计算选中帧的 k-NN 平均距离（局部密度指标）
  3. 找出最低密度的 filter_ratio 比例（最孤立点）
  4. 从全局未选帧中，找密度最高的帧替换
"""
import numpy as np
from config import *


class DensityFilteredFPSSelector:
    """
    密度过滤最远点采样选择器
    标准 FPS + 后处理剔除孤立噪声点
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 seed: int = 42,
                 k: int = 5,
                 filter_ratio: float = 0.15):
        """
        Args:
            target_ratio: 目标筛选比例
            seed: 随机种子
            k: k-近邻数，用于计算局部密度
            filter_ratio: 替换比例（最低密度的多少比例被替换）
        """
        self.target_ratio = target_ratio
        self.seed = seed
        self.k = k
        self.filter_ratio = filter_ratio
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """执行密度过滤 FPS"""
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        # Step 1: 标准 FPS 选出初始核心集
        from approaches.farthest_point_sampling.src.selector_fps import FarthestPointSamplingSelector
        fps = FarthestPointSamplingSelector(target_ratio=self.target_ratio, seed=self.seed)
        selected = fps.select(features, actions, episode_indices).tolist()
        
        # Step 2: 计算选中帧的局部密度（k-NN 平均距离）
        from sklearn.neighbors import NearestNeighbors
        k = min(self.k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        distances, _ = nbrs.kneighbors(features[selected])
        avg_dists = distances[:, 1:].mean(axis=1)  # 距离越大 = 密度越低
        
        # Step 3: 找出最低密度的 filter_ratio 比例
        n_replace = max(1, int(len(selected) * self.filter_ratio))
        replace_idx = np.argsort(avg_dists)[-n_replace:]  # 距离最大的 = 最稀疏
        
        print(f"[Density-Filtered FPS] Replacing {n_replace} sparsest frames "
              f"(density threshold: {avg_dists[replace_idx[0]]:.4f})")
        
        # Step 4: 从全局未选帧中，找密度最高的帧替换
        unselected = [i for i in range(n) if i not in selected]
        if len(unselected) > 0:
            u_distances, _ = nbrs.kneighbors(features[unselected])
            u_avg_dists = u_distances[:, 1:].mean(axis=1)
            # 选未选帧中密度最高的（距离最小的）
            best_unselected = np.argsort(u_avg_dists)[:n_replace]
            for i, idx in enumerate(replace_idx):
                if i < len(best_unselected):
                    old = selected[idx]
                    new = unselected[best_unselected[i]]
                    selected[idx] = new
                    print(f"  Replace frame {old} (density={avg_dists[idx]:.4f}) "
                          f"-> frame {new} (density={u_avg_dists[best_unselected[i]]:.4f})")
        
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Density-Filtered FPS] Final: {len(selected)} frames")
        print(f"[Density-Filtered FPS] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
