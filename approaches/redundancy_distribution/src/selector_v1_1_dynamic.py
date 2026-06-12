"""
脑启发核心集选择算法 v1.1 - 动态保底
Brain-Inspired Coreset Selection v1.1 (Dynamic Quota)

思路：冗余分布 -> 防止分布冗余
机制：K-Means 聚类 + 簇大小动态配额 + 全局高分补齐
改进：大簇多保底（max(1, min(3, size//50+1))），小簇少保底，避免名额浪费

历史：2026-06-01 从 v1.0 改进，Test MSE 0.003961（↓2.6% vs v1.0）
"""
import numpy as np
from sklearn.cluster import KMeans
from config import *


class BrainInspiredCoresetSelector:
    """
    脑启发核心集选择器 v1.1
    基于视觉特征分布多样性，自动筛选高价值数据子集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO):
        self.target_ratio = target_ratio
    
    def compute_diversity_scores(self, features: np.ndarray, n_clusters: int = None):
        """
        计算分布多样性得分
        原理：视觉特征聚类后，稀疏簇和远离中心的样本 -> 信息效用高 -> 得分高
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        if n_clusters is None:
            n_clusters = max(2, target_size // 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        dists = np.linalg.norm(features - centers[labels], axis=1)
        
        sizes = np.bincount(labels, minlength=n_clusters)
        size_penalty = 1.0 / np.sqrt(sizes[labels] + 1.0)
        
        scores = dists * size_penalty
        
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores, labels
    
    def _get_cluster_quota(self, cluster_size: int):
        """动态保底：大簇多保底，小簇少保底"""
        return max(1, min(3, cluster_size // 50 + 1))
    
    def select(self, features: np.ndarray, episode_indices: np.ndarray):
        """
        执行核心集选择（v1.1 动态保底）
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        scores, labels = self.compute_diversity_scores(features)
        
        # v1.1：动态保底，按簇大小分配名额
        selected = set()
        for c in np.unique(labels):
            idx_in_c = np.where(labels == c)[0]
            quota = self._get_cluster_quota(len(idx_in_c))
            sorted_idx = idx_in_c[np.argsort(scores[idx_in_c])[-quota:]]
            for idx in sorted_idx:
                selected.add(int(idx))
        
        if len(selected) > target_size:
            selected = set(np.argsort(scores)[-target_size:])
        else:
            remaining = target_size - len(selected)
            if remaining > 0:
                unselected = [i for i in range(n) if i not in selected]
                unselected_scores = scores[unselected]
                top_local_idx = np.argsort(unselected_scores)[-remaining:]
                for idx in top_local_idx:
                    selected.add(unselected[idx])
        
        selected = np.array(sorted(list(selected)), dtype=np.int64)
        
        print(f"[Coreset v1.1] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Coreset v1.1] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
