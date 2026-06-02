"""
脑启发核心集选择算法 v1.0 - 固定保底
Brain-Inspired Coreset Selection v1.0 (Fixed Quota)

思路：冗余分布 -> 防止分布冗余
机制：K-Means 聚类 + 每簇固定保底 1 帧 + 全局高分补齐

历史：2026-06-01 首次实现，Test MSE 0.004066
"""
import numpy as np
from sklearn.cluster import KMeans
from config import *


class BrainInspiredCoresetSelector:
    """
    脑启发核心集选择器 v1.0
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
    
    def select(self, features: np.ndarray, episode_indices: np.ndarray):
        """
        执行核心集选择（v1.0 固定保底：每簇 1 帧）
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        scores, labels = self.compute_diversity_scores(features)
        
        # v1.0：每簇固定保底 1 帧（得分最高的那个）
        selected = set()
        for c in np.unique(labels):
            idx_in_c = np.where(labels == c)[0]
            best_idx = idx_in_c[np.argmax(scores[idx_in_c])]
            selected.add(int(best_idx))
        
        remaining = target_size - len(selected)
        if remaining > 0:
            unselected = [i for i in range(n) if i not in selected]
            unselected_scores = scores[unselected]
            top_local_idx = np.argsort(unselected_scores)[-remaining:]
            for idx in top_local_idx:
                selected.add(unselected[idx])
        elif remaining < 0:
            selected = set(np.argsort(scores)[-target_size:])
        
        selected = np.array(sorted(list(selected)), dtype=np.int64)
        
        print(f"[Coreset v1.0] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Coreset v1.0] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
