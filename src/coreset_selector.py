"""
脑启发核心集选择算法 (Brain-Inspired Coreset Selection)

核心机制：分布均衡 -> 防止分布冗余
某些简单动作占比过高会导致模型过拟合。
对应到数据：视觉特征 K-Means 聚类 + 稀疏簇偏好 -> 保证特征空间覆盖均衡。
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from config import *


class BrainInspiredCoresetSelector:
    """
    脑启发核心集选择器
    基于视觉特征分布多样性，自动筛选高价值数据子集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO):
        """
        Args:
            target_ratio: 目标筛选比例（如 0.1 = 10%）
        """
        self.target_ratio = target_ratio
    
    def compute_diversity_scores(self, features: np.ndarray, n_clusters: int = None):
        """
        计算分布多样性得分
        原理：视觉特征聚类后，稀疏簇和远离中心的样本 -> 信息效用高 -> 得分高
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            n_clusters: K-Means 聚类数，默认按目标大小自适应
        Returns:
            scores: np.ndarray, shape=(N,), 值域 [0, 1]
            labels: np.ndarray, shape=(N,), 聚类标签
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
        执行核心集选择（动态保底版本）
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        # PCA 降维（仅用于聚类，不改变原始特征用于训练）
        if PCA_N_COMPONENTS and PCA_N_COMPONENTS < features.shape[1]:
            pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
            features_pca = pca.fit_transform(features)
            print(f"[PCA] Reduced {features.shape[1]}d -> {features_pca.shape[1]}d "
                  f"(explained variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)")
        else:
            features_pca = features
        
        scores, labels = self.compute_diversity_scores(features_pca)
        
        # 动态保底：按簇大小分配保底名额
        selected = set()
        for c in np.unique(labels):
            idx_in_c = np.where(labels == c)[0]
            quota = self._get_cluster_quota(len(idx_in_c))
            # 在簇内按得分排序，选前 quota 个
            sorted_idx = idx_in_c[np.argsort(scores[idx_in_c])[-quota:]]
            for idx in sorted_idx:
                selected.add(int(idx))
        
        # 如果保底超过 target_size，只保留得分最高的 target_size 个
        if len(selected) > target_size:
            selected = set(np.argsort(scores)[-target_size:])
        else:
            # 剩余名额按全局得分补齐
            remaining = target_size - len(selected)
            if remaining > 0:
                unselected = [i for i in range(n) if i not in selected]
                unselected_scores = scores[unselected]
                top_local_idx = np.argsort(unselected_scores)[-remaining:]
                for idx in top_local_idx:
                    selected.add(unselected[idx])
        
        selected = np.array(sorted(list(selected)), dtype=np.int64)
        
        print(f"[Coreset] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Coreset] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
