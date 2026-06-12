"""
得分加权融合选择器 (Score Fusion Selector)

思路：在 PCA 400d 降维后的特征空间中，融合三种独立信号：
  1. v1.1 语义稀疏得分：K-Means 簇内距离 × 稀疏惩罚
  2. FPS 几何偏离得分：到最近邻的距离（孤立度）
  3. 密度典型性得分：k-NN 平均距离的倒数（局部典型性）

融合公式：final_score = α × s_v1 + β × s_fps + γ × s_density

与单一思路的关系：
  - v1.1  alone：只利用语义稀疏性，有 K-Means 假设偏差
  - FPS alone：只利用几何距离，无稀疏偏好，易选噪声
  - 融合：语义稀疏 + 几何偏离 + 局部典型，三重验证
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from config import *


class ScoreFusionSelector:
    """
    得分加权融合选择器
    融合 v1.1、FPS、密度三种信号选择核心集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 pca_components: int = 400,
                 seed: int = 42):
        """
        Args:
            target_ratio: 目标筛选比例
            alpha: v1.1 语义稀疏得分权重
            beta: FPS 几何偏离得分权重
            gamma: 密度典型性得分权重
            pca_components: PCA 降维维度
            seed: 随机种子
        """
        self.target_ratio = target_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pca_components = pca_components
        self.seed = seed
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行得分加权融合采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量（本思路不使用，保留接口一致性）
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        # Step 1: PCA 400d 降维
        pca = PCA(n_components=self.pca_components, random_state=self.seed)
        features_pca = pca.fit_transform(features)
        var_ratio = np.sum(pca.explained_variance_ratio_) * 100
        print(f"[Fusion] PCA {features.shape[1]}d -> {features_pca.shape[1]}d "
              f"(explained variance: {var_ratio:.1f}%)")
        
        # Step 2: 计算 v1.1 语义稀疏得分
        # s_v1[i] = dist_to_cluster_center × sparse_penalty
        n_clusters = max(2, target_size // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        centers = kmeans.cluster_centers_
        dists = np.linalg.norm(features_pca - centers[labels], axis=1)
        sizes = np.bincount(labels, minlength=n_clusters)
        size_penalty = 1.0 / np.sqrt(sizes[labels] + 1.0)
        s_v1 = dists * size_penalty
        
        # Step 3: 计算 FPS 几何偏离得分
        # s_fps[i] = distance_to_nearest_neighbor（几何孤立度）
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(features_pca)
        distances, _ = nbrs.kneighbors(features_pca)
        s_fps = distances[:, 1]  # 到最近邻的距离
        
        # Step 4: 计算密度典型性得分
        # s_density[i] = 1 / mean(k-NN distances)（局部典型性）
        k = min(5, n - 1)
        nbrs_k = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features_pca)
        distances_k, _ = nbrs_k.kneighbors(features_pca)
        s_density = 1.0 / (distances_k[:, 1:].mean(axis=1) + 1e-8)
        
        # Step 5: 归一化到 [0, 1]
        s_v1 = (s_v1 - s_v1.min()) / (s_v1.max() - s_v1.min() + 1e-8)
        s_fps = (s_fps - s_fps.min()) / (s_fps.max() - s_fps.min() + 1e-8)
        s_density = (s_density - s_density.min()) / (s_density.max() - s_density.min() + 1e-8)
        
        # Step 6: 加权融合
        final_score = (self.alpha * s_v1 + 
                       self.beta * s_fps + 
                       self.gamma * s_density)
        
        # Step 7: 选 top target_size
        selected = np.argsort(final_score)[-target_size:]
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Score Fusion] Weights: α={self.alpha}, β={self.beta}, γ={self.gamma}")
        print(f"[Score Fusion] Score stats: max={final_score[selected].max():.4f}, "
              f"min={final_score[selected].min():.4f}, "
              f"mean={final_score[selected].mean():.4f}")
        print(f"[Score Fusion] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Score Fusion] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
