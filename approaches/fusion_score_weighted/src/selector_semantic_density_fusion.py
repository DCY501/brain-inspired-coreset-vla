"""
语义-密度联合替换选择器 (Semantic-Density Joint Replacement Fusion)

思路：在密度过滤 FPS 的替换阶段，不单纯用"全局密度最高"作为替换标准，
      而是结合 v1.1 语义多样性得分，优先选择"密度相对较高 + 语义得分也高"的帧。

与互补融合 v3.0 的根本区别：
  - v3.0：先发现语义盲区，再强制替换（可能替换太多，破坏几何覆盖）
  - 本方法：在密度过滤的替换阶段，自然地优先选择语义有价值的帧，
            替换数量固定为 filter_ratio，不会过度替换

核心机制：
  Step 1: 标准 FPS 选出 target_size 帧
  Step 2: 计算选中帧的 k-NN 局部密度，找出最稀疏的 filter_ratio 比例
  Step 3: 对全局未选帧，计算联合得分 = α * density_score + β * semantic_score
          - density_score: k-NN 平均距离的倒数（密度越高得分越高）
          - semantic_score: v1.1 的语义多样性得分（K-Means 簇内距离 × 稀疏惩罚）
  Step 4: 用联合得分最高的帧替换最稀疏的帧
  Step 5: 保持总帧数不变

设计哲学：
  - 密度过滤保证全局几何覆盖最优性（全局 k-NN）
  - 语义得分作为"替换优先级"的辅助信号，而非主导信号
  - 替换数量固定，不会破坏 FPS 的贪心覆盖完整性
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from config import *


class SemanticDensityFusionSelector:
    """
    语义-密度联合替换选择器
    密度过滤 FPS + v1.1 语义得分指导替换
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 seed: int = 42,
                 k: int = 5,
                 filter_ratio: float = 0.15,
                 alpha: float = 0.5,   # density_score 权重
                 beta: float = 0.5,    # semantic_score 权重
                 pca_components: int = 400):
        """
        Args:
            target_ratio: 目标筛选比例
            seed: 随机种子
            k: k-近邻数，用于计算局部密度
            filter_ratio: 替换比例（最低密度的多少比例被替换）
            alpha: density_score 权重
            beta: semantic_score 权重
            pca_components: v1.1 语义得分的 PCA 维度
        """
        self.target_ratio = target_ratio
        self.seed = seed
        self.k = k
        self.filter_ratio = filter_ratio
        self.alpha = alpha
        self.beta = beta
        self.pca_components = pca_components
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行语义-密度联合替换采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        print(f"[Semantic-Density Fusion] target={target_size}, filter_ratio={self.filter_ratio}, "
              f"alpha={self.alpha}, beta={self.beta}")
        
        # ===== Step 1: 标准 FPS 选出初始核心集 =====
        from approaches.farthest_point_sampling.src.selector_fps import FarthestPointSamplingSelector
        fps = FarthestPointSamplingSelector(target_ratio=self.target_ratio, seed=self.seed)
        selected = fps.select(features, actions, episode_indices).tolist()
        
        # ===== Step 2: 计算选中帧的局部密度 =====
        k = min(self.k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        distances, _ = nbrs.kneighbors(features[selected])
        avg_dists = distances[:, 1:].mean(axis=1)  # 距离越大 = 密度越低
        
        # 找出最低密度的 filter_ratio 比例
        n_replace = max(1, int(len(selected) * self.filter_ratio))
        replace_idx = np.argsort(avg_dists)[-n_replace:]  # 距离最大的 = 最稀疏
        
        print(f"[Semantic-Density] Replacing {n_replace} sparsest frames "
              f"(density threshold: {avg_dists[replace_idx[0]]:.4f})")
        
        # ===== Step 3: 计算全局未选帧的语义得分 =====
        unselected = [i for i in range(n) if i not in selected]
        
        if len(unselected) > 0:
            # 3a: 计算未选帧的密度得分（k-NN 平均距离的倒数）
            u_distances, _ = nbrs.kneighbors(features[unselected])
            u_avg_dists = u_distances[:, 1:].mean(axis=1)
            # 密度得分 = 1 / (距离 + epsilon)，距离越小 = 密度越高 = 得分越高
            density_scores = 1.0 / (u_avg_dists + 1e-8)
            
            # 3b: 计算未选帧的语义得分（v1.1 多样性得分）
            from approaches.redundancy_distribution.src.selector_v1_1_pca_best import BrainInspiredCoresetSelector
            v1_selector = BrainInspiredCoresetSelector(target_ratio=1.0, pca_components=self.pca_components)
            
            # PCA 400d 降维
            pca = PCA(n_components=self.pca_components, random_state=42)
            features_pca = pca.fit_transform(features)
            print(f"[Semantic-Density] PCA {features.shape[1]}d -> {features_pca.shape[1]}d "
                  f"(explained variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)")
            
            # 全局聚类（n_clusters 基于全局数据量）
            n_clusters = max(10, n // 30)
            semantic_scores, labels = v1_selector.compute_diversity_scores(features_pca, n_clusters=n_clusters)
            print(f"[Semantic-Density] Global K-Means: {n_clusters} clusters")
            
            # 只取未选帧的语义得分
            unselected_semantic_scores = semantic_scores[unselected]
            
            # 3c: 归一化两个得分
            if density_scores.max() > density_scores.min():
                density_scores_norm = (density_scores - density_scores.min()) / (density_scores.max() - density_scores.min() + 1e-8)
            else:
                density_scores_norm = np.ones_like(density_scores)
            
            if unselected_semantic_scores.max() > unselected_semantic_scores.min():
                semantic_scores_norm = (unselected_semantic_scores - unselected_semantic_scores.min()) / \
                                       (unselected_semantic_scores.max() - unselected_semantic_scores.min() + 1e-8)
            else:
                semantic_scores_norm = np.ones_like(unselected_semantic_scores)
            
            # 3d: 联合得分
            joint_scores = self.alpha * density_scores_norm + self.beta * semantic_scores_norm
            
            # ===== Step 4: 用联合得分最高的帧替换最稀疏的帧 =====
            best_unselected = np.argsort(joint_scores)[-n_replace:]
            
            for i, idx in enumerate(replace_idx):
                if i < len(best_unselected):
                    old = selected[idx]
                    new = unselected[best_unselected[i]]
                    selected[idx] = new
                    print(f"  Replace frame {old} (density={avg_dists[idx]:.4f}) "
                          f"-> frame {new} (density_score={density_scores_norm[best_unselected[i]]:.3f}, "
                          f"semantic_score={semantic_scores_norm[best_unselected[i]]:.3f}, "
                          f"joint={joint_scores[best_unselected[i]]:.3f})")
        
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Semantic-Density Fusion] Final: {len(selected)} frames")
        print(f"[Semantic-Density Fusion] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
