"""
分层融合选择器 v2.0 (Hierarchical Fusion)

思路：两层结构，v1.1 负责语义粗筛，FPS 负责几何精筛。

机制：
  Layer 1 (粗筛): v1.1 + PCA 400d 选出较大的候选池
    → 保证语义覆盖，动态配额防止大簇信息丢失
  
  Layer 2 (精筛): 在候选池内运行标准 FPS 选出最终核心集
    → 在语义已覆盖的基础上，优化几何最大最小距离覆盖

与得分加权融合(v1)的区别：
  - v1: 三种静态得分线性加权，丢失方法的过程性优势
  - v2: 保留各方法的完整选择过程，分层执行
"""
import numpy as np
from config import *


class HierarchicalFusionSelector:
    """
    分层融合选择器 v2.0
    v1.1 粗筛候选池 + FPS 精筛最终集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 candidate_ratio: float = 2.0,
                 pca_components: int = 400,
                 seed: int = 42):
        """
        Args:
            target_ratio: 最终核心集比例
            candidate_ratio: 候选池大小倍数（候选池 = target_size * candidate_ratio）
            pca_components: v1.1 的 PCA 维度
            seed: 随机种子
        """
        self.target_ratio = target_ratio
        self.candidate_ratio = candidate_ratio
        self.pca_components = pca_components
        self.seed = seed
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行分层融合采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        print(f"[Hierarchical Fusion] target={target_size}, candidate_ratio={self.candidate_ratio}")
        
        # ===== Layer 1: v1.1 + PCA 400d 粗筛候选池 =====
        v1_ratio = min(1.0, self.target_ratio * self.candidate_ratio)
        from approaches.redundancy_distribution.src.selector_v1_1_pca_best import BrainInspiredCoresetSelector as V1Selector
        v1_selector = V1Selector(target_ratio=v1_ratio, pca_components=self.pca_components)
        # 基于原始数据量计算固定 n_clusters，避免 candidate_ratio 过大时超时
        fixed_n_clusters = max(10, len(features) // 30)
        print(f"[Hierarchical] Fixed n_clusters for v1.1: {fixed_n_clusters} (based on {len(features)} frames)")
        candidates = v1_selector.select(features, episode_indices, n_clusters=fixed_n_clusters)
        
        print(f"[Hierarchical] Layer 1 (v1.1+PCA{self.pca_components}): {len(candidates)} candidates")
        
        # ===== Layer 2: FPS 在候选池内精筛 =====
        pool_size = len(candidates)
        fps_ratio = target_size / pool_size
        
        from approaches.farthest_point_sampling.src.selector_fps import FarthestPointSamplingSelector
        fps_selector = FarthestPointSamplingSelector(target_ratio=fps_ratio, seed=self.seed)
        
        # 在候选池内运行 FPS
        fps_local_idx = fps_selector.select(
            features[candidates],
            actions[candidates],
            episode_indices[candidates]
        )
        
        # 映射回全局索引
        selected = candidates[fps_local_idx]
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Hierarchical] Layer 2 (FPS in pool): {len(selected)} frames")
        print(f"[Hierarchical] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
