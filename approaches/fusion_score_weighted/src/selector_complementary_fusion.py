"""
互补后处理融合选择器 (Complementary Post-Processing Fusion)

思路：以全局最优的密度过滤 FPS 为基础集，用 v1.1 语义聚类检查"语义盲区"，
      对低覆盖率簇进行针对性补充，替换基础集中最冗余的帧。

与分层融合的根本区别：
  - 分层融合：v1.1 粗筛 → FPS 精筛（信息漏斗，不可逆）
  - 互补融合：FPS 全局基础 → v1.1 盲区检查 → 局部微调（信息完整保留）

核心洞察：
  密度过滤 FPS 的全局最优性（0.003792）依赖于全局 k-NN 密度估计，
  不能嵌入子空间。因此让 FPS 在全局运行，v1.1 只做"盲区检查+微调"。
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from config import *


class ComplementaryFusionSelector:
    """
    互补后处理融合选择器
    密度过滤 FPS 全局基础 + v1.1 语义盲区修正
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 seed: int = 42,
                 pca_components: int = 400,
                 coverage_threshold: float = 0.05,
                 k: int = 5):
        """
        Args:
            target_ratio: 最终核心集比例
            seed: 随机种子
            pca_components: v1.1 的 PCA 维度
            coverage_threshold: 语义盲区判定阈值（簇覆盖率低于此值则补充）
            k: k-近邻数，用于计算冗余度
        """
        self.target_ratio = target_ratio
        self.seed = seed
        self.pca_components = pca_components
        self.coverage_threshold = coverage_threshold
        self.k = k
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray,
               base_selected: set = None):
        """
        执行互补后处理融合采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量
            episode_indices: np.ndarray, shape=(N,), episode 编号
            base_selected: set, 预计算的密度过滤 FPS 基础集（避免重新计算）
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        print(f"[Complementary Fusion] target={target_size}, coverage_threshold={self.coverage_threshold}")
        
        # ===== Step 1: 密度过滤 FPS 全局选出基础集 =====
        if base_selected is not None:
            base_selected = set(base_selected)
            print(f"[Complementary] Using pre-computed base set: {len(base_selected)} frames")
        else:
            from approaches.farthest_point_sampling.src.selector_density_filtered_fps import DensityFilteredFPSSelector
            fps_selector = DensityFilteredFPSSelector(target_ratio=self.target_ratio, seed=self.seed)
            base_selected = set(fps_selector.select(features, actions, episode_indices).tolist())
            print(f"[Complementary] Base set (Density-Filtered FPS): {len(base_selected)} frames")
        
        # ===== Step 2: v1.1 全局语义聚类（仅用于盲区检查） =====
        from approaches.redundancy_distribution.src.selector_v1_1_pca_best import BrainInspiredCoresetSelector
        v1_selector = BrainInspiredCoresetSelector(target_ratio=1.0, pca_components=self.pca_components)
        
        # PCA 400d 降维
        pca = PCA(n_components=self.pca_components, random_state=42)
        features_pca = pca.fit_transform(features)
        print(f"[Complementary] PCA {features.shape[1]}d -> {features_pca.shape[1]}d "
              f"(explained variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%)")
        
        # 全局聚类（n_clusters 基于全局数据量）
        n_clusters = max(10, n // 30)
        scores, labels = v1_selector.compute_diversity_scores(features_pca, n_clusters=n_clusters)
        print(f"[Complementary] Global K-Means: {n_clusters} clusters")
        
        # ===== Step 3: 检查每个语义簇的覆盖率 =====
        blind_spots = []  # (cluster_id, missing_count, best_candidate_idx)
        
        for c in np.unique(labels):
            idx_in_c = np.where(labels == c)[0]
            cluster_size = len(idx_in_c)
            covered = len([i for i in idx_in_c if i in base_selected])
            coverage = covered / cluster_size if cluster_size > 0 else 1.0
            
            if coverage < self.coverage_threshold:
                # 找出该簇中 v1.1 得分最高且不在 base_selected 中的帧
                candidates = [i for i in idx_in_c if i not in base_selected]
                if len(candidates) > 0:
                    best_idx = candidates[np.argmax(scores[candidates])]
                    blind_spots.append({
                        'cluster': int(c),
                        'size': cluster_size,
                        'covered': covered,
                        'coverage': coverage,
                        'best_candidate': int(best_idx),
                        'candidate_score': float(scores[best_idx])
                    })
        
        print(f"[Complementary] Found {len(blind_spots)} blind spot clusters "
              f"(coverage < {self.coverage_threshold})")
        for bs in blind_spots[:10]:  # 只打印前10个
            print(f"  Cluster {bs['cluster']}: {bs['covered']}/{bs['size']} covered "
                  f"({bs['coverage']*100:.1f}%) -> candidate frame {bs['best_candidate']} "
                  f"(score={bs['candidate_score']:.4f})")
        if len(blind_spots) > 10:
            print(f"  ... and {len(blind_spots)-10} more")
        
        # ===== Step 4: 从基础集中找最冗余的帧进行替换 =====
        selected = set(base_selected)
        
        if len(blind_spots) > 0:
            # 计算 base_selected 中每帧的局部密度（k-NN 平均距离）
            # 距离越小 = 密度越高 = 越冗余
            k = min(self.k, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
            distances, _ = nbrs.kneighbors(features[list(selected)])
            avg_dists = distances[:, 1:].mean(axis=1)  # 距离越小 = 密度越高 = 越冗余
            
            # 按冗余度排序（距离从小到大 = 最冗余到最不冗余）
            selected_list = list(selected)
            redundancy_order = np.argsort(avg_dists)  # 最冗余的在前面
            
            n_replaced = 0
            for i, bs in enumerate(blind_spots):
                if i >= len(redundancy_order):
                    break
                
                # 要替换的帧：base_selected 中最冗余的
                replace_local_idx = redundancy_order[i]
                old_frame = selected_list[replace_local_idx]
                
                # 新帧：盲区簇中 v1.1 得分最高的
                new_frame = bs['best_candidate']
                
                if new_frame not in selected:
                    selected.remove(old_frame)
                    selected.add(new_frame)
                    n_replaced += 1
                    print(f"[Complementary] Replace frame {old_frame} (redundancy={avg_dists[replace_local_idx]:.4f}) "
                          f"-> frame {new_frame} (cluster {bs['cluster']}, score={bs['candidate_score']:.4f})")
            
            print(f"[Complementary] Total replaced: {n_replaced} frames")
        
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Complementary Fusion] Final: {len(selected)} frames")
        print(f"[Complementary Fusion] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
