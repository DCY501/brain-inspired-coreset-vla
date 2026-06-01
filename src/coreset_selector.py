"""
脑启发核心集选择算法 (Brain-Inspired Coreset Selection)

核心机制映射：
1. 预测编码 (Predictive Coding) -> 时序冗余过滤
   大脑持续预测下一时刻状态；当实际输入与预测一致时，神经活动被抑制。
   对应到数据：相邻帧动作变化极小 -> 预测误差极低 -> 标记为时序冗余 -> 可被修剪。

2. 网状激活系统 (RAS) + 效用过滤 -> 高信息效用聚焦
   RAS 根据任务目标过滤背景噪音，只关注高信息效用刺激。
   对应到数据：动作轨迹突变、夹爪状态变化等时刻 -> 信息效用高 -> 优先保留。

3. 分布均衡 -> 防止分布冗余
   某些简单动作占比过高会导致模型过拟合。
   对应到数据：视觉特征 K-Means 聚类 + 稀疏簇偏好 -> 保证特征空间覆盖均衡。
"""
import numpy as np
from sklearn.cluster import KMeans
from config import *


class BrainInspiredCoresetSelector:
    """
    脑启发核心集选择器
    综合时序重要性 + 分布多样性，自动筛选高价值数据子集
    """
    def __init__(self,
                 target_ratio: float = CORESET_RATIO,
                 temporal_weight: float = TEMPORAL_WEIGHT,
                 diversity_weight: float = DIVERSITY_WEIGHT):
        """
        Args:
            target_ratio: 目标筛选比例（如 0.1 = 10%）
            temporal_weight: 时序得分权重
            diversity_weight: 分布得分权重
        """
        self.target_ratio = target_ratio
        self.temporal_weight = temporal_weight
        self.diversity_weight = diversity_weight
    
    def compute_temporal_scores(self, actions: np.ndarray, episode_indices: np.ndarray):
        """
        计算时序重要性得分（对应 Predictive Coding 机制）
        原理：相邻帧动作差异越大 -> 预测误差越大 -> 信息价值越高
        
        Args:
            actions: np.ndarray, shape=(N, 7), 单臂动作
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            scores: np.ndarray, shape=(N,), 值域 [0, 1]
        """
        scores = np.zeros(len(actions), dtype=np.float32)
        
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            idx_in_ep = np.where(mask)[0]
            ep_actions = actions[mask]
            n = len(ep_actions)
            
            if n < 2:
                scores[idx_in_ep] = 1.0  # 太短则保留
                continue
            
            # 一阶差异：||a_t - a_{t-1}||
            diff1 = np.zeros(n, dtype=np.float32)
            diff1[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
            
            # 二阶差异（加速度）：检测突变时刻（夹爪接触、轨迹转折等）
            diff2 = np.zeros(n, dtype=np.float32)
            if n > 2:
                diff2[1:-1] = np.abs(diff1[2:] - diff1[1:-1])
            
            # 综合得分 = 一阶 + 二阶
            combined = diff1 + diff2
            
            # 归一化到 [0, 1]
            if combined.max() > combined.min():
                combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
            
            scores[idx_in_ep] = combined
        
        return scores
    
    def compute_diversity_scores(self, features: np.ndarray, n_clusters: int = None):
        """
        计算分布多样性得分（对应 RAS + 分布均衡机制）
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
            # 平均每个目标簇约 3 个样本，保证分布粒度
            n_clusters = max(2, target_size // 3)
        
        # K-Means 聚类（视觉特征空间）
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        
        # 到所属聚类中心的距离
        dists = np.linalg.norm(features - centers[labels], axis=1)
        
        # 簇大小惩罚：大簇中的样本重要性降低，稀疏簇重要性提高
        sizes = np.bincount(labels, minlength=n_clusters)
        size_penalty = 1.0 / np.sqrt(sizes[labels] + 1.0)
        
        scores = dists * size_penalty
        
        # 归一化到 [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores, labels
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行核心集选择
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 单臂动作
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引（相对于输入数组）
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        # 1. 时序重要性得分（Predictive Coding -> 去除发呆帧）
        t_scores = self.compute_temporal_scores(actions, episode_indices)
        
        # 2. 分布多样性得分（RAS + 均衡 -> 防止简单动作过拟合）
        d_scores, labels = self.compute_diversity_scores(features)
        
        # 3. 综合得分（加权融合）
        final_scores = self.temporal_weight * t_scores + self.diversity_weight * d_scores
        
        # 4. 选择策略：
        #    先保证每个簇至少选 1 帧（分布覆盖），再按全局得分补齐到 target_size
        selected = set()
        for c in np.unique(labels):
            idx_in_c = np.where(labels == c)[0]
            best_idx = idx_in_c[np.argmax(final_scores[idx_in_c])]
            selected.add(int(best_idx))
        
        # 剩余名额按全局分数从高到低补充
        remaining = target_size - len(selected)
        if remaining > 0:
            unselected = [i for i in range(n) if i not in selected]
            unselected_scores = final_scores[unselected]
            top_local_idx = np.argsort(unselected_scores)[-remaining:]
            for idx in top_local_idx:
                selected.add(unselected[idx])
        elif remaining < 0:
            # 如果 target_size 小于聚类数，只保留得分最高的 target_size 个（每个簇的代表）
            selected = set(np.argsort(final_scores)[-target_size:])
        
        selected = np.array(sorted(list(selected)), dtype=np.int64)
        
        print(f"[Coreset] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Coreset] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
