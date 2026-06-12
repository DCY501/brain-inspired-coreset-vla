"""
时序事件边界采样 (Temporal Event Boundary Sampling)

思路：机器人操作由离散动作事件串联而成。事件边界处（动作突变）的信息价值
远高于稳定段（动作缓慢变化）。模拟海马体的事件分割理论——大脑对行为转换
时刻分配更多注意力。

机制：
  1. 计算每帧的动作变化率 δ(t) = ||a_t - a_{t-1}||_2
  2. 在每个 episode 内检测局部峰值（事件边界）
  3. 事件边界帧保底入选 + 全局变化率得分补齐/截断
"""
import numpy as np
from config import *


class TemporalEventBoundarySelector:
    """
    时序事件边界选择器
    基于动作变化率检测事件边界，优先采样行为转换帧
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 boundary_percentile: float = 85.0,
                 min_peak_distance: int = 5):
        """
        Args:
            target_ratio: 目标筛选比例
            boundary_percentile: 事件边界阈值（变化率百分位数）
            min_peak_distance: 两个边界之间的最小帧间隔
        """
        self.target_ratio = target_ratio
        self.boundary_percentile = boundary_percentile
        self.min_peak_distance = min_peak_distance
    
    def compute_action_deltas(self, actions: np.ndarray, episode_indices: np.ndarray):
        """
        计算动作变化率，跨 episode 边界处重置为 0
        
        Returns:
            deltas: np.ndarray, shape=(N,), 每帧的动作变化率
        """
        n = len(actions)
        deltas = np.zeros(n)
        
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            idx = np.where(mask)[0]
            if len(idx) <= 1:
                continue
            
            ep_actions = actions[idx]
            # L2 范数衡量跨关节动作变化强度
            d = np.zeros(len(idx))
            d[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
            deltas[idx] = d
        
        return deltas
    
    def find_event_boundaries(self, deltas: np.ndarray, episode_indices: np.ndarray):
        """
        在每个 episode 内检测局部峰值作为事件边界
        
        Returns:
            boundaries: set(int), 事件边界帧的索引
        """
        boundaries = set()
        
        for ep in np.unique(episode_indices):
            idx = np.where(episode_indices == ep)[0]
            ep_deltas = deltas[idx]
            
            if len(ep_deltas) < 5:
                continue
            
            threshold = np.percentile(ep_deltas, self.boundary_percentile)
            
            # 找局部峰值（比左右邻居都大，且超过阈值）
            i = 1
            while i < len(idx) - 1:
                curr = ep_deltas[i]
                if curr > threshold and curr > ep_deltas[i - 1] and curr > ep_deltas[i + 1]:
                    boundaries.add(int(idx[i]))
                    i += self.min_peak_distance  # 跳过最小间隔，避免过密
                else:
                    i += 1
        
        return boundaries
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行时序事件边界采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征（本思路不使用，保留接口一致性）
            actions: np.ndarray, shape=(N, 7), 动作向量
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(actions)
        target_size = max(1, int(n * self.target_ratio))
        
        # Step 1: 计算动作变化率
        deltas = self.compute_action_deltas(actions, episode_indices)
        
        # Step 2: 检测事件边界
        boundaries = self.find_event_boundaries(deltas, episode_indices)
        
        # Step 3: 分层采样
        selected = set(boundaries)
        
        # Layer 1: 每个 episode 至少保底 1 帧（变化率最高的）
        for ep in np.unique(episode_indices):
            idx = np.where(episode_indices == ep)[0]
            if not any(int(i) in selected for i in idx):
                best = idx[np.argmax(deltas[idx])]
                selected.add(int(best))
        
        # Layer 2: 若超量，按变化率截断
        if len(selected) > target_size:
            selected_arr = np.array(sorted(list(selected)))
            top_idx = np.argsort(deltas[selected_arr])[-target_size:]
            selected = set(selected_arr[top_idx])
        
        # Layer 3: 若不足，全局按变化率补齐
        elif len(selected) < target_size:
            remaining = target_size - len(selected)
            unselected = [i for i in range(n) if i not in selected]
            unselected_deltas = deltas[unselected]
            top_local = np.argsort(unselected_deltas)[-remaining:]
            for i in top_local:
                selected.add(unselected[i])
        
        selected = np.array(sorted(list(selected)), dtype=np.int64)
        
        n_boundaries = len(boundaries)
        print(f"[Temporal] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[Temporal] Event boundaries detected: {n_boundaries} "
              f"({n_boundaries/len(selected)*100:.1f}% of coreset)")
        print(f"[Temporal] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
