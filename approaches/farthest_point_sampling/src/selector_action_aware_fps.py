"""
动作感知密度过滤最远点采样 (Action-Aware Density-Filtered FPS)

核心洞察：
  当前密度过滤只在视觉特征空间做 k-NN 密度估计，但目标是预测动作。
  两个帧视觉接近、动作也接近 → 真正冗余
  两个帧视觉接近、但动作差异大 → 高价值关键帧（动作变化剧烈）
  当前密度过滤会错误地把第二类帧当作"噪声"替换掉。

改进方案：
  动作感知密度 = 视觉密度 × 动作多样性奖励
  
  density_aware = visual_density × (1 + λ × action_diversity)
  
  - visual_density: k-NN 平均距离的倒数（距离越小 = 密度越高）
  - action_diversity: k-NN 邻居的动作标准差（动作变化越大 = 多样性越高）
  - λ: 奖励系数

效果：
  - 视觉密集 + 动作单一 → 真正冗余（保留低密度 = 被替换）
  - 视觉密集 + 动作多样 → 高价值帧（密度奖励高，不会被替换）
  - 视觉稀疏 + 动作单一 → 可能是噪声（低密度，会被替换）
  - 视觉稀疏 + 动作多样 → 关键过渡帧（密度奖励高，保留！）
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import *


class ActionAwareDensityFilteredFPSSelector:
    """
    动作感知密度过滤最远点采样选择器
    在密度过滤中引入动作多样性奖励，保留动作变化剧烈的关键帧
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 seed: int = 42,
                 k: int = 5,
                 filter_ratio: float = 0.15,
                 lambda_reward: float = 2.0):
        """
        Args:
            target_ratio: 目标筛选比例
            seed: 随机种子
            k: k-近邻数，用于计算局部密度和动作多样性
            filter_ratio: 替换比例（最低密度的多少比例被替换）
            lambda_reward: 动作多样性奖励系数
        """
        self.target_ratio = target_ratio
        self.seed = seed
        self.k = k
        self.filter_ratio = filter_ratio
        self.lambda_reward = lambda_reward
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """执行动作感知密度过滤 FPS"""
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        
        # Step 1: 标准 FPS 选出初始核心集
        from approaches.farthest_point_sampling.src.selector_fps import FarthestPointSamplingSelector
        fps = FarthestPointSamplingSelector(target_ratio=self.target_ratio, seed=self.seed)
        selected = fps.select(features, actions, episode_indices).tolist()
        
        # Step 2: 计算全局 k-NN（视觉特征空间）
        k = min(self.k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        distances, neighbor_indices = nbrs.kneighbors(features)
        
        # Step 3: 计算选中帧的"动作感知密度"
        # 3a: 视觉密度（k-NN 平均距离，距离越大 = 密度越低）
        visual_dists = distances[:, 1:].mean(axis=1)
        visual_density = 1.0 / (visual_dists + 1e-8)
        
        # 3b: 动作多样性（k-NN 邻居的动作标准差）
        action_diversity = np.zeros(n)
        for i in range(n):
            neighbors = neighbor_indices[i, 1:]  # 排除自身
            neighbor_actions = actions[neighbors]
            # 动作多样性 = 邻居动作的标准差（跨所有 7 个维度取平均）
            action_diversity[i] = np.mean(np.std(neighbor_actions, axis=0))
        
        # 3c: 动作感知密度 = 视觉密度 × (1 + λ × 动作多样性)
        # 归一化动作多样性到 [0, 1]
        if action_diversity.max() > action_diversity.min():
            action_diversity_norm = (action_diversity - action_diversity.min()) / \
                                    (action_diversity.max() - action_diversity.min() + 1e-8)
        else:
            action_diversity_norm = np.zeros(n)
        
        action_aware_density = visual_density * (1.0 + self.lambda_reward * action_diversity_norm)
        
        # Step 4: 找出选中帧中"动作感知密度最低"的 filter_ratio 比例
        selected_density = action_aware_density[selected]
        n_replace = max(1, int(len(selected) * self.filter_ratio))
        replace_idx = np.argsort(selected_density)[:n_replace]  # 密度最低的 = 最稀疏且动作单一
        
        print(f"[Action-Aware FPS] Replacing {n_replace} frames with lowest action-aware density")
        print(f"  Visual density range: [{visual_density[selected].min():.4f}, {visual_density[selected].max():.4f}]")
        print(f"  Action diversity range: [{action_diversity[selected].min():.4f}, {action_diversity[selected].max():.4f}]")
        print(f"  Action-aware density range: [{action_aware_density[selected].min():.4f}, {action_aware_density[selected].max():.4f}]")
        
        # Step 5: 从全局未选帧中，找"动作感知密度最高"的帧替换
        unselected = [i for i in range(n) if i not in selected]
        if len(unselected) > 0:
            unselected_density = action_aware_density[unselected]
            best_unselected = np.argsort(unselected_density)[-n_replace:]
            
            for i, idx in enumerate(replace_idx):
                if i < len(best_unselected):
                    old = selected[idx]
                    new = unselected[best_unselected[i]]
                    selected[idx] = new
                    print(f"  Replace frame {old} (vis_density={visual_density[old]:.4f}, "
                          f"action_div={action_diversity[old]:.4f}, "
                          f"aware_density={action_aware_density[old]:.4f}) "
                          f"-> frame {new} (vis_density={visual_density[new]:.4f}, "
                          f"action_div={action_diversity[new]:.4f}, "
                          f"aware_density={action_aware_density[new]:.4f})")
        
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[Action-Aware FPS] Final: {len(selected)} frames")
        print(f"[Action-Aware FPS] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
