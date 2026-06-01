"""
Baseline 模块：随机抽样
从训练集中随机抽取 10% 的 episodes 作为 Baseline 训练数据
"""
import numpy as np
from config import *


class RandomBaselineSampler:
    """
    随机基线采样器
    按 episode 级别随机抽取指定比例
    """
    def __init__(self, sample_ratio: float = BASELINE_SAMPLE_RATIO, seed: int = RANDOM_SEED):
        """
        Args:
            sample_ratio: 采样比例（episode 级别）
            seed: 随机种子，保证可复现
        """
        self.sample_ratio = sample_ratio
        self.seed = seed
    
    def sample(self, train_episode_indices: np.ndarray):
        """
        从训练 episode 中随机抽取
        Args:
            train_episode_indices: shape=(N_train,), 每帧对应的 episode 编号
        Returns:
            selected_frames: np.ndarray, 被选中的帧索引（相对于训练集）
        """
        unique_eps = np.unique(train_episode_indices)
        n_sample = max(1, int(len(unique_eps) * self.sample_ratio))
        
        rng = np.random.RandomState(self.seed)
        selected_eps = rng.choice(unique_eps, size=n_sample, replace=False)
        
        mask = np.isin(train_episode_indices, selected_eps)
        selected_frames = np.where(mask)[0]
        
        print(f"[RandomBaseline] Selected {len(selected_eps)} episodes "
              f"({len(selected_frames)} frames) out of {len(unique_eps)} train episodes")
        return selected_frames
