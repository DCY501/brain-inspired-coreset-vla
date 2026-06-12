"""
数据加载与预处理模块
负责：
1. 通过 lerobot 加载 ALOHA 数据集
2. 提取单臂 7DoF 动作
3. 按 episode 划分训练/测试集
4. 提供基于预提取特征的 PyTorch Dataset
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from config import *


class AlohaDataLoader:
    """
    ALOHA 数据集加载器
    封装 lerobot 接口，提供统一的数据访问方式
    """
    def __init__(self, root=DATA_ROOT, repo_id=REPO_ID):
        print(f"Loading dataset from: {root}")
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)
        self.num_frames = len(self.dataset)
        # 通过遍历获取 episode 数量（兼容不同版本 lerobot）
        self.num_episodes = len(np.unique([self.dataset[i]["episode_index"] for i in range(self.num_frames)]))
        print(f"Dataset loaded: {self.num_frames} frames, {self.num_episodes} episodes")
    
    def get_frame(self, idx: int):
        """获取第 idx 帧的原始样本字典"""
        return self.dataset[idx]
    
    def get_all_actions(self):
        """
        遍历全部帧，提取动作向量（已降维为单臂 7DoF）
        Returns:
            actions: np.ndarray, shape=(N, 7)
        """
        actions = []
        for i in range(self.num_frames):
            act = self.dataset[i][ACTION_KEY].numpy()
            if USE_SINGLE_ARM:
                act = act[SINGLE_ARM_DIMS]
            actions.append(act)
        return np.array(actions, dtype=np.float32)
    
    def get_all_states(self):
        """
        遍历全部帧，提取状态向量（已降维为单臂 7DoF）
        Returns:
            states: np.ndarray, shape=(N, 7)
        """
        states = []
        for i in range(self.num_frames):
            st = self.dataset[i][STATE_KEY].numpy()
            if USE_SINGLE_ARM:
                st = st[SINGLE_ARM_DIMS]
            states.append(st)
        return np.array(states, dtype=np.float32)
    
    def get_all_episode_indices(self):
        """
        获取每帧对应的 episode 编号
        Returns:
            ep_indices: np.ndarray, shape=(N,)
        """
        return np.array([int(self.dataset[i]["episode_index"]) for i in range(self.num_frames)], dtype=np.int64)
    
    def get_all_task_texts(self):
        """获取每帧对应的语言指令文本"""
        return [str(self.dataset[i][TASK_KEY]) for i in range(self.num_frames)]
    
    def get_image_batch(self, indices, device='cpu'):
        """
        按索引批量获取图像 tensor（用于特征提取）
        Args:
            indices: list or np.ndarray of int
            device: torch device
        Returns:
            images: torch.Tensor, shape=(B, 3, H, W), dtype=float32
        """
        images = []
        for i in indices:
            img = self.dataset[int(i)][IMAGE_KEY]  # (3, H, W)
            images.append(img)
        return torch.stack(images).to(device)
    
    def split_train_test(self, test_ratio=TEST_EPISODE_RATIO, seed=RANDOM_SEED):
        """
        按 episode 级别划分训练集与测试集，保证数据隔离
        Args:
            test_ratio: 测试集 episode 比例
            seed: 随机种子
        Returns:
            train_idx: np.ndarray, 训练帧索引
            test_idx: np.ndarray, 测试帧索引
        """
        rng = np.random.RandomState(seed)
        all_eps = np.arange(self.num_episodes)
        rng.shuffle(all_eps)
        
        n_test = max(1, int(self.num_episodes * test_ratio))
        test_eps = all_eps[:n_test]
        train_eps = all_eps[n_test:]
        
        ep_indices = self.get_all_episode_indices()
        train_mask = np.isin(ep_indices, train_eps)
        test_mask = np.isin(ep_indices, test_eps)
        
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        print(f"[Split] Train episodes: {train_eps} ({len(train_idx)} frames)")
        print(f"[Split] Test episodes:  {test_eps} ({len(test_idx)} frames)")
        return train_idx, test_idx


class FeatureDataset(Dataset):
    """
    PyTorch Dataset，基于已离线提取的视觉特征
    样本格式: (feature_vector, action_vector)
    """
    def __init__(self, features: np.ndarray, actions: np.ndarray):
        """
        Args:
            features: np.ndarray, shape=(N, D)
            actions: np.ndarray, shape=(N, 7)
        """
        self.features = torch.from_numpy(features).float()
        self.actions = torch.from_numpy(actions).float()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.actions[idx]
