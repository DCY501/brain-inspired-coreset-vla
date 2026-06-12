"""
自监督重建误差采样 (Autoencoder Reconstruction Error Sampling)

思路：训练一个轻量自编码器压缩视觉特征。重建误差大的帧 = 模型"看不懂"
的帧 = 信息最丰富。模拟大脑对"意外/难预测"刺激分配更多注意力的机制。

机制：
  1. 在训练集视觉特征上训练小型 MLP 自编码器
  2. 计算每帧的重建误差（MSE）
  3. 选重建误差最大的帧作为核心集

与冗余分布思路(v1.1)的正交性：
  - v1.1 用 K-Means 空间聚类，假设"稀疏簇 = 高价值"
  - 本思路用重建误差，假设"难重建 = 高价值"
  - 两者利用的信息维度完全不同
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import *


class SimpleAutoencoder(nn.Module):
    """
    轻量 MLP 自编码器
    输入: 768d 视觉特征 -> 隐藏层 -> 重建 768d
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )
    
    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x: torch.Tensor):
        return self.encoder(x)


class AutoencoderReconstructionSelector:
    """
    自监督重建误差选择器
    基于自编码器重建误差筛选高价值数据子集
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 hidden_dim: int = 128,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 256):
        self.target_ratio = target_ratio
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
    
    def _train_ae(self, features: np.ndarray, seed: int = 42, device: str = 'cpu'):
        """
        在全部训练特征上训练自编码器
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            seed: 随机种子
            device: 计算设备
        Returns:
            model: 训练好的自编码器
        """
        n, d = features.shape
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SimpleAutoencoder(input_dim=d, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(torch.from_numpy(features).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (x,) in loader:
                x = x.to(device)
                recon = model(x)
                loss = F.mse_loss(recon, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x.size(0)
            
            avg_loss = total_loss / n
            if (epoch + 1) % 10 == 0:
                print(f"  [AE Epoch {epoch+1:03d}/{self.epochs}] Loss: {avg_loss:.6f}")
        
        return model
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行自监督重建误差采样
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征
            actions: np.ndarray, shape=(N, 7), 动作向量（本思路不使用，保留接口一致性）
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(features)
        target_size = max(1, int(n * self.target_ratio))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[AE] Training autoencoder: {features.shape[1]}d -> {self.hidden_dim}d "
              f"on {n} frames ({self.epochs} epochs)")
        
        model = self._train_ae(features, seed=42, device=device)
        
        # 计算每帧重建误差
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device)
            recon = model(x)
            errors = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
        
        # 选重建误差最大的帧
        selected = np.argsort(errors)[-target_size:]
        selected = np.array(sorted(selected), dtype=np.int64)
        
        print(f"[AE Reconstruction] Selected {len(selected)} frames out of {n} "
              f"({len(selected)/n*100:.1f}%)")
        print(f"[AE Reconstruction] Error stats (selected): "
              f"max={errors[selected].max():.6f}, "
              f"min={errors[selected].min():.6f}, "
              f"mean={errors[selected].mean():.6f}")
        print(f"[AE Reconstruction] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
