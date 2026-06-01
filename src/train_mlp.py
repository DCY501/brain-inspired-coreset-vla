"""
MLP 回归模型定义与训练脚本
输入: [视觉特征(512) + 指令特征] -> 输出: 单臂 7DoF 动作
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *


class MLPRegressor(nn.Module):
    """
    轻量级多层感知机回归模型
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims=MLP_HIDDEN_DIMS, dropout: float = MLP_DROPOUT):
        """
        Args:
            input_dim: 输入维度（视觉特征 + 指令编码）
            output_dim: 输出维度（7DoF 动作）
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 比率
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim)
        Returns:
            out: (B, output_dim)
        """
        return self.net(x)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = EPOCHS,
                lr: float = LEARNING_RATE,
                device: str = 'cuda',
                save_path: str = None):
    """
    训练 MLP 模型，支持早停和学习率衰减
    
    Returns:
        model: 训练好的模型（已加载最优权重）
        history: dict, 包含 train_loss 和 val_loss 曲线
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # ---------- Validation ----------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | "
                  f"Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        
        # ---------- Early Stopping & Checkpoint ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
    
    # 加载最优权重
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    
    return model, history
