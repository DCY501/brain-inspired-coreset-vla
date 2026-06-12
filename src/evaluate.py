"""
模型评估模块
在固定测试集上计算 MSE、MAE 及每维动作误差
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import *


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: str = 'cuda'):
    """
    在测试集上评估模型性能
    
    Args:
        model: 训练好的 MLPRegressor
        test_loader: 测试集 DataLoader
        device: 'cuda' 或 'cpu'
    Returns:
        results: dict, 包含 mse, mae, dim_mse, preds, targets
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.numpy())
    
    preds = np.concatenate(all_preds, axis=0)      # (N_test, 7)
    targets = np.concatenate(all_targets, axis=0)  # (N_test, 7)
    
    # 整体 MSE / MAE
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    
    # 每维 MSE（用于分析哪个关节预测最难）
    dim_mse = np.mean((preds - targets) ** 2, axis=0).tolist()
    
    return {
        'mse': mse,
        'mae': mae,
        'dim_mse': dim_mse,
        'preds': preds,
        'targets': targets
    }
