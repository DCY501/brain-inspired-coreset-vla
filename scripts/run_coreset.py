"""
核心集实验脚本
使用脑启发核心集选择算法筛选 10% 高价值帧 -> 训练 MLP -> 测试集评估 MSE
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import json

from config import *
from data_loader import FeatureDataset
from coreset_selector import BrainInspiredCoresetSelector
from train_mlp import MLPRegressor, train_model
from evaluate import evaluate_model


def run_single_coreset(seed: int = RANDOM_SEED):
    """运行一次核心集实验"""
    print("\n" + "=" * 60)
    print(f"Coreset Experiment | Seed = {seed}")
    print("=" * 60)
    
    visual_features = np.load(os.path.join(FEATURES_DIR, "clip_visual_features.npy"))
    text_features = np.load(os.path.join(FEATURES_DIR, "clip_text_features.npy"))
    actions = np.load(os.path.join(FEATURES_DIR, "actions.npy"))
    ep_indices = np.load(os.path.join(FEATURES_DIR, "episode_indices.npy"))
    
    X = np.concatenate([visual_features, text_features], axis=1)
    y = actions
    
    unique_eps = np.unique(ep_indices)
    n_test = max(1, int(len(unique_eps) * TEST_EPISODE_RATIO))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_eps)
    test_eps = shuffled[:n_test]
    train_eps = shuffled[n_test:]
    
    test_mask = np.isin(ep_indices, test_eps)
    train_mask = np.isin(ep_indices, train_eps)
    
    X_train_full, y_train_full = X[train_mask], y[train_mask]
    ep_train = ep_indices[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"[Data] Train: {len(X_train_full)} frames | Test: {len(X_test)} frames")
    
    visual_train = X_train_full[:, :VISUAL_FEATURE_DIM]
    
    selector = BrainInspiredCoresetSelector(target_ratio=CORESET_RATIO)
    coreset_idx = selector.select(visual_train, ep_train)
    
    X_train_core = X_train_full[coreset_idx]
    y_train_core = y_train_full[coreset_idx]
    print(f"[Coreset] Training set size: {len(X_train_core)} frames")
    
    train_ds = FeatureDataset(X_train_core, y_train_core)
    n_val = max(1, int(len(train_ds) * 0.1))
    n_train = len(train_ds) - n_val
    train_sub, val_sub = random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE_TRAIN)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=BATCH_SIZE_TRAIN)
    
    model = MLPRegressor(input_dim=X.shape[1], output_dim=y.shape[1])
    pca_tag = f"_pca{PCA_N_COMPONENTS}" if PCA_N_COMPONENTS else ""
    save_path = os.path.join(PROCESSED_DIR, f"coreset{pca_tag}_seed{seed}.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_model(model, train_loader, val_loader, save_path=save_path, device=device)
    
    results = evaluate_model(model, test_loader, device=device)
    
    print("\n[Results] Coreset Test MSE: {:.6f} | MAE: {:.6f}".format(results['mse'], results['mae']))
    
    result_dict = {
        'method': 'brain_inspired_coreset',
        'seed': seed,
        'test_mse': results['mse'],
        'test_mae': results['mae'],
        'dim_mse': results['dim_mse'],
        'train_frames': int(len(X_train_core)),
        'test_frames': int(len(X_test)),
        'coreset_indices': coreset_idx.tolist(),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    out_path = os.path.join(PROCESSED_DIR, f"coreset{pca_tag}_result_seed{seed}.json")
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"[Saved] {out_path}")
    
    return result_dict


def main():
    res = run_single_coreset(seed=RANDOM_SEED)
    print("\n" + "=" * 60)
    print(f"Coreset Test MSE: {res['test_mse']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
