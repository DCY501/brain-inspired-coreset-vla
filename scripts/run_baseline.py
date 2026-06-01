"""
Baseline 实验脚本
随机抽取 10% 训练 episode -> 训练 MLP -> 测试集评估 MSE
支持多次随机采样取平均，排除随机波动
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
from baseline import RandomBaselineSampler
from train_mlp import MLPRegressor, train_model
from evaluate import evaluate_model


def run_single_baseline(seed: int):
    """运行一次 baseline 实验"""
    print("\n" + "=" * 60)
    print(f"Baseline Experiment | Seed = {seed}")
    print("=" * 60)
    
    # ---------- 加载 CLIP 多模态特征 ----------
    visual_features = np.load(os.path.join(FEATURES_DIR, "clip_visual_features.npy"))
    text_features = np.load(os.path.join(FEATURES_DIR, "clip_text_features.npy"))
    actions = np.load(os.path.join(FEATURES_DIR, "actions.npy"))
    ep_indices = np.load(os.path.join(FEATURES_DIR, "episode_indices.npy"))
    
    # 构建输入: [视觉特征(768) ; 文本特征(512)] = (N, 1280)
    # 单任务下文本特征为常数，但保留完整 512 维以与多任务 VLA 框架一致
    X = np.concatenate([visual_features, text_features], axis=1)
    y = actions
    
    # ---------- 按 episode 划分训练/测试 ----------
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
    
    print(f"[Data] Train: {len(X_train_full)} frames (eps {train_eps}) | Test: {len(X_test)} frames (eps {test_eps})")
    
    # ---------- Baseline: 随机抽 10% 训练 episode ----------
    sampler = RandomBaselineSampler(sample_ratio=BASELINE_SAMPLE_RATIO, seed=seed)
    baseline_idx = sampler.sample(ep_train)
    
    X_train_base = X_train_full[baseline_idx]
    y_train_base = y_train_full[baseline_idx]
    print(f"[Baseline] Training set size: {len(X_train_base)} frames")
    
    # ---------- 构造 DataLoader ----------
    train_ds = FeatureDataset(X_train_base, y_train_base)
    n_val = max(1, int(len(train_ds) * 0.1))
    n_train = len(train_ds) - n_val
    train_sub, val_sub = random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE_TRAIN)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=BATCH_SIZE_TRAIN)
    
    # ---------- 训练 MLP ----------
    model = MLPRegressor(input_dim=X.shape[1], output_dim=y.shape[1])
    save_path = os.path.join(PROCESSED_DIR, f"baseline_seed{seed}.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_model(model, train_loader, val_loader, save_path=save_path, device=device)
    
    # ---------- 测试集评估 ----------
    results = evaluate_model(model, test_loader, device=device)
    
    print("\n[Results] Baseline Test MSE: {:.6f} | MAE: {:.6f}".format(results['mse'], results['mae']))
    
    # ---------- 保存结果 ----------
    result_dict = {
        'method': 'baseline_random_clip',
        'seed': seed,
        'test_mse': results['mse'],
        'test_mae': results['mae'],
        'dim_mse': results['dim_mse'],
        'train_frames': int(len(X_train_base)),
        'test_frames': int(len(X_test)),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    out_path = os.path.join(PROCESSED_DIR, f"baseline_result_seed{seed}.json")
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"[Saved] {out_path}")
    
    return result_dict


def main():
    n_runs = 5  # 多次运行取平均，减少随机波动
    all_results = []
    
    for run in range(n_runs):
        res = run_single_baseline(seed=RANDOM_SEED + run)
        all_results.append(res)
    
    mses = [r['test_mse'] for r in all_results]
    avg_mse = float(np.mean(mses))
    std_mse = float(np.std(mses))
    
    print("\n" + "=" * 60)
    print(f"Baseline Summary ({n_runs} runs)")
    print(f"  Average Test MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print("=" * 60)
    
    summary = {
        'method': 'baseline_random_clip',
        'n_runs': n_runs,
        'avg_mse': avg_mse,
        'std_mse': std_mse,
        'all_runs': all_results
    }
    with open(os.path.join(PROCESSED_DIR, "baseline_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
