"""
得分加权融合实验脚本
对比多种权重组合：v1.1 主导 / FPS 主导 / 平衡 / 密度主导
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import json

from config import *
from data_loader import FeatureDataset
from approaches.fusion_score_weighted.src.selector_score_fusion import ScoreFusionSelector
from train_mlp import MLPRegressor, train_model
from evaluate import evaluate_model


def run_single(alpha: float, beta: float, gamma: float, seed: int = RANDOM_SEED):
    """运行指定权重组合的融合实验"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n" + "=" * 70)
    print(f"Score Fusion | α={alpha} β={beta} γ={gamma} | Seed={seed}")
    print("=" * 70)
    
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
    
    visual_train = X_train_full[:, :VISUAL_FEATURE_DIM]
    
    selector = ScoreFusionSelector(
        target_ratio=CORESET_RATIO,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        pca_components=400,
        seed=seed
    )
    coreset_idx = selector.select(visual_train, y_train_full, ep_train)
    
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
    tag = f"a{alpha}_b{beta}_g{gamma}"
    save_path = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        f"fusion_{tag}_seed{seed}.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, history = train_model(model, train_loader, val_loader, save_path=save_path, device=device)
    
    results = evaluate_model(model, test_loader, device=device)
    
    print(f"\n[Results] α={alpha} β={beta} γ={gamma} Test MSE: {results['mse']:.6f} | MAE: {results['mae']:.6f}")
    
    result_dict = {
        'method': 'score_fusion',
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'seed': seed,
        'test_mse': results['mse'],
        'test_mae': results['mae'],
        'dim_mse': results['dim_mse'],
        'train_frames': int(len(X_train_core)),
        'test_frames': int(len(X_test)),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    out_path = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        f"fusion_{tag}_result_seed{seed}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"[Saved] {out_path}")
    
    return result_dict


def main():
    # 四种权重组合
    configs = [
        (0.5, 0.3, 0.2, "v1.1主导"),
        (0.3, 0.5, 0.2, "FPS主导"),
        (0.4, 0.3, 0.3, "平衡"),
        (0.3, 0.2, 0.5, "密度主导"),
    ]
    
    all_results = []
    for alpha, beta, gamma, label in configs:
        res = run_single(alpha=alpha, beta=beta, gamma=gamma, seed=RANDOM_SEED)
        res['label'] = label
        all_results.append(res)
    
    print("\n" + "=" * 70)
    print("Score Fusion Summary")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['label']:10s} (α={r['alpha']:.1f} β={r['beta']:.1f} γ={r['gamma']:.1f}): "
              f"MSE = {r['test_mse']:.6f}")


if __name__ == "__main__":
    main()
