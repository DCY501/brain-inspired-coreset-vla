"""
消融实验结果对比与可视化脚本
读取 Baseline + v1(temporal_only) + v2(diversity_only) + v3(fusion) 的结果
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, REPORT_DIR


def load_results(pattern: str):
    files = glob.glob(os.path.join(PROCESSED_DIR, pattern))
    results = []
    for f in sorted(files):
        with open(f, 'r') as fp:
            results.append(json.load(fp))
    return results


def print_stats(name, runs):
    if not runs:
        print(f"[{name}] No results found")
        return None
    mses = [r['test_mse'] for r in runs]
    mean = np.mean(mses)
    std = np.std(mses)
    print(f"\n[{name}] {len(runs)} runs")
    print(f"  Test MSE: {mean:.6f} ± {std:.6f}")
    print(f"  Range:    [{np.min(mses):.6f}, {np.max(mses):.6f}]")
    return {'mean': mean, 'std': std, 'mses': mses, 'runs': runs}


def main():
    print("=" * 60)
    print("Ablation Study: Baseline vs v1 vs v2 vs v3")
    print("=" * 60)
    
    baseline = load_results("baseline_result_seed*.json")
    v1 = load_results("coreset_temporal_only_seed*.json")
    v2 = load_results("coreset_diversity_only_seed*.json")
    v3 = load_results("coreset_fusion_seed*.json")
    
    baseline_s = print_stats("Baseline (Random)", baseline)
    v1_s = print_stats("v1 Temporal Only", v1)
    v2_s = print_stats("v2 Diversity Only", v2)
    v3_s = print_stats("v3 Fusion", v3)
    
    if baseline_s and v3_s:
        imp = (baseline_s['mean'] - v3_s['mean']) / baseline_s['mean'] * 100
        print(f"\n[v3 vs Baseline] {imp:+.2f}%")
    
    # ==================== 绘制对比图 ====================
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # ---- 图1: MSE 柱状图 ----
    ax = axes[0]
    methods = []
    means = []
    stds = []
    colors = []
    
    if baseline_s:
        methods.append('Random\nBaseline')
        means.append(baseline_s['mean'])
        stds.append(baseline_s['std'])
        colors.append('#e74c3c')
    if v1_s:
        methods.append('v1\nTemporal')
        means.append(v1_s['mean'])
        stds.append(v1_s['std'])
        colors.append('#f39c12')
    if v2_s:
        methods.append('v2\nDiversity')
        means.append(v2_s['mean'])
        stds.append(v2_s['std'])
        colors.append('#3498db')
    if v3_s:
        methods.append('v3\nFusion')
        means.append(v3_s['mean'])
        stds.append(v3_s['std'])
        colors.append('#2ecc71')
    
    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title('Ablation Study: Action Prediction MSE', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s,
                f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ---- 图2: 学习曲线 ----
    ax = axes[1]
    legend_entries = []
    if baseline_s and 'history' in baseline_s['runs'][0]:
        h = baseline_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#e74c3c', linewidth=2, marker='o', markersize=3, markevery=10)
        legend_entries.append('Baseline')
    if v1_s and 'history' in v1_s['runs'][0]:
        h = v1_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#f39c12', linewidth=2, marker='^', markersize=3, markevery=10)
        legend_entries.append('v1 Temporal')
    if v2_s and 'history' in v2_s['runs'][0]:
        h = v2_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#3498db', linewidth=2, marker='s', markersize=3, markevery=10)
        legend_entries.append('v2 Diversity')
    if v3_s and 'history' in v3_s['runs'][0]:
        h = v3_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#2ecc71', linewidth=2, marker='D', markersize=3, markevery=10)
        legend_entries.append('v3 Fusion')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation MSE', fontsize=11)
    ax.set_title('Training Convergence', fontsize=12, fontweight='bold')
    ax.legend(legend_entries, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # ---- 图3: 每维 MSE 对比 ----
    ax = axes[2]
    dim_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    x = np.arange(len(dim_names))
    width = 0.2
    offset = -1.5
    
    if baseline_s:
        base_dim = np.mean([r['dim_mse'] for r in baseline_s['runs']], axis=0)
        ax.bar(x + width*offset, base_dim, width, label='Baseline', color='#e74c3c', alpha=0.8, edgecolor='black')
        offset += 1
    if v1_s:
        v1_dim = np.mean([r['dim_mse'] for r in v1_s['runs']], axis=0)
        ax.bar(x + width*offset, v1_dim, width, label='v1 Temporal', color='#f39c12', alpha=0.8, edgecolor='black')
        offset += 1
    if v2_s:
        v2_dim = np.mean([r['dim_mse'] for r in v2_s['runs']], axis=0)
        ax.bar(x + width*offset, v2_dim, width, label='v2 Diversity', color='#3498db', alpha=0.8, edgecolor='black')
        offset += 1
    if v3_s:
        v3_dim = np.mean([r['dim_mse'] for r in v3_s['runs']], axis=0)
        ax.bar(x + width*offset, v3_dim, width, label='v3 Fusion', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('MSE per Joint', fontsize=11)
    ax.set_title('Per-Joint Prediction Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, 'ablation_comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n[Plot saved] {plot_path}")
    
    # ==================== 保存统计摘要 ====================
    summary = {}
    if baseline_s:
        summary['baseline_avg_mse'] = float(baseline_s['mean'])
        summary['baseline_std_mse'] = float(baseline_s['std'])
    if v1_s:
        summary['v1_temporal_avg_mse'] = float(v1_s['mean'])
        summary['v1_temporal_std_mse'] = float(v1_s['std'])
    if v2_s:
        summary['v2_diversity_avg_mse'] = float(v2_s['mean'])
        summary['v2_diversity_std_mse'] = float(v2_s['std'])
    if v3_s:
        summary['v3_fusion_avg_mse'] = float(v3_s['mean'])
        summary['v3_fusion_std_mse'] = float(v3_s['std'])
    if baseline_s and v3_s:
        summary['v3_improvement_percent'] = float((baseline_s['mean'] - v3_s['mean']) / baseline_s['mean'] * 100)
    
    summary_path = os.path.join(REPORT_DIR, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary saved] {summary_path}")


if __name__ == "__main__":
    main()
