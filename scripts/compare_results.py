"""
结果对比与可视化脚本
读取 baseline、coreset-v2.0 和 coreset-v2.1 的实验结果，生成对比图表和统计摘要
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头环境使用 Agg 后端
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, REPORT_DIR


def load_results(pattern: str):
    """加载符合通配符模式的所有结果文件"""
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
    print("Comparing Baseline vs Coreset-v2.0 vs Coreset-v2.1")
    print("=" * 60)
    
    baseline = load_results("baseline_result_seed*.json")
    core_v20 = load_results("coreset_result_seed*.json")
    core_v21 = load_results("coreset_v21_*_result_seed*.json")
    
    baseline_s = print_stats("Baseline", baseline)
    core_v20_s = print_stats("Coreset v2.0 (diff-based)", core_v20)
    core_v21_s = print_stats("Coreset v2.1 (predictive coding)", core_v21)
    
    if baseline_s and core_v20_s:
        imp_v20 = (baseline_s['mean'] - core_v20_s['mean']) / baseline_s['mean'] * 100
        print(f"\n[Improvement v2.0] {imp_v20:+.2f}%")
    if baseline_s and core_v21_s:
        imp_v21 = (baseline_s['mean'] - core_v21_s['mean']) / baseline_s['mean'] * 100
        print(f"[Improvement v2.1] {imp_v21:+.2f}%")
    if core_v20_s and core_v21_s:
        imp_v21_vs_v20 = (core_v20_s['mean'] - core_v21_s['mean']) / core_v20_s['mean'] * 100
        print(f"[v2.1 vs v2.0] {imp_v21_vs_v20:+.2f}%")
    
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
    if core_v20_s:
        methods.append('Coreset\nv2.0')
        means.append(core_v20_s['mean'])
        stds.append(core_v20_s['std'])
        colors.append('#2ecc71')
    if core_v21_s:
        methods.append('Coreset\nv2.1')
        means.append(core_v21_s['mean'])
        stds.append(core_v21_s['std'])
        colors.append('#3498db')
    
    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title('Action Prediction MSE Comparison', fontsize=12, fontweight='bold')
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
    if core_v20_s and 'history' in core_v20_s['runs'][0]:
        h = core_v20_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#2ecc71', linewidth=2, marker='s', markersize=3, markevery=10)
        legend_entries.append('Coreset v2.0')
    if core_v21_s and 'history' in core_v21_s['runs'][0]:
        h = core_v21_s['runs'][0]['history']
        ax.plot(h['val_loss'], color='#3498db', linewidth=2, marker='^', markersize=3, markevery=10)
        legend_entries.append('Coreset v2.1')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation MSE', fontsize=11)
    ax.set_title('Training Convergence', fontsize=12, fontweight='bold')
    ax.legend(legend_entries, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    # ---- 图3: 每维 MSE 对比 ----
    ax = axes[2]
    dim_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    x = np.arange(len(dim_names))
    width = 0.25
    offset = 0
    
    if baseline_s:
        base_dim = np.mean([r['dim_mse'] for r in baseline_s['runs']], axis=0)
        ax.bar(x - width, base_dim, width, label='Baseline', color='#e74c3c', alpha=0.8, edgecolor='black')
    if core_v20_s:
        core_dim = np.mean([r['dim_mse'] for r in core_v20_s['runs']], axis=0)
        ax.bar(x, core_dim, width, label='Coreset v2.0', color='#2ecc71', alpha=0.8, edgecolor='black')
    if core_v21_s:
        core21_dim = np.mean([r['dim_mse'] for r in core_v21_s['runs']], axis=0)
        ax.bar(x + width, core21_dim, width, label='Coreset v2.1', color='#3498db', alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('MSE per Joint', fontsize=11)
    ax.set_title('Per-Joint Prediction Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, 'comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n[Plot saved] {plot_path}")
    
    # ==================== 保存统计摘要 ====================
    summary = {}
    if baseline_s:
        summary['baseline_avg_mse'] = float(baseline_s['mean'])
        summary['baseline_std_mse'] = float(baseline_s['std'])
    if core_v20_s:
        summary['coreset_v20_avg_mse'] = float(core_v20_s['mean'])
        summary['coreset_v20_std_mse'] = float(core_v20_s['std'])
    if core_v21_s:
        summary['coreset_v21_avg_mse'] = float(core_v21_s['mean'])
        summary['coreset_v21_std_mse'] = float(core_v21_s['std'])
    if baseline_s and core_v20_s:
        summary['improvement_v20_percent'] = float(imp_v20)
    if baseline_s and core_v21_s:
        summary['improvement_v21_percent'] = float(imp_v21)
    if core_v20_s and core_v21_s:
        summary['v21_vs_v20_percent'] = float(imp_v21_vs_v20)
    
    summary_path = os.path.join(REPORT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary saved] {summary_path}")


if __name__ == "__main__":
    main()
