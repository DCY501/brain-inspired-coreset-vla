"""
结果对比与可视化脚本
读取 baseline 和 coreset 的实验结果，生成对比图表和统计摘要
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


def main():
    print("=" * 60)
    print("Comparing Baseline vs Brain-Inspired Coreset")
    print("=" * 60)
    
    baseline_runs = load_results("baseline_result_seed*.json")
    coreset_runs = load_results("coreset_result_seed*.json")
    
    if not baseline_runs:
        print("[Error] No baseline results found in", PROCESSED_DIR)
        return
    if not coreset_runs:
        print("[Error] No coreset results found in", PROCESSED_DIR)
        return
    
    baseline_mses = [r['test_mse'] for r in baseline_runs]
    coreset_mses = [r['test_mse'] for r in coreset_runs]
    
    base_mean = np.mean(baseline_mses)
    base_std = np.std(baseline_mses)
    core_mean = np.mean(coreset_mses)
    core_std = np.std(coreset_mses)
    
    print(f"\n[Baseline] {len(baseline_runs)} runs")
    print(f"  Test MSE: {base_mean:.6f} ± {base_std:.6f}")
    print(f"  Range:    [{np.min(baseline_mses):.6f}, {np.max(baseline_mses):.6f}]")
    
    print(f"\n[Coreset] {len(coreset_runs)} runs")
    print(f"  Test MSE: {core_mean:.6f} ± {core_std:.6f}")
    print(f"  Range:    [{np.min(coreset_mses):.6f}, {np.max(coreset_mses):.6f}]")
    
    improvement = (base_mean - core_mean) / base_mean * 100
    print(f"\n[Improvement] {improvement:+.2f}% (negative means coreset is better)")
    
    # ==================== 绘制对比图 ====================
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # ---- 图1: MSE 柱状图 ----
    ax = axes[0]
    methods = ['Random\nBaseline', 'Brain-Inspired\nCoreset']
    means = [base_mean, core_mean]
    stds = [base_std, core_std]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title('Action Prediction MSE Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s,
                f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=10)
    
    # ---- 图2: 学习曲线 ----
    ax = axes[1]
    if baseline_runs and 'history' in baseline_runs[0]:
        h = baseline_runs[0]['history']
        ax.plot(h['val_loss'], label='Baseline', color='#e74c3c', linewidth=2, marker='o', markersize=3, markevery=10)
    if coreset_runs and 'history' in coreset_runs[0]:
        h = coreset_runs[0]['history']
        ax.plot(h['val_loss'], label='Coreset', color='#2ecc71', linewidth=2, marker='s', markersize=3, markevery=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation MSE', fontsize=11)
    ax.set_title('Training Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    # ---- 图3: 每维 MSE 对比 ----
    ax = axes[2]
    dim_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    base_dim = np.mean([r['dim_mse'] for r in baseline_runs], axis=0)
    core_dim = np.mean([r['dim_mse'] for r in coreset_runs], axis=0)
    x = np.arange(len(dim_names))
    width = 0.35
    ax.bar(x - width/2, base_dim, width, label='Baseline', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, core_dim, width, label='Coreset', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('MSE per Joint', fontsize=11)
    ax.set_title('Per-Joint Prediction Error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, 'comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n[Plot saved] {plot_path}")
    
    # ==================== 保存统计摘要 ====================
    summary = {
        'baseline_avg_mse': float(base_mean),
        'baseline_std_mse': float(base_std),
        'coreset_avg_mse': float(core_mean),
        'coreset_std_mse': float(core_std),
        'improvement_percent': float(improvement),
        'baseline_runs': len(baseline_runs),
        'coreset_runs': len(coreset_runs)
    }
    summary_path = os.path.join(REPORT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary saved] {summary_path}")


if __name__ == "__main__":
    main()
