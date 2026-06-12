"""
可视化分析脚本 - 生成所有实验图表
保存到 PROJECT/report/ 目录
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
os.makedirs(REPORT_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(name, dpi=300):
    path = os.path.join(REPORT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {path}')


# ==================== 图1: 全局方法对比柱状图 ====================
def plot_method_comparison():
    """所有核心集选择方法的 Test MSE 对比"""
    methods = [
        ('密度过滤 FPS', 'approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json', '#2ecc71'),
        ('v1.1 + PCA400d', 'approaches/redundancy_distribution/results/v1_1_dynamic/coreset_result_seed42.json', '#3498db'),
        ('加权FPS(feature_norm)', 'approaches/farthest_point_sampling/results/weighted_fps_feature_norm_result_seed42.json', '#9b59b6'),
        ('标准 FPS', 'approaches/farthest_point_sampling/results/fps_result_seed42.json', '#e74c3c'),
        ('分层融合 cr2.0', 'approaches/fusion_score_weighted/results/hierarchical_cr2.0_result_seed42.json', '#f39c12'),
        ('互补融合 v3.0', 'approaches/fusion_score_weighted/results/complementary_ct0.05_result_seed42.json', '#1abc9c'),
        ('分层融合 cr3.0', 'approaches/fusion_score_weighted/results/hierarchical_cr3.0_result_seed42.json', '#e67e22'),
        ('时序事件边界', 'approaches/temporal_awareness/results/temporal_result_seed42.json', '#95a5a6'),
        ('AE 重建误差', 'approaches/autoencoder_reconstruction/results/ae_recon_result_seed42.json', '#34495e'),
    ]
    
    # 随机基线
    baseline_mses = []
    for seed in [42, 43, 44, 45, 46]:
        d = load_json(f'data/processed/baseline_result_seed{seed}.json')
        baseline_mses.append(d['test_mse'])
    baseline_avg = np.mean(baseline_mses)
    baseline_std = np.std(baseline_mses)
    
    names, mses, colors = [], [], []
    for name, path, color in methods:
        d = load_json(path)
        names.append(name)
        mses.append(d['test_mse'])
        colors.append(color)
    
    # 添加随机基线
    names.append('随机基线 (avg)')
    mses.append(baseline_avg)
    colors.append('#7f8c8d')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), mses, color=colors, edgecolor='white', height=0.7)
    
    # 标注数值
    for i, (bar, mse) in enumerate(zip(bars, mses)):
        if i == len(names) - 1:  # 随机基线
            ax.text(mse + 0.0001, i, f'{mse:.4f}±{baseline_std:.4f}', 
                   va='center', fontsize=9, color='#7f8c8d')
        else:
            ax.text(mse + 0.0001, i, f'{mse:.4f}', 
                   va='center', fontsize=9, color='black')
    
    # 标注最优
    best_idx = np.argmin(mses[:-1])
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    ax.text(mses[best_idx] / 2, best_idx, '★ 最优', ha='center', va='center',
           fontsize=11, color='white', fontweight='bold')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Test MSE', fontsize=12)
    ax.set_title('核心集选择方法性能对比 (Test MSE, 越低越好)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(mses) * 1.15)
    ax.axvline(baseline_avg, color='#7f8c8d', linestyle='--', alpha=0.7, label='随机基线')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    save_fig('fig1_method_comparison.png')


# ==================== 图2: filter_ratio 消融 U 型曲线 ====================
def plot_filter_ratio_ablation():
    """密度过滤 FPS 的 filter_ratio 消融实验"""
    ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    mses = []
    maes = []
    
    for fr in ratios:
        if fr == 0.15:
            d = load_json('approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json')
        else:
            d = load_json(f'approaches/farthest_point_sampling/results/density_filtered_fps_fr{fr}_result_seed42.json')
        mses.append(d['test_mse'])
        maes.append(d['test_mae'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE 曲线
    ax1.plot(ratios, mses, 'o-', color='#e74c3c', linewidth=2, markersize=10)
    best_idx = np.argmin(mses)
    ax1.plot(ratios[best_idx], mses[best_idx], '*', color='gold', markersize=20, 
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax1.annotate(f'最优: fr={ratios[best_idx]}\nMSE={mses[best_idx]:.4f}', 
                xy=(ratios[best_idx], mses[best_idx]),
                xytext=(ratios[best_idx] + 0.03, mses[best_idx] + 0.0003),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('filter_ratio', fontsize=12)
    ax1.set_ylabel('Test MSE', fontsize=12)
    ax1.set_title('filter_ratio 消融实验 (MSE)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(ratios)
    
    # MAE 曲线
    ax2.plot(ratios, maes, 's-', color='#3498db', linewidth=2, markersize=10)
    best_idx_mae = np.argmin(maes)
    ax2.plot(ratios[best_idx_mae], maes[best_idx_mae], '*', color='gold', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax2.annotate(f'最优: fr={ratios[best_idx_mae]}\nMAE={maes[best_idx_mae]:.4f}',
                xy=(ratios[best_idx_mae], maes[best_idx_mae]),
                xytext=(ratios[best_idx_mae] + 0.03, maes[best_idx_mae] + 0.001),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('filter_ratio', fontsize=12)
    ax2.set_ylabel('Test MAE', fontsize=12)
    ax2.set_title('filter_ratio 消融实验 (MAE)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(ratios)
    
    plt.tight_layout()
    save_fig('fig2_filter_ratio_ablation.png')


# ==================== 图3: t-SNE 核心集分布 ====================
def plot_tsne_coreset():
    """t-SNE 可视化：密度过滤 FPS 核心集 vs 全部训练数据"""
    from sklearn.manifold import TSNE
    
    # 加载特征和核心集索引
    visual_features = np.load(os.path.join(FEATURES_DIR, 'clip_visual_features.npy'))
    ep_indices = np.load(os.path.join(FEATURES_DIR, 'episode_indices.npy'))
    
    # 训练集 mask
    unique_eps = np.unique(ep_indices)
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_eps)
    n_test = max(1, int(len(unique_eps) * 0.2))
    test_eps = shuffled[:n_test]
    train_mask = ~np.isin(ep_indices, test_eps)
    
    X_train = visual_features[train_mask]
    
    # 核心集索引
    d = load_json('approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json')
    coreset_idx = np.array(d['coreset_indices'])
    
    # 子采样加速 t-SNE（全部 16000 帧太慢，采样 2000 帧）
    np.random.seed(42)
    sample_size = 2000
    all_sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    # 确保核心集样本被包含
    coreset_in_sample = np.isin(coreset_idx, all_sample_idx)
    if not coreset_in_sample.all():
        # 替换一些样本为核心集样本
        n_needed = (~coreset_in_sample).sum()
        replace_idx = np.random.choice(all_sample_idx, n_needed, replace=False)
        all_sample_idx = np.setdiff1d(all_sample_idx, replace_idx)
        all_sample_idx = np.concatenate([all_sample_idx, coreset_idx[~coreset_in_sample]])
    
    X_sample = X_train[all_sample_idx]
    
    print(f'[t-SNE] Fitting on {len(X_sample)} samples...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_sample)
    
    # 判断哪些是核心集
    is_coreset = np.isin(all_sample_idx, coreset_idx)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 全部数据（灰色，半透明）
    ax.scatter(X_2d[~is_coreset, 0], X_2d[~is_coreset, 1], 
              c='#bdc3c7', s=10, alpha=0.3, label=f'全部训练数据 (n={(~is_coreset).sum()})')
    
    # 核心集（绿色，明显）
    ax.scatter(X_2d[is_coreset, 0], X_2d[is_coreset, 1],
              c='#2ecc71', s=40, alpha=0.8, edgecolors='darkgreen', linewidth=0.5,
              label=f'密度过滤 FPS 核心集 (n={is_coreset.sum()})')
    
    ax.set_title('t-SNE 可视化：核心集在特征空间中的分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlabel('t-SNE 维度 1', fontsize=11)
    ax.set_ylabel('t-SNE 维度 2', fontsize=11)
    ax.grid(alpha=0.2)
    
    save_fig('fig3_tsne_coreset.png')


# ==================== 图4: PCA 核心集分布 ====================
def plot_pca_coreset():
    """PCA 可视化：密度过滤 FPS 核心集 vs 全部训练数据"""
    from sklearn.decomposition import PCA
    
    visual_features = np.load(os.path.join(FEATURES_DIR, 'clip_visual_features.npy'))
    ep_indices = np.load(os.path.join(FEATURES_DIR, 'episode_indices.npy'))
    
    unique_eps = np.unique(ep_indices)
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_eps)
    n_test = max(1, int(len(unique_eps) * 0.2))
    test_eps = shuffled[:n_test]
    train_mask = ~np.isin(ep_indices, test_eps)
    
    X_train = visual_features[train_mask]
    
    d = load_json('approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json')
    coreset_idx = np.array(d['coreset_indices'])
    
    # PCA
    print(f'[PCA] Fitting on {len(X_train)} samples...')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：2D 散点
    ax1.scatter(X_pca[~np.isin(np.arange(len(X_pca)), coreset_idx), 0],
               X_pca[~np.isin(np.arange(len(X_pca)), coreset_idx), 1],
               c='#bdc3c7', s=5, alpha=0.3, label='全部训练数据')
    ax1.scatter(X_pca[coreset_idx, 0], X_pca[coreset_idx, 1],
               c='#2ecc71', s=30, alpha=0.7, edgecolors='darkgreen', linewidth=0.5,
               label='密度过滤 FPS 核心集')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax1.set_title('PCA 2D：核心集分布', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    
    # 右图：方差解释率
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax2.plot(range(1, 51), cumvar[:50], 'o-', color='#3498db', linewidth=2, markersize=4)
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90% 方差')
    ax2.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='95% 方差')
    
    # 标注需要多少维度
    dim_90 = np.argmax(cumvar >= 0.9) + 1
    dim_95 = np.argmax(cumvar >= 0.95) + 1
    ax2.annotate(f'{dim_90} 维 → 90%', xy=(dim_90, 0.9), xytext=(dim_90+5, 0.85),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate(f'{dim_95} 维 → 95%', xy=(dim_95, 0.95), xytext=(dim_95+5, 0.90),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='orange'))
    
    ax2.set_xlabel('主成分数量', fontsize=11)
    ax2.set_ylabel('累计方差解释率', fontsize=11)
    ax2.set_title('PCA 累计方差解释率', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    save_fig('fig4_pca_coreset.png')


# ==================== 图5: 学习曲线对比 ====================
def plot_learning_curves():
    """密度过滤 FPS vs 随机基线的学习曲线"""
    # 密度过滤 FPS
    d_fps = load_json('approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json')
    hist_fps = d_fps['history']
    
    # 随机基线（取 seed42）
    d_rand = load_json('data/processed/baseline_result_seed42.json')
    hist_rand = d_rand['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_fps = range(1, len(hist_fps['train_loss']) + 1)
    epochs_rand = range(1, len(hist_rand['train_loss']) + 1)
    
    # 训练 loss
    ax1.plot(epochs_fps, hist_fps['train_loss'], '-', color='#2ecc71', linewidth=2, label='密度过滤 FPS (train)')
    ax1.plot(epochs_fps, hist_fps['val_loss'], '--', color='#2ecc71', linewidth=2, alpha=0.7, label='密度过滤 FPS (val)')
    ax1.plot(epochs_rand, hist_rand['train_loss'], '-', color='#7f8c8d', linewidth=2, label='随机基线 (train)')
    ax1.plot(epochs_rand, hist_rand['val_loss'], '--', color='#7f8c8d', linewidth=2, alpha=0.7, label='随机基线 (val)')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('学习曲线对比 (Train / Val)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # 最终测试 MSE 对比（带误差条）
    baseline_mses = []
    for seed in [42, 43, 44, 45, 46]:
        d = load_json(f'data/processed/baseline_result_seed{seed}.json')
        baseline_mses.append(d['test_mse'])
    
    methods = ['密度过滤 FPS', 'v1.1+PCA400', '标准 FPS', '随机基线']
    mses = [
        d_fps['test_mse'],
        load_json('approaches/redundancy_distribution/results/v1_1_dynamic/coreset_result_seed42.json')['test_mse'],
        load_json('approaches/farthest_point_sampling/results/fps_result_seed42.json')['test_mse'],
        np.mean(baseline_mses)
    ]
    errors = [0, 0, 0, np.std(baseline_mses)]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#7f8c8d']
    
    bars = ax2.bar(methods, mses, yerr=errors, color=colors, edgecolor='white', 
                   capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax2.set_ylabel('Test MSE', fontsize=11)
    ax2.set_title('最终测试性能对比', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for bar, mse in zip(bars, mses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig5_learning_curves.png')


# ==================== 图6: 7DoF 每关节误差 ====================
def plot_dof_error():
    """7 个关节各自的 MSE 对比"""
    methods_data = [
        ('密度过滤 FPS', 'approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json', '#2ecc71'),
        ('v1.1+PCA400', 'approaches/redundancy_distribution/results/v1_1_dynamic/coreset_result_seed42.json', '#3498db'),
        ('标准 FPS', 'approaches/farthest_point_sampling/results/fps_result_seed42.json', '#e74c3c'),
    ]
    
    # 随机基线平均
    dim_mses_rand = []
    for seed in [42, 43, 44, 45, 46]:
        d = load_json(f'data/processed/baseline_result_seed{seed}.json')
        dim_mses_rand.append(d['dim_mse'])
    dim_mse_rand = np.mean(dim_mses_rand, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(7)
    width = 0.2
    
    for i, (name, path, color) in enumerate(methods_data):
        d = load_json(path)
        dim_mse = d['dim_mse']
        ax.bar(x + i*width, dim_mse, width, label=name, color=color, edgecolor='white')
    
    ax.bar(x + 3*width, dim_mse_rand, width, label='随机基线 (avg)', color='#7f8c8d', edgecolor='white')
    
    ax.set_xlabel('关节维度', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('7DoF 每关节预测误差对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([f'Dim {i}' for i in range(7)], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 标注最难预测的维度
    all_dim_mses = [load_json(path)['dim_mse'] for _, path, _ in methods_data]
    all_dim_mses.append(dim_mse_rand)
    avg_by_dim = np.mean(all_dim_mses, axis=0)
    hardest_dim = np.argmax(avg_by_dim)
    ax.annotate(f'最难预测\nDim {hardest_dim}', 
                xy=(hardest_dim + 1.5*width, avg_by_dim[hardest_dim]),
                xytext=(hardest_dim + 1.5*width + 0.5, avg_by_dim[hardest_dim] + 0.002),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')
    
    plt.tight_layout()
    save_fig('fig6_dof_error.png')


# ==================== 图7: 密度替换效果散点图 ====================
def plot_density_replacement():
    """展示密度过滤中被替换的稀疏点 vs 替换进来的密集点"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    
    visual_features = np.load(os.path.join(FEATURES_DIR, 'clip_visual_features.npy'))
    ep_indices = np.load(os.path.join(FEATURES_DIR, 'episode_indices.npy'))
    
    unique_eps = np.unique(ep_indices)
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_eps)
    n_test = max(1, int(len(unique_eps) * 0.2))
    test_eps = shuffled[:n_test]
    train_mask = ~np.isin(ep_indices, test_eps)
    
    X_train = visual_features[train_mask]
    
    # 读取核心集索引
    d = load_json('approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json')
    coreset_idx = np.array(d['coreset_indices'])
    
    # 用 PCA 做 2D 投影
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    
    # 计算局部密度（k=5）
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(X_train)
    distances, _ = knn.kneighbors(X_train)
    densities = np.mean(distances[:, 1:], axis=1)  # 排除自身
    
    # 核心集密度
    coreset_densities = densities[coreset_idx]
    
    # 找出最稀疏的 15% 和最密集的 15%
    n_replace = int(len(coreset_idx) * 0.15)
    sparsest_in_coreset = coreset_idx[np.argsort(coreset_densities)[-n_replace:]]
    
    # 未选中的帧
    all_idx = np.arange(len(X_train))
    unselected = np.setdiff1d(all_idx, coreset_idx)
    unselected_densities = densities[unselected]
    densest_unselected = unselected[np.argsort(unselected_densities)[:n_replace]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 全部数据（淡灰色）
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='#ecf0f1', s=5, alpha=0.3, label='全部训练数据')
    
    # 核心集（绿色）
    ax.scatter(X_pca[coreset_idx, 0], X_pca[coreset_idx, 1], 
              c='#2ecc71', s=20, alpha=0.5, label='核心集 (1600帧)')
    
    # 被替换的稀疏点（红色叉）
    ax.scatter(X_pca[sparsest_in_coreset, 0], X_pca[sparsest_in_coreset, 1],
              c='red', s=80, marker='x', linewidth=2, label=f'被替换的稀疏点 (n={len(sparsest_in_coreset)})')
    
    # 替换进来的密集点（蓝色星）
    ax.scatter(X_pca[densest_unselected, 0], X_pca[densest_unselected, 1],
              c='blue', s=80, marker='*', label=f'替换进来的密集点 (n={len(densest_unselected)})')
    
    ax.set_title('密度过滤效果：稀疏噪声点被替换为典型密集点', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.2)
    
    save_fig('fig7_density_replacement.png')


# ==================== 图8: Episode 覆盖热力图 ====================
def plot_episode_coverage():
    """各方法对每个 episode 的覆盖情况"""
    ep_indices = np.load(os.path.join(FEATURES_DIR, 'episode_indices.npy'))
    
    unique_eps = np.unique(ep_indices)
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_eps)
    n_test = max(1, int(len(unique_eps) * 0.2))
    test_eps = set(shuffled[:n_test])
    train_eps = shuffled[n_test:]
    
    methods = [
        ('密度过滤 FPS', 'approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json'),
        ('v1.1+PCA400', 'approaches/redundancy_distribution/results/v1_1_dynamic/coreset_result_seed42.json'),
        ('标准 FPS', 'approaches/farthest_point_sampling/results/fps_result_seed42.json'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, path) in zip(axes, methods):
        d = load_json(path)
        coreset_idx = np.array(d['coreset_indices'])
        
        # 计算每个 episode 被选中的帧数
        train_mask = ~np.isin(ep_indices, list(test_eps))
        ep_train = ep_indices[train_mask]
        
        coverage_counts = []
        for ep in train_eps:
            total_in_ep = (ep_train == ep).sum()
            selected_in_ep = (ep_train[coreset_idx] == ep).sum() if len(coreset_idx) > 0 else 0
            coverage_counts.append(selected_in_ep / total_in_ep if total_in_ep > 0 else 0)
        
        colors = ['#e74c3c' if c < 0.05 else '#f39c12' if c < 0.08 else '#2ecc71' for c in coverage_counts]
        bars = ax.bar(range(len(train_eps)), coverage_counts, color=colors, edgecolor='white')
        ax.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='10% 期望比例')
        ax.set_xlabel('Episode (训练集)', fontsize=10)
        ax.set_ylabel('选中比例', fontsize=10)
        ax.set_title(f'{name}\n覆盖率分布', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(coverage_counts) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # 标注覆盖率统计
        mean_cov = np.mean(coverage_counts)
        std_cov = np.std(coverage_counts)
        ax.text(0.5, 0.95, f'均值: {mean_cov:.3f}\n标准差: {std_cov:.3f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_fig('fig8_episode_coverage.png')


# ==================== 图9: PCA 维度消融曲线 ====================
def plot_pca_sweep():
    """v1.1 在不同 PCA 维度下的性能"""
    dims = [50, 100, 200, 300, 400, 500, 600]
    mses = []
    
    for dim in dims:
        d = load_json(f'approaches/redundancy_distribution/results/pca_sweep/coreset_pca{dim}_result_seed42.json')
        mses.append(d['test_mse'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(dims, mses, 'o-', color='#3498db', linewidth=2, markersize=10)
    best_idx = np.argmin(mses)
    ax.plot(dims[best_idx], mses[best_idx], '*', color='gold', markersize=20,
           markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.annotate(f'最优: PCA={dims[best_idx]}\nMSE={mses[best_idx]:.4f}',
               xy=(dims[best_idx], mses[best_idx]),
               xytext=(dims[best_idx] + 50, mses[best_idx] + 0.0003),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('PCA 维度', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('v1.1 K-Means 的 PCA 维度消融实验', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(dims)
    
    # 添加说明文字
    ax.text(0.02, 0.98, 'PCA 维度过低 → 信息损失\nPCA 维度过高 → 噪声干扰',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    save_fig('fig9_pca_sweep.png')


# ==================== 图10: 融合方法失败分析 ====================
def plot_fusion_failure():
    """展示分层融合 vs 单一最优的对比
    同一思路下调参只放最优结果，不同思路才分别展示
    """
    methods = [
        # 单一最优（基准线）
        ('密度过滤 FPS\n(单一最优)', 0.003792, '#2ecc71', 'single'),
        
        # 融合思路（各思路最优代表）
        ('分层融合 v2.0\n(v1.1粗筛+FPS精筛)', 0.004004, '#f39c12', 'fusion'),
        ('Action-aware FPS\n(动作多样性加权)', 0.004040, '#9b59b6', 'fusion'),
        ('语义-密度联合替换\n(α=1.0,β=0.0)', 0.004079, '#1abc9c', 'fusion'),
        ('互补融合 v3.0\n(语义盲区补充)', 0.004429, '#e67e22', 'fusion'),
        ('得分融合 v1.0\n(线性加权α=0.4,β=0.3,γ=0.3)', 0.005434, '#e74c3c', 'fusion'),
    ]
    
    names = [m[0] for m in methods]
    mses = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    categories = [m[3] for m in methods]
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    bars = ax.barh(range(len(names)), mses, color=colors, edgecolor='white', height=0.55)
    
    # 标注数值
    for i, (bar, mse) in enumerate(zip(bars, mses)):
        ax.text(mse + 0.00015, i, f'{mse:.4f}', va='center', fontsize=9)
    
    # 在单一最优和融合方法之间画分隔线
    split_pos = 0.5  # 1个单一方法，5个融合方法，分隔线在0.5
    ax.axhline(y=split_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(0.0058, 0.15, '← 单一最优（基准）', fontsize=10, color='gray', fontstyle='italic')
    ax.text(0.0058, 0.85, '融合/组合思路 →', fontsize=10, color='gray', fontstyle='italic')
    
    # 标注关键洞察
    ax.annotate('信息漏斗:\nv1.1粗筛过滤掉\n部分有用帧，FPS无法恢复',
               xy=(0.004004, 3), xytext=(0.0046, 4.2),
               fontsize=8, color='#f39c12',
               arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    ax.annotate('跨episode k-NN\n动作多样性失真',
               xy=(0.004040, 4), xytext=(0.0046, 3.3),
               fontsize=8, color='#9b59b6',
               arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5eef8', alpha=0.7))
    
    ax.annotate('语义稀有 ≠\n动作预测价值',
               xy=(0.004429, 6), xytext=(0.0050, 5.5),
               fontsize=8, color='#e67e22',
               arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef5e7', alpha=0.7))
    
    ax.annotate('双重强调\n离群噪声',
               xy=(0.005434, 7), xytext=(0.0058, 6.5),
               fontsize=8, color='#e74c3c',
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', alpha=0.7))
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Test MSE', fontsize=12)
    ax.set_title('融合方法失败分析：5 种融合思路均无法超越单一最优', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, max(mses) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加结论框
    ax.text(0.98, 0.02, 
           '核心结论:\n• 14 个融合实验（5 种思路）\n• 最优融合 0.004004 仍落后\n  单一最优 0.003792 约 5.6%\n• 信息完整性无法通过\n  分层结构复现',
           transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    save_fig('fig10_fusion_failure.png')


# ==================== 主函数 ====================
def main():
    print('=' * 60)
    print('生成可视化分析图表')
    print('=' * 60)
    
    print('\n[1/10] 方法对比柱状图...')
    plot_method_comparison()
    
    print('\n[2/10] filter_ratio 消融曲线...')
    plot_filter_ratio_ablation()
    
    print('\n[3/10] t-SNE 核心集分布...')
    plot_tsne_coreset()
    
    print('\n[4/10] PCA 核心集分布...')
    plot_pca_coreset()
    
    print('\n[5/10] 学习曲线对比...')
    plot_learning_curves()
    
    print('\n[6/10] 7DoF 每关节误差...')
    plot_dof_error()
    
    print('\n[7/10] 密度替换效果散点图...')
    plot_density_replacement()
    
    print('\n[8/10] Episode 覆盖热力图...')
    plot_episode_coverage()
    
    print('\n[9/10] PCA 维度消融曲线...')
    plot_pca_sweep()
    
    print('\n[10/10] 融合方法失败分析...')
    plot_fusion_failure()
    
    print('\n' + '=' * 60)
    print('所有图表已保存到:', REPORT_DIR)
    print('=' * 60)


if __name__ == '__main__':
    main()
