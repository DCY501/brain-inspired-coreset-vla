"""
生成图11 v2: 五种冗余定义 + 随机采样 的 t-SNE 可视化对比 (2x3)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE

# 强制刷新字体缓存以确保能找到中文字体
fm.fontManager.__init__()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FEATURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'features')
CORESET_RATIO = 0.2

def load_json(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    visual_features = np.load(os.path.join(FEATURES_DIR, 'clip_visual_features.npy'))
    ep_indices = np.load(os.path.join(FEATURES_DIR, 'episode_indices.npy'))
    
    # 固定测试集划分（与评估一致）
    unique_eps = np.unique(ep_indices)
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_eps)
    n_test = max(1, int(len(unique_eps) * 0.2))
    test_eps = shuffled[:n_test]
    train_mask = ~np.isin(ep_indices, test_eps)
    X_train = visual_features[train_mask]
    train_idx_global = np.where(train_mask)[0]
    n_train = len(X_train)
    target_size = int(n_train * CORESET_RATIO)
    
    # 五种方法 + 随机采样
    methods = [
        ('密度过滤 FPS\n(MSE=0.003792)', 
         'approaches/farthest_point_sampling/results/density_filtered_fps_result_seed42.json',
         '#2ecc71', '最优'),
        ('自适应聚类 + PCA\n(MSE=0.003835)',
         'approaches/redundancy_distribution/results/pca_sweep/coreset_pca400_result_seed42.json',
         '#3498db', '成功'),
        ('预测编码滑动窗口\n(MSE=0.005688)',
         'approaches/predictive_coding_sliding_window/results/predictive_coding_result_seed42.json',
         '#9b59b6', '有效但有限'),
        ('时序事件边界\n(MSE=0.008871)',
         'approaches/temporal_awareness/results/temporal_result_seed42.json',
         '#e74c3c', '失败'),
        ('AE 重建误差\n(MSE=0.007292)',
         'approaches/autoencoder_reconstruction/results/ae_recon_result_seed42.json',
         '#f39c12', '失败'),
    ]
    
    # 收集所有核心集索引
    coresets = []
    for name, path, color, status in methods:
        d = load_json(path)
        idx = np.array(d['coreset_indices'])
        coresets.append((name, idx, color, status))
    
    # 随机采样核心集
    np.random.seed(42)
    random_idx = np.random.choice(n_train, target_size, replace=False)
    random_idx_global = train_idx_global[random_idx]
    coresets.append(('随机采样\n(MSE=0.007229)', random_idx_global, '#2c3e50', '基线'))
    
    # 统一背景：所有训练帧
    all_train_idx = train_idx_global
    
    # t-SNE 降维（为效率，对大规模数据做均匀采样）
    if n_train > 8000:
        step = n_train // 4000 + 1
        tsne_sample_mask = np.arange(0, n_train, step)
    else:
        tsne_sample_mask = np.arange(n_train)
    
    X_tsne_input = X_train[tsne_sample_mask]
    print(f'[t-SNE] Fitting on {len(X_tsne_input)} samples...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne_input)-1), max_iter=1000)
    X_2d_full = tsne.fit_transform(X_tsne_input)
    
    # 建立全局索引到 t-SNE 坐标的映射
    tsne_global_idx = train_idx_global[tsne_sample_mask]
    idx_to_tsne = {g: i for i, g in enumerate(tsne_global_idx)}
    
    # 创建 2x3 子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, idx, color, status) in enumerate(coresets):
        ax = axes[i]
        
        # 映射核心集索引到 t-SNE 坐标
        valid_idx = [ig for ig in idx if ig in idx_to_tsne]
        sel_tsne_idx = [idx_to_tsne[ig] for ig in valid_idx]
        X_sel = X_2d_full[sel_tsne_idx]
        
        # 映射背景索引到 t-SNE 坐标（排除核心集）
        bg_global_idx = [ig for ig in all_train_idx if ig not in set(valid_idx) and ig in idx_to_tsne]
        bg_tsne_idx = [idx_to_tsne[ig] for ig in bg_global_idx]
        X_bg = X_2d_full[bg_tsne_idx]
        
        # 画背景
        ax.scatter(X_bg[:, 0], X_bg[:, 1], c='#cccccc', s=2, alpha=0.3, label='未选中')
        
        # 画核心集
        ax.scatter(X_sel[:, 0], X_sel[:, 1], c=color, s=20, alpha=0.85, 
                   edgecolors='white', linewidths=0.4, label='核心集', zorder=5)
        
        # 标题
        status_symbol = {'最优': '★', '成功': '√', '有效但有限': '△', '失败': '×', '基线': '○'}
        sym = status_symbol.get(status, '')
        ax.set_title(f'{sym} {name}', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('t-SNE 维度 1', fontsize=9)
        ax.set_ylabel('t-SNE 维度 2', fontsize=9)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    
    fig.suptitle('五种冗余定义方法与随机采样的核心集分布对比', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'report', 'photo', 'fig11_redundancy_comparison.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'[Saved] {out_path}')


if __name__ == '__main__':
    main()
