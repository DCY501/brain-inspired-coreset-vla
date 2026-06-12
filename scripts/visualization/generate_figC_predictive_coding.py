"""
生成 figC: 预测编码滑动窗口流程图（扁平版）
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(-0.8, 8.5)
ax.axis('off')

# 节点定义: (x_center, y_bottom, width, height, text, facecolor, edgecolor, text_color)
# 使用更小的节点高度和更紧凑的间距
nodes = [
    (6, 7.2, 6.0, 0.55, '输入: 训练集动作序列 a_1, a_2, ..., a_T', '#e8f4f8', '#3498db', '#333'),
    (6, 6.1, 3.2, 0.50, '按 episode 分组', '#f5eef8', '#9b59b6', '#333'),
    (6, 5.0, 4.5, 0.50, '逐帧滑动窗口 (window=w)', '#eafaf1', '#2ecc71', '#333'),
    (6, 3.9, 7.0, 0.55, '最小二乘线性回归预测 + 计算残差: r_t = ||a_t - a_hat_t||_2', '#fef9e7', '#f39c12', '#333'),
    (6, 2.7, 4.8, 0.75, '该 episode 所有帧残差计算完毕?', '#eafaf1', '#2ecc71', '#333'),
    (6, 1.5, 6.5, 0.55, '全局按残差排序: 残差大 = 难预测 = 信息丰富 = 保留', '#eafaf1', '#2ecc71', '#333'),
    (6, 0.4, 4.5, 0.50, '取 Top-K 核心集 (K = 20% x N_train)', '#eafaf1', '#2ecc71', '#333'),
    (6, -0.7, 3.0, 0.50, '输出: 核心集索引', '#d5f5e3', '#2ecc71', '#333'),
]

node_patches = []
for x, y, w, h, text, fc, ec, tc in nodes:
    # 判断是否为菱形（决策节点）
    if '?' in text:
        half_w = w / 2
        half_h = h / 2
        diamond = plt.Polygon([
            [x, y + half_h],
            [x + half_w, y + half_h * 0.25],
            [x, y - half_h * 0.25],
            [x - half_w, y + half_h * 0.25],
        ], closed=True, facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y + half_h * 0.05, text, ha='center', va='center', fontsize=11,
                color=tc, fontweight='bold', linespacing=1.2)
    else:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.03,rounding_size=0.12",
                             facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
                color=tc, fontweight='bold')
    node_patches.append((x, y, w, h, '?' in text))

# 画箭头
for i in range(len(node_patches) - 1):
    x1, y1, w1, h1, is_diamond1 = node_patches[i]
    x2, y2, w2, h2, is_diamond2 = node_patches[i+1]
    
    if is_diamond1:
        start_y = y1 - h1 * 0.25
    else:
        start_y = y1 - h1/2
    
    if is_diamond2:
        end_y = y2 + h2 * 0.25
    else:
        end_y = y2 + h2/2
    
    arrow = FancyArrowPatch((x1, start_y - 0.03), (x2, end_y + 0.03),
                           arrowstyle='->', mutation_scale=18, linewidth=1.5,
                           color='#666')
    ax.add_patch(arrow)

# 循环箭头: 从决策节点(D)回到滑动窗口(C)
x_d, y_d, w_d, h_d, _ = node_patches[4]
x_c, y_c, w_c, h_c, _ = node_patches[2]

loop_arrow = FancyArrowPatch(
    (x_d + w_d/2 * 0.95, y_d + 0.1),
    (x_c + w_c/2 + 0.5, y_c + 0.1),
    arrowstyle='->', mutation_scale=18, linewidth=1.5,
    color='#666', connectionstyle="arc3,rad=0.28"
)
ax.add_patch(loop_arrow)
ax.text(x_c + w_c/2 + 0.65, (y_d + y_c)/2 + 0.35, '否', fontsize=10, color='#666', fontweight='bold')

# "是"标签
ax.text(x_d + 0.12, y_d - h_d * 0.25 - 0.18, '是', fontsize=10, color='#666', fontweight='bold')

ax.set_title('预测编码滑动窗口: 核心集选择流程', fontsize=14, fontweight='bold', y=0.98)

out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'report', 'photo', 'figC_predictive_coding.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'[Saved] {out_path}')
