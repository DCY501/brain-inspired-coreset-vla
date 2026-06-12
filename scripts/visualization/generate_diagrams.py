"""
算法示意图生成脚本 v2 - 精致版
每张图一个独立函数，保存到 PROJECT/report/photo/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PHOTO_DIR = os.path.join(PROJECT_ROOT, 'report', 'photo')
os.makedirs(PHOTO_DIR, exist_ok=True)


def save_fig(name, dpi=300):
    path = os.path.join(PHOTO_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {path}')


def draw_box(ax, x, y, w, h, text, color='#3498db', text_color='white', fontsize=10, alpha=1.0):
    """绘制带圆角的流程框"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=2):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle="arc3,rad=0"))


def draw_dashed_box(ax, x, y, w, h, label, color='#e74c3c', fontsize=11):
    """绘制虚线分组框"""
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color,
                         linewidth=2, linestyle='--', alpha=0.8)
    ax.add_patch(rect)
    # 标签放在框的上方中央
    ax.text(x + w/2, y + h + 0.15, label, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color))


# ==================== 图A: 自适应聚类筛选算法流程 ====================
def plot_figA():
    """自适应聚类筛选算法流程图 - 精致版"""
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_title('图A：自适应聚类筛选算法流程', fontsize=16, fontweight='bold', pad=20)

    # 统一参数
    box_w, box_h = 6.0, 0.75
    center_x = 5.0
    y_positions = [12.0, 10.8, 9.6, 8.2, 7.0, 6.0, 4.8, 3.6]

    # 1. 输入
    draw_box(ax, center_x, y_positions[0], box_w, box_h,
             '输入：768d CLIP 视觉特征\n(16,000 帧训练数据)', '#2c3e50', fontsize=11)
    draw_arrow(ax, center_x, y_positions[0]-box_h/2, center_x, y_positions[1]+box_h/2)

    # 2. K-Means 聚类
    draw_box(ax, center_x, y_positions[1], box_w, box_h,
             'K-Means 聚类 → 533 个簇 (k=533)', '#e74c3c', fontsize=11)
    # 右侧注释
    ax.text(8.6, y_positions[1], '簇数 = 目标核心集大小\n(每簇选 1 个代表)',
            ha='left', va='center', fontsize=9, color='#e74c3c',
            bbox=dict(boxstyle='round', facecolor='#fdedec', alpha=0.9, edgecolor='#e74c3c'))
    ax.annotate('', xy=(center_x+box_w/2+0.1, y_positions[1]),
                xytext=(8.5, y_positions[1]),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    draw_arrow(ax, center_x, y_positions[1]-box_h/2, center_x, y_positions[2]+box_h/2)

    # 3. 计算多样性得分
    draw_box(ax, center_x, y_positions[2], box_w+0.5, box_h,
             '计算多样性得分\nscore = dist_to_center × 1/√cluster_size', '#f39c12', fontsize=10)
    ax.text(8.6, y_positions[2], '距离中心越远\n且簇越小 = 越独特',
            ha='left', va='center', fontsize=9, color='#f39c12',
            bbox=dict(boxstyle='round', facecolor='#fef5e7', alpha=0.9, edgecolor='#f39c12'))
    ax.annotate('', xy=(center_x+box_w/2+0.3, y_positions[2]),
                xytext=(8.5, y_positions[2]),
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))
    draw_arrow(ax, center_x, y_positions[2]-box_h/2, center_x, y_positions[3]+box_h/2+0.15)

    # ===== 动态保底机制（虚线框区域）=====
    # 虚线框：框住"自适应配额" + "小簇/大簇分支" + "汇聚"
    dashed_x, dashed_y, dashed_w, dashed_h = 1.3, 4.9, 7.4, 2.6
    draw_dashed_box(ax, dashed_x, dashed_y, dashed_w, dashed_h,
                    '动态保底机制（核心创新）', '#e74c3c', fontsize=12)

    # 4. 按簇大小自适应配额
    draw_box(ax, center_x, y_positions[3], box_w, box_h,
             '按簇大小自适应配额', '#9b59b6', fontsize=11)
    draw_arrow(ax, center_x, y_positions[3]-box_h/2, center_x, y_positions[4]+box_h/2+0.1)

    # 5. 小簇 / 大簇分支
    branch_y = y_positions[4]
    small_x, large_x = 3.0, 7.0
    draw_box(ax, small_x, branch_y, 2.8, 0.7,
             '小簇 (<10帧)\n保底 1 帧', '#3498db', fontsize=10)
    draw_box(ax, large_x, branch_y, 2.8, 0.7,
             '大簇 (>100帧)\n最多 3 帧', '#3498db', fontsize=10)

    # 右侧注释
    ax.text(8.6, branch_y, '避免大簇浪费名额\n避免小簇被忽略',
            ha='left', va='center', fontsize=9, color='#9b59b6',
            bbox=dict(boxstyle='round', facecolor='#f5eef8', alpha=0.9, edgecolor='#9b59b6'))
    ax.annotate('', xy=(large_x+1.4, branch_y), xytext=(8.5, branch_y),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

    # 汇聚箭头
    merge_y = y_positions[5]
    ax.annotate('', xy=(center_x-0.3, merge_y+box_h/2+0.05),
                xytext=(small_x, branch_y-box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(center_x+0.3, merge_y+box_h/2+0.05),
                xytext=(large_x, branch_y-box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))

    # 6. 全局补齐
    draw_box(ax, center_x, merge_y, box_w, box_h,
             '全局补齐 → 按得分排序\n选取 Top-1,600 帧', '#1abc9c', fontsize=10)
    draw_arrow(ax, center_x, merge_y-box_h/2, center_x, y_positions[6]+box_h/2)

    # 7. 输出
    draw_box(ax, center_x, y_positions[6], box_w, box_h,
             '输出：1,600 帧核心集\n(分布冗余已去除)', '#2ecc71', fontsize=11)

    # 底部核心洞察
    ax.text(center_x, 2.2,
            '核心洞察：簇大小自适应配额\n比固定保底更公平地挖掘数据内部多样性',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    save_fig('figA_kmeans_flowchart.png')


# ==================== 图B: 密度过滤 FPS 算法流程 ====================
def plot_figB():
    """密度过滤 FPS 算法流程图 - 精致版"""
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_title('图B：密度过滤最远点采样算法流程', fontsize=16, fontweight='bold', pad=20)

    box_w, box_h = 6.0, 0.75
    center_x = 5.0
    y_positions = [12.0, 10.8, 9.6, 8.3, 7.1, 6.0, 4.8, 3.5]

    # 1. 输入
    draw_box(ax, center_x, y_positions[0], box_w, box_h,
             '输入：768d CLIP 视觉特征\n(16,000 帧训练数据)', '#2c3e50', fontsize=11)
    draw_arrow(ax, center_x, y_positions[0]-box_h/2, center_x, y_positions[1]+box_h/2)

    # 2. 标准 FPS
    draw_box(ax, center_x, y_positions[1], box_w, box_h,
             '标准 FPS：贪心选最远点 → 1,600 帧候选核心集', '#3498db', fontsize=11)
    ax.text(8.6, y_positions[1], '贪心策略\n最大化空间覆盖',
            ha='left', va='center', fontsize=9, color='#3498db',
            bbox=dict(boxstyle='round', facecolor='#ebf5fb', alpha=0.9, edgecolor='#3498db'))
    ax.annotate('', xy=(center_x+box_w/2+0.1, y_positions[1]),
                xytext=(8.5, y_positions[1]),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))
    draw_arrow(ax, center_x, y_positions[1]-box_h/2, center_x, y_positions[2]+box_h/2)

    # 3. k-NN 密度估计
    draw_box(ax, center_x, y_positions[2], box_w, box_h,
             'k-NN 局部密度估计 (k=6, 欧氏距离)', '#9b59b6', fontsize=11)
    draw_arrow(ax, center_x, y_positions[2]-box_h/2, center_x, y_positions[3]+box_h/2+0.15)

    # 4. 识别孤立点
    draw_box(ax, center_x, y_positions[3], box_w+0.5, box_h,
             '识别孤立点：找出 15% 最稀疏帧 → 240 帧疑似噪声', '#e74c3c', fontsize=10)
    ax.text(8.6, y_positions[3], 'FPS 在特征空间边缘选点\n这些点可能是噪声!',
            ha='left', va='center', fontsize=9, color='#e74c3c',
            bbox=dict(boxstyle='round', facecolor='#fdedec', alpha=0.9, edgecolor='#e74c3c'))
    ax.annotate('', xy=(center_x+box_w/2+0.3, y_positions[3]),
                xytext=(8.5, y_positions[3]),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    draw_arrow(ax, center_x, y_positions[3]-box_h/2, center_x, y_positions[4]+box_h/2+0.1)

    # ===== 密度过滤修复（虚线框区域）=====
    dashed_x, dashed_y, dashed_w, dashed_h = 1.3, 4.7, 7.4, 2.7
    draw_dashed_box(ax, dashed_x, dashed_y, dashed_w, dashed_h,
                    '密度过滤修复（核心创新）', '#e74c3c', fontsize=12)

    # 5. 用全局最密集的典型帧替换
    draw_box(ax, center_x, y_positions[4], box_w, box_h,
             '用全局最密集的典型帧替换', '#1abc9c', fontsize=11)
    draw_arrow(ax, center_x, y_positions[4]-box_h/2, center_x, y_positions[5]+box_h/2+0.1)

    # 6. 替换前后对比
    compare_y = y_positions[5]
    before_x, after_x = 3.0, 7.0
    draw_box(ax, before_x, compare_y, 2.8, 0.7,
             '替换前密度\n0.055 ~ 0.19', '#bdc3c7', text_color='#2c3e50', fontsize=10)
    draw_box(ax, after_x, compare_y, 2.8, 0.7,
             '替换后密度\n0.0026 ~ 0.0041', '#2ecc71', fontsize=10)

    ax.text(8.6, compare_y, '密度降低 20x\n确认替换有效',
            ha='left', va='center', fontsize=9, color='#2ecc71',
            bbox=dict(boxstyle='round', facecolor='#e9f7ef', alpha=0.9, edgecolor='#2ecc71'))
    ax.annotate('', xy=(after_x+1.4, compare_y), xytext=(8.5, compare_y),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    # 汇聚箭头
    merge_y = y_positions[6]
    ax.annotate('', xy=(center_x-0.3, merge_y+box_h/2+0.05),
                xytext=(before_x, compare_y-box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(center_x+0.3, merge_y+box_h/2+0.05),
                xytext=(after_x, compare_y-box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))

    # 7. 输出
    draw_box(ax, center_x, merge_y, box_w, box_h,
             '输出：1,600 帧核心集\n"远且不孤立" = 最优', '#2ecc71', fontsize=11)

    # 底部核心洞察
    ax.text(center_x, 2.2,
            '核心洞察："最远" ≠ "最好"\n但 "远且不孤立" = "最好"',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    save_fig('figB_fps_flowchart.png')


# ==================== 图E: 五种融合架构对比 ====================
def plot_figE():
    """五种融合架构对比图 - 2行3列，最后一格总结"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    axes = axes.flatten()

    titles = ['得分融合', '分层融合', '互补融合', 'Action-aware FPS', '语义-密度联合', '核心结论']
    subtitles = [
        '(多信号加权求和)',
        '(粗筛→精筛漏斗)',
        '(基础集+盲区补充)',
        '(FPS+动作多样性加权)',
        '(密度过滤+语义盲区)',
        ''
    ]
    colors = ['#e74c3c', '#f39c12', '#1abc9c', '#9b59b6', '#3498db', '#2c3e50']

    # 五种架构的数据流
    architectures = [
        # 0: 得分融合
        [
            ('CLIP视觉特征', 8.8, '#2c3e50'),
            ('动作变化率', 7.6, '#3498db'),
            ('重建误差', 6.4, '#3498db'),
            ('线性加权', 5.2, '#3498db'),
            ('统一得分排序', 4.0, '#3498db'),
            ('Top-1600 核心集', 2.5, '#e74c3c'),
        ],
        # 1: 分层融合
        [
            ('全部 16,000 帧', 8.8, '#2c3e50'),
            ('聚类筛选粗筛', 7.4, '#3498db'),
            ('3,200 帧候选池', 6.2, '#bdc3c7'),
            ('FPS 精筛', 5.0, '#3498db'),
            ('1,600 帧核心集', 3.5, '#f39c12'),
        ],
        # 2: 互补融合
        [
            ('密度过滤 FPS', 8.8, '#2c3e50'),
            ('1,600 帧基础集', 7.4, '#bdc3c7'),
            ('检测语义盲区', 6.0, '#3498db'),
            ('聚类筛选补充帧', 4.8, '#bdc3c7'),
            ('合并去重', 3.6, '#3498db'),
            ('1,600 帧核心集', 2.5, '#1abc9c'),
        ],
        # 3: Action-aware FPS
        [
            ('标准 FPS 候选集', 8.8, '#2c3e50'),
            ('跨 episode k-NN', 7.4, '#3498db'),
            ('动作多样性得分', 6.0, '#3498db'),
            ('加权距离排序', 4.8, '#3498db'),
            ('1,600 帧核心集', 3.5, '#9b59b6'),
        ],
        # 4: 语义-密度联合
        [
            ('密度过滤 FPS', 8.8, '#2c3e50'),
            ('1,600 帧基础集', 7.4, '#bdc3c7'),
            ('聚类筛选语义盲区', 6.0, '#3498db'),
            ('加权合并', 4.8, '#3498db'),
            ('1,600 帧核心集', 3.5, '#3498db'),
        ],
    ]

    failure_reasons = [
        '失败：信号不正交\n双重强调噪声\nMSE=0.0054',
        '失败：信息漏斗效应\n粗筛过滤有用帧\nMSE=0.0040',
        '失败：语义盲区≠预测价值\n补充引入噪声\nMSE=0.0044',
        '失败：跨 episode k-NN\n动作多样性失真\nMSE=0.0041',
        '失败：语义稀有≠预测价值\n联合无增益\nMSE=0.0043',
    ]

    # 绘制前5个子图
    for idx in range(5):
        ax = axes[idx]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(f'{titles[idx]}\n{subtitles[idx]}',
                     fontsize=13, fontweight='bold', color=colors[idx], pad=10)

        arch = architectures[idx]
        box_w, box_h = 5.0, 0.7
        center_x = 5.0

        # 绘制流程框
        for i, (text, y, box_color) in enumerate(arch):
            is_input = (i == 0)
            is_output = (i == len(arch) - 1)
            text_c = 'white' if box_color != '#bdc3c7' else '#2c3e50'
            draw_box(ax, center_x, y, box_w, box_h, text, box_color,
                    text_color=text_c, fontsize=10)

            # 向下箭头
            if i < len(arch) - 1:
                next_y = arch[i+1][1]
                ax.annotate('', xy=(center_x, next_y + box_h/2 + 0.05),
                           xytext=(center_x, y - box_h/2 - 0.05),
                           arrowprops=dict(arrowstyle='->', color=colors[idx], lw=2))

        # 失败原因标注(底部)
        ax.text(center_x, 0.8, failure_reasons[idx], ha='center', va='bottom',
                fontsize=9, color=colors[idx], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fdedec' if idx==0 else '#fef5e7' if idx==1 else '#e9f7ef' if idx==2 else '#f5eef8' if idx==3 else '#ebf5fb',
                         alpha=0.9, edgecolor=colors[idx], linewidth=2))

    # 第6格：核心结论
    ax = axes[5]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('核心结论', fontsize=14, fontweight='bold', color='#2c3e50', pad=15)

    conclusion_text = (
        '14 个融合实验\n'
        '5 种融合思路\n'
        '→ 无一超越单一最优\n\n'
        '最优融合 MSE = 0.0040\n'
        '单一最优 MSE = 0.0038\n'
        '→ 差距 5.6%\n\n'
        '信息漏斗效应：\n'
        '分层结构无法恢复\n'
        '粗筛过滤的信息\n\n'
        '信息相关性上限：\n'
        '即使发现语义盲区\n'
        '若与预测任务无关\n'
        '补充也是无效的'
    )
    ax.text(5, 5.5, conclusion_text, ha='center', va='center',
            fontsize=12, color='#2c3e50', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                     alpha=0.95, edgecolor='#2c3e50', linewidth=2))

    # 添加红色大叉号在结论上方
    ax.text(5, 8.5, '✗', ha='center', va='center',
            fontsize=80, color='#e74c3c', alpha=0.3, fontweight='bold')

    fig.suptitle('图E：五种融合架构对比（均无法超越单一最优 0.003792）',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('figE_fusion_architecture.png')


# ==================== 主函数 ====================
def main():
    print('=' * 60)
    print('生成算法示意图到 report/photo/')
    print('=' * 60)

    diagrams = [
        ('图A: 自适应聚类筛选流程', plot_figA),
        ('图B: 密度过滤 FPS 流程', plot_figB),
        ('图E: 五种融合架构对比', plot_figE),
    ]

    for i, (name, func) in enumerate(diagrams, 1):
        print(f'\n[{i}/{len(diagrams)}] {name}...')
        try:
            func()
            print('  [OK] 成功')
        except Exception as e:
            print(f'  [FAIL] 失败: {e}')

    print('\n' + '=' * 60)
    print('所有示意图已保存到:', PHOTO_DIR)
    print('=' * 60)


if __name__ == '__main__':
    main()
