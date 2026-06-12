# 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

本项目探索如何在视觉-语言-动作（VLA）模仿学习任务中，通过脑启发核心集选择算法从海量训练帧中筛选最具信息价值的高质量子集，以少量数据训练出高性能的轻量级动作预测模型。

核心思想是从"冗余如何定义"出发，将大脑的预测编码、RAS 网状激活、事件分割、意外检测和海马体空间编码五种信息筛选机制映射为核心集选择算法，系统验证不同冗余定义与数据特性的匹配度。

---

## 核心贡献

- **系统性的冗余定义探索**：独立设计并验证了分布冗余、时序事件分割、预测编码、压缩冗余、几何覆盖冗余五种核心集选择算法，以及多种融合策略。
- **密度过滤 FPS 最优算法**：结合最远点采样的空间覆盖能力与 k-NN 局部密度过滤的噪声修正机制，以仅 10% 训练数据达到 Test MSE = **0.003792**，较随机基线降低 **47.5%**。
- **融合实验的深层规律**：18 组融合实验全部未能超越单一最优，揭示了信息漏斗效应、信号非正交性与信息相关性上限等结构性瓶颈。

---

## 项目结构

```
PROJECT/
├── data/                       # 数据集与特征
│   ├── raw/                    # 原始 ALOHA Sim 数据
│   ├── features/               # CLIP 视觉/文本特征、动作标签
│   └── processed/              # 实验输出
├── src/                        # 公共基础设施
│   ├── config.py               # 全局配置
│   ├── data_loader.py          # 数据加载与划分
│   ├── train_mlp.py            # MLP 训练流程
│   └── evaluate.py             # 测试评估
├── approaches/                 # 各核心集选择算法
│   ├── redundancy_distribution/    # 自适应聚类 + PCA
│   ├── temporal_awareness/         # 时序事件边界
│   ├── autoencoder_reconstruction/ # AE 重建误差
│   ├── farthest_point_sampling/    # FPS 系列（全局最优）
│   ├── predictive_coding_sliding_window/  # 预测编码
│   └── fusion_score_weighted/      # 融合策略
├── scripts/                    # 特征提取、基线、对比脚本
├── requirements.txt
└── README.md
```

---

## 环境配置

```bash
cd PROJECT

# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

### 1. 提取 CLIP 特征

```bash
cd scripts
python extract_features.py
```

### 2. 运行随机基线

```bash
python run_baseline.py
```

### 3. 运行核心集方法

```bash
# 全局最优：密度过滤 FPS
python -m approaches.farthest_point_sampling.scripts.run_density_filtered_fps

# 亚军：自适应聚类 + PCA
python -m approaches.redundancy_distribution.scripts.run_v1_1_pca

# 对比所有结果
python compare_results.py
```

---

## 核心实验结果

所有方法在相同测试集上评估，使用相同 MLP 结构与训练配置。

| 排名 | 方法 | 冗余定义 | Test MSE | vs 随机基线 |
|------|------|---------|---------|------------|
| 1 | **密度过滤 FPS** | 几何覆盖冗余 | **0.003792** | **↓47.5%** |
| 2 | 自适应聚类 + PCA 400d | 分布冗余 | 0.003835 | ↓47.0% |
| 3 | 分层融合 cr2.0 | 融合 | 0.004004 | ↓44.6% |
| 4 | 预测编码滑动窗口 | 时序冗余 | 0.005688 | ↓21.3% |
| — | 随机基线 | — | 0.007229 | — |
| — | 时序事件边界 | 时序冗余 | 0.008871 | ↑22.7% |
| — | AE 重建误差 | 压缩冗余 | ~0.007 | ~0% |

---

## 完整实验记录

全部模型权重、训练结果、论文报告与中间数据请参见 [`backup`](https://github.com/DCY501/brain-inspired-coreset-vla/tree/backup) 分支。

---

## 参考文献

- Zhao T, et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *RSS*, 2023.
- Sorscher B, et al. Beyond neural scaling laws: beating power law scaling via data pruning. *NeurIPS*, 2022.
- Kim K, et al. OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*, 2024.
- Radford A, et al. Learning Transferable Visual Models From Natural Language Supervision. *ICML*, 2021.
- He K, et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
- Millidge A, et al. Predictive coding: a theoretical and experimental review. *arXiv:2107.12979*, 2021.
