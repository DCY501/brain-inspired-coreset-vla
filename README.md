# 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测（完整记录版）

> **本分支（`backup`）为完整实验记录版本**，包含所有模型权重（`.pt`）、训练结果（`.json`）、中间数据与原始实验脚本。若只需查看算法与报告，请切换至 [`main`](https://github.com/DCY501/brain-inspired-coreset-vla/tree/main) 分支。

---

## 项目简介

本项目探索如何在视觉-语言-动作（VLA）模仿学习任务中，通过脑启发核心集选择算法从海量训练帧中筛选最具信息价值的高质量子集，以少量数据训练出高性能的轻量级动作预测模型。

核心思想是从"冗余如何定义"出发，将大脑的预测编码、RAS 网状激活、事件分割、意外检测和海马体空间编码五种信息筛选机制映射为核心集选择算法，系统验证不同冗余定义与数据特性的匹配度。

## 核心贡献

- **系统性的冗余定义探索**：独立设计并验证了分布冗余、时序事件分割、预测编码、压缩冗余、几何覆盖冗余五种核心集选择算法，以及多种融合策略。
- **密度过滤 FPS 最优算法**：结合最远点采样的空间覆盖能力与 k-NN 局部密度过滤的噪声修正机制，以仅 10% 训练数据达到 Test MSE = **0.003792**，较随机基线降低 **47.5%**。
- **融合实验的深层规律**：18 组融合实验全部未能超越单一最优，揭示了信息漏斗效应、信号非正交性与信息相关性上限等结构性瓶颈。

## 实验记录

本分支完整保留了所有实验产物：

- `approaches/*/results/*.pt`：各方法训练得到的模型权重
- `approaches/*/results/*.json`：详细的训练/测试结果
- `approaches/*/results/*.npy`：核心集索引等中间数据
- `approaches/*/README.md`：各思路完整的实验历程与失败分析
- `report/`：论文报告 LaTeX 源文件与全部图表

## 核心结果

| 排名 | 方法 | 冗余定义 | Test MSE | vs 随机基线 |
|------|------|---------|---------|------------|
| 1 | **密度过滤 FPS** | 几何覆盖冗余 | **0.003792** | **↓47.5%** |
| 2 | 自适应聚类 + PCA 400d | 分布冗余 | 0.003835 | ↓47.0% |
| 3 | 分层融合 cr2.0 | 融合 | 0.004004 | ↓44.6% |
| 4 | 预测编码滑动窗口 | 时序冗余 | 0.005688 | ↓21.3% |
| — | 随机基线 | — | 0.007229 | — |
| — | 时序事件边界 | 时序冗余 | 0.008871 | ↑22.7% |
| — | AE 重建误差 | 压缩冗余 | ~0.007 | ~0% |

## 分支说明

- **`main`**：精简版，只包含算法源码、报告与核心图表，无大文件，适合快速浏览与复现。
- **`backup`**：完整记录版（本分支），包含全部权重、结果与实验历程，用于追溯和对比分析。

## 环境配置

```bash
cd PROJECT
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

## 快速开始

```bash
# 提取 CLIP 特征
cd scripts
python extract_features.py

# 运行全局最优方法
python -m approaches.farthest_point_sampling.scripts.run_density_filtered_fps

# 运行随机基线对比
python run_baseline.py
```

## 参考文献

- Zhao T, et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *RSS*, 2023.
- Sorscher B, et al. Beyond neural scaling laws: beating power law scaling via data pruning. *NeurIPS*, 2022.
- Kim K, et al. OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*, 2024.
- Radford A, et al. Learning Transferable Visual Models From Natural Language Supervision. *ICML*, 2021.
- He K, et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
- Millidge A, et al. Predictive coding: a theoretical and experimental review. *arXiv:2107.12979*, 2021.
