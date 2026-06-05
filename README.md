# 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

> **课程**：视觉认知工程（必修）大作业
> **题目**：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测
> **版本**：v2.0（CLIP 多模态 + 预测编码进阶算法）
> **核心目标**：在 ALOHA 仿真数据集上验证——通过脑启发核心集选择算法筛选出的 10% 高质量数据，能够优于随机抽样的 10% 数据训练出的 MLP 模型。

---

## 📌 版本说明

本项目已从 v1.0（ResNet-18 基线）升级至 **v2.0（CLIP + 进阶算法）**：

| 维度 | v1.0 | v2.0（当前） |
|------|------|-------------|
| 前端网络 | ResNet-18（仅视觉） | **CLIP**（视觉+语言对齐） |
| 语言指令 | one-hot 退化常数 | **CLIP text encoder 语义编码** |
| 时序过滤 | 相邻帧数值差分 | **滑动窗口自回归预测编码** |
| 权重融合 | 固定 0.6/0.4 | **自适应 sigmoid 权重** |
| 消融实验 | ❌ 无 | **✅ 四组对照实验** |

v1.0 的所有代码、特征、实验结果已归档至项目根目录的 `resnet18_legacy/` 文件夹。

---

## 📁 项目目录详解

```
PROJECT/
├── venv/                           # Python 虚拟环境（已配置好，见下文）
├── data/
│   ├── raw/                        # 原始 ALOHA 数据集（由 lerobot 下载）
│   ├── features/                   # 离线提取的 CLIP 特征
│   │   ├── clip_visual_features.npy    # (20000, 768) CLIP 视觉特征
│   │   ├── clip_text_features.npy      # (20000, 512) CLIP 文本特征
│   │   ├── actions.npy                 # (20000, 7) 单臂动作标签
│   │   └── episode_indices.npy         # (20000,) 每帧所属 episode 编号
│   └── processed/                  # 实验输出（运行后自动生成）
├── src/                            # 公共基础设施（被各思路脚本调用）
│   ├── config.py                   # 全局常量：路径、超参数、实验配置
│   ├── data_loader.py              # 封装 lerobot 接口 + 训练/测试划分 + PyTorch Dataset
│   ├── train_mlp.py                # MLPRegressor 模型定义 + 训练流程（含早停）
│   └── evaluate.py                 # 测试集评估：MSE、MAE、每关节误差
├── approaches/                     # 各思路独立存放（核心集选择算法实验）
│   ├── redundancy_distribution/    # 思路一：v1.1 + PCA 400d K-Means 聚类
│   │   ├── src/selector_v1_1_pca_best.py
│   │   └── results/                # 实验结果 JSON + PT 权重
│   ├── temporal_awareness/         # 思路二：动作变化率事件边界（已废弃）
│   ├── autoencoder_reconstruction/ # 思路三：AE 重建误差（已废弃）
│   ├── farthest_point_sampling/    # 思路四：FPS 系列（密度过滤 FPS 全局最优）
│   │   ├── src/selector_fps.py
│   │   ├── src/selector_fps_fast.py
│   │   ├── src/selector_density_filtered_fps.py
│   │   └── results/
│   ├── fusion_score_weighted/      # 融合思路：得分加权 v1（失败）+ 分层融合 v2.0
│   │   ├── src/selector_score_fusion.py
│   │   ├── src/selector_hierarchical_fusion.py
│   │   └── results/
│   └── README.md                   # 各思路详细说明
├── scripts/                        # 可执行脚本（特征提取、基线、消融等）
│   ├── extract_features.py
│   ├── run_baseline.py
│   ├── run_ablation.py
│   └── compare_results.py
├── report/                         # 实验报告图表
├── requirements.txt
└── README.md                       # 本文件
```

---

## 🚀 完整实验流程

### 前置要求

- **Python**：3.10 或更高版本
- **操作系统**：Windows 10/11（本项目在 Windows 下开发测试）
- **硬件**：CPU 即可（数据集极小，无需 GPU）

### ⚠️ 重要：必须使用虚拟环境

本项目已配置好独立的 Python 虚拟环境 `PROJECT/venv/`，与系统 Python **完全隔离**。

**环境隔离状态**：

| 环境 | torch 版本 | 用途 | 是否会影响其他项目 |
|------|-----------|------|------------------|
| 系统 Python | `2.6.0+cu124` (GPU) | 你原来的其他项目 | ❌ 不受影响 |
| `PROJECT/venv` | `2.10.0+cpu` (CPU) | 本作业项目 | ❌ 完全隔离 |

### Step 0: 激活虚拟环境并安装依赖

打开终端，进入 `PROJECT/` 目录，**激活虚拟环境**：

```bash
cd PROJECT

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装依赖（仅需一次）
pip install -r requirements.txt
```

激活成功后，命令行提示符前会出现 `(venv)` 字样，例如：
```
(venv) E:\...\大作业\PROJECT>
```

> 以后每次运行实验脚本前，都要先执行 `venv\Scripts\activate` 激活虚拟环境。用完可执行 `deactivate` 退出。

---

### Step 1: 离线提取 CLIP 多模态特征

```bash
cd scripts

# 确保已激活虚拟环境（提示符前有 (venv)）
python extract_features.py
```

**这个脚本做了什么？**
1. 加载 `data/raw/` 下的 ALOHA 数据集（50 episodes，20,000 帧）
2. 用 **冻结权重的 CLIP** 提取多模态特征：
   - `CLIPVisionModel`：每帧图像 `(3, 480, 640)` → `Resize(224,224)` → **512 维视觉特征**
   - `CLIPTextModel`：语言指令 → **512 维文本特征**
3. 视觉+文本特征拼接为 `(1024,)` 向量，用于后续 MLP 训练
4. 分布过滤模块单独使用 512 维视觉特征（语言指令恒定，无分布信息）
5. 同时提取并保存：单臂 7DoF 动作、`episode_index`、任务名称
6. 所有 `.npy` 文件保存到 `data/features/`

**预计耗时**：CPU 约 15~30 分钟（CLIP 比 ResNet-18 稍慢）。

**输出文件**：
- `data/features/clip_visual_features.npy` —— (20000, 512)
- `data/features/clip_text_features.npy` —— (20000, 512)
- `data/features/actions.npy` —— (20000, 7)
- `data/features/episode_indices.npy` —— (20000,)

> **只需运行一次**。后续 Baseline、消融实验和 Coreset 实验直接读取这些 `.npy` 文件。

---

### Step 2: 运行 Baseline 实验

```bash
# 确保已激活虚拟环境
python run_baseline.py
```

**实验设计细节**：
- **训练/测试划分**：固定按 episode 级别划分。50 个 episode 中，10 个（20%）作为**固定测试集**，40 个作为训练集。
- **Baseline 采样**：从 40 个训练 episode 中**随机抽取 4 个**（10%），约 1,600 帧。
- **模型**：轻量级 MLP，输入 1024+1=1025 维（CLIP 拼接特征），输出 7 维（单臂动作）。隐藏层 `[512, 256, 128]`。
- **训练**：Adam 优化器，学习率 1e-3，BatchSize 64，最大 100 epoch，早停耐心 15。
- **重复性**：默认运行 **5 次**（不同随机种子），每次独立采样 + 独立训练，最终取 MSE 平均与标准差。

**输出文件**：
- `data/processed/baseline_seed*.pt` —— 每次运行的最优模型权重
- `data/processed/baseline_result_seed*.json` —— 每次运行的详细结果
- `data/processed/baseline_summary.json` —— 5 次汇总（avg ± std）

---

### Step 3: 运行消融实验

```bash
# 确保已激活虚拟环境
python run_ablation.py
```

**实验设计细节**：
- **Temporal-Only**：仅使用滑动窗口预测编码时序过滤，无分布过滤
- **Diversity-Only**：仅使用 CLIP 视觉特征 K-Means 分布过滤，无时序过滤
- **BrainInspired-v2**：完整混合策略（预测编码 + 分布过滤 + 自适应权重）
- **训练数据量**：每组均筛选约 1,600 帧（10%）
- **MLP 结构与超参数**：与 Baseline 完全一致

**输出文件**：
- `data/processed/ablation_*.json` —— 各组实验结果
- `data/ablation/` —— 中间数据（如各组选中的核心集索引）

**预期结论**：`BrainInspired-v2 > Temporal-Only ≈ Diversity-Only > Random`

---

### Step 4: 运行核心集实验（完整版）

```bash
# 确保已激活虚拟环境
python run_coreset.py
```

**实验设计细节**：
- **同样的训练/测试划分**：与 Baseline 使用完全相同的测试集
- **核心集选择**：从全部 40 个训练 episode（约 16,000 帧）中，用脑启发算法 v2.0 筛选 **10% 核心帧**（约 1,600 帧）
- **同样的模型与超参数**：MLP 结构、学习率、BatchSize、早停策略与 Baseline 完全一致
- **唯一变量**：训练数据的筛选方式

**核心集算法 v2.0 原理**：
1. **预测编码时序得分**：滑动窗口自回归预测动作，预测误差小的帧 = 时序冗余
2. **分布多样性得分**：对 CLIP 视觉特征做 K-Means 聚类，稀疏簇和远离中心的样本得分高
3. **自适应权重融合**：根据 episode 动作变化率动态调整 `w_temporal`
4. **两阶段选择**：簇覆盖（每簇至少1帧）+ 全局补齐

**输出文件**：
- `data/processed/coreset_seed*.pt` —— 最优模型权重
- `data/processed/coreset_result_seed*.json` —— 详细结果（含被选中的核心集索引）

---

### Step 5: 结果对比与可视化

```bash
# 确保已激活虚拟环境
python compare_results.py
```

**自动生成的图表**（保存到 `report/comparison.png`）：

| 子图 | 内容 | 说明 |
|------|------|------|
| 左图 | **MSE 柱状图** | Random / Temporal-Only / Diversity-Only / BrainInspired 四组对比 |
| 中图 | **验证集学习曲线** | 展示各组的收敛速度与最终 loss |
| 右图 | **每关节 MSE 对比** | 7 个关节各自的预测误差细粒度分析 |

**同时生成**：`report/summary.json`，包含各组平均 MSE、标准差、提升百分比。

---

## 🧠 脑启发核心集选择算法 v2.0 详解

### 算法与脑机制的映射关系

| 脑机制 | 论文来源 | 算法对应模块 | 冗余定义 |
|--------|---------|-------------|---------|
| **预测编码 (Predictive Coding)** | Millidge et al., 2021 | `compute_predictive_coding_scores()` | 自回归预测误差小的帧 = 大脑能预测 = 时序冗余 |
| **RAS 网状激活系统** | 题目背景 + 认知神经科学 | `compute_diversity_scores()` 中的稀疏簇偏好 | 视觉特征稀疏簇中的帧 = 高信息效用 |
| **分布均衡 / 数据修剪** | Sorscher et al., 2022 | K-Means 聚类 + 簇覆盖约束 | 视觉特征空间中过度聚集的帧 = 分布冗余 |
| **皮层可塑性 / 自适应** | 本文独立设计 | `adaptive_weight()` | 根据数据特性动态调整时序/分布权重 |

### 综合策略流程图

```
输入: 训练集全部帧 (N=16000)
    │
    ├─→ [预测编码时序过滤]
    │              滑动窗口自回归预测动作
    │              → 计算预测误差 s_pc (N,)
    │
    ├─→ [分布过滤] 对 CLIP 视觉特征做 K-Means 聚类
    │              → 计算到中心距离 × 簇大小惩罚
    │              → 得到分布得分 s_diversity (N,) 和聚类标签 labels
    │
    ├─→ [自适应权重] w_temporal = sigmoid(α · (σ_action - μ_global))
    │
    └─→ [加权融合] final_scores = w_temporal * s_pc + (1-w_temporal) * s_diversity
                   │
                   ├─→ [簇覆盖] 每个簇至少保留 1 帧（保证分布均衡）
                   │
                   └─→ [补齐] 按全局得分从高到低补充到 target_size=1600

输出: 核心集索引 (1600,) → 用于训练 MLP
```

---

## ⚙️ 配置说明（`src/config.py`）

所有超参数集中在此文件，修改一处即可全局生效：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_SINGLE_ARM` | `True` | 是否降维为单臂（作业要求降维实验） |
| `SINGLE_ARM_DIMS` | `list(range(7))` | 取左臂 0~6；改为 `range(7,14)` 取右臂 |
| `TEST_EPISODE_RATIO` | `0.2` | 20% episode 作为固定测试集（10/50） |
| `BASELINE_SAMPLE_RATIO` | `0.1` | Baseline 随机抽 10% 训练 episode（4/40） |
| `CORESET_RATIO` | `0.1` | 核心集筛选 10% 帧（约 1600/16000） |
| `CLIP_MODEL_NAME` | `"openai/clip-vit-base-patch32"` | CLIP 预训练模型名称 |
| `FEATURE_DIM` | `1280` | CLIP 视觉(768) + 文本(512) 拼接维度 |
| `VISUAL_FEATURE_DIM` | `768` | CLIP 视觉特征单独维度（用于分布过滤） |
| `TEMPORAL_WINDOW` | `5` | 预测编码滑动窗口大小 |
| `TEMPORAL_WEIGHT_ALPHA` | `1.0` | 自适应权重温度系数 |
| `MLP_HIDDEN_DIMS` | `[512, 256, 128]` | MLP 隐藏层神经元数 |
| `MLP_DROPOUT` | `0.1` | Dropout 比率 |
| `LEARNING_RATE` | `1e-3` | Adam 初始学习率 |
| `BATCH_SIZE_TRAIN` | `64` | 训练批次大小 |
| `EPOCHS` | `100` | 最大训练轮数 |
| `EARLY_STOPPING_PATIENCE` | `15` | 验证集 MSE 不改善的容忍轮数 |
| `RANDOM_SEED` | `42` | 全局随机种子 |

---

## 📊 实验设计要点（公平性保障）

1. **控制变量**：所有实验组使用**同样的 MLP 结构**、**同样的超参数**、**同样的训练/测试划分**，唯一区别是**训练数据的筛选方式**。
2. **数据隔离**：测试集按 **episode 级别**划分，确保测试数据从未在训练时见过，避免信息泄露。
3. **多次随机**：Baseline 运行 5 次取平均，排除"恰好抽到好 episode"的随机波动干扰。
4. **降维合理**：只取单视角 (`observation.images.top`) 和单臂前 7 维动作，符合题目"可选择仅提取其中一个视角和单臂动作标签进行降维实验"的要求。
5. **消融对照**：设计 Temporal-Only / Diversity-Only / BrainInspired / Random 四组实验，证明混合策略中每个模块的独立贡献。

---

## 📊 实验结果汇总（approaches/ 全部方法）

所有实验在 **相同测试集** 上评估（按 episode 划分，20% 测试集，seed=42），MLP 结构一致。

### 核心集选择方法对比

| 排名 | 方法 | 思路 | Test MSE | Test MAE | 状态 |
|------|------|------|---------|---------|------|
| 🥇 | **密度过滤 FPS** | 思路四 | **0.003792** | 0.0339 | ✅ 全局最优 |
| 🥈 | **v1.1 + PCA 400d** | 思路一 | **0.003835** | — | ✅ 亚军 |
| 3 | 加权 FPS (feature_norm) | 思路四 | 0.004162 | 0.0353 | ✅ |
| 4 | 标准 FPS | 思路四 | 0.004294 | 0.0359 | ✅ |
| 5 | **分层融合 cr2.0** | 融合 v2.0 | **0.004004** | 0.0349 | ✅ |
| 6 | **分层融合 cr3.0** | 融合 v2.0 | **0.004604** | 0.0384 | ✅ |
| 7-10 | 得分融合 v1.0 (4组) | 融合 v1.0 | 0.0054~0.0095 | — | ❌ 均失败 |
| — | 时序事件边界 | 思路二 | 0.008871 | 0.0612 | ❌ 已废弃 |
| — | AE 重建误差 | 思路三 | ~0.007~0.008 | — | ❌ 已废弃 |

**随机基线**：~0.007229（参考值）

### 关键结论

1. **密度过滤 FPS 全局最优**：标准 FPS 容易选到孤立噪声点，k-NN 局部密度过滤剔除 15% 最稀疏点后补齐，效果最好（MSE 0.003792）。
2. **v1.1 + PCA 400d 紧随其后**：K-Means 聚类 + 动态保底 + PCA 降维去噪，MSE 0.003835，与密度过滤 FPS 差距极小。
3. **分层融合无法超越单一最优**：cr2.0 (0.004004) 介于 v1.1 和标准 FPS 之间，cr3.0 (0.004604) 反而更差。说明 v1.1 粗筛的"信息漏斗"效应无法被 FPS 精筛弥补。
4. **得分融合 v1.0 失败**：`s_v1`（语义离群）和 `s_fps`（几何离群）都倾向选离群帧，线性加权后双重强调噪声。
5. **时序和 AE 思路废弃**：单任务时序耦合过强、CLIP 特征太规整导致重建误差无区分度。

### 运行各思路实验

```bash
cd PROJECT
venv\Scripts\activate

# 思路一：v1.1 + PCA 400d
python -m approaches.redundancy_distribution.scripts.run_v1_1_pca

# 思路四：FPS 系列
python -m approaches.farthest_point_sampling.scripts.run_fps
python -m approaches.farthest_point_sampling.scripts.run_fps_fast
python -m approaches.farthest_point_sampling.scripts.run_density_filtered_fps

# 融合思路
python -m approaches.fusion_score_weighted.scripts.run_hierarchical_fusion
```

---

## 📝 报告撰写指南

运行完实验后，`report/comparison.png` 即为核心图表。撰写报告时建议按以下结构展开：

### 1. 背景与动机（对应题目要求"背景机制调研"）
- 介绍 VLA 模型与具身智能的关系（引用 OpenVLA / ACT）
- 阐述数据冗余问题的严重性：时序冗余、分布冗余、次优噪声
- 引出脑启发机制：预测编码（Millidge）、RAS 注意力、核心集选择原理（Sorscher）
- **v2.0 新增**：CLIP 多模态对齐在 VLA 中的重要性（引用 Radford）

### 2. 核心集选择算法设计（对应题目要求"重点体现对冗余如何定义的独立思考"）
- **预测编码的量化公式**：滑动窗口自回归预测误差的定义
- **分布冗余的量化公式**：K-Means 聚类 + 簇大小惩罚
- **自适应权重**：根据动作变化率动态调整时序/分布权重
- **脑机制映射表**：明确写出算法哪一行代码对应预测编码、哪一部分对应 RAS
- **消融实验设计**：四组对照实验的必要性说明

### 3. 实验设置
- 数据集介绍（ALOHA Sim Transfer Cube，50 episodes，20,000 帧）
- **CLIP 特征提取**：视觉+文本双编码器的详细配置
- 训练/测试划分方式
- Baseline、消融实验、Coreset 的公平性控制说明
- MLP 结构、超参数

### 4. 对比实验结果分析（对应题目要求"对比实验结果分析"）
- MSE 数值对比表格（Random / Temporal-Only / Diversity-Only / BrainInspired）
- 消融实验分析：证明"混合 > 单一 > 随机"
- 学习曲线分析（收敛速度、过拟合程度）
- 每关节误差分析（哪个关节最难预测？为什么？）
- **结论**：高质量核心集是否优于随机子集？提升百分比是多少？预测编码是否优于数值差分？

### 5. v1.0 → v2.0 的升级总结
- 表格对比 ResNet-18 基线与 CLIP 进阶版的差异
- 说明为什么升级前端网络和算法

### 6. 附录：源代码与注释
- 报告末尾附上关键代码（核心集选择算法、MLP 定义、训练流程）
- 代码需有清晰注释，说明每一步的数学含义

---

## 🛠️ 常见问题排查（FAQ）

### Q1: `ImportError: attempted relative import with no known parent package`
**原因**：`src/` 下的模块使用了相对导入（`from .config import *`），但直接运行 `scripts/*.py` 时 `src/` 未被识别为包。
**解决**：确保 `scripts/*.py` 开头已包含：
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

### Q2: CLIP 特征提取速度极慢 / 内存占用高
**原因**：CLIP ViT-B/32 在 CPU 上处理 20,000 帧较耗时。
**解决**：
- 减小 `src/config.py` 中的 `BATCH_SIZE_EXTRACT`（如从 32 改为 16 或 8）
- 如有 NVIDIA GPU，修改 `extract_features.py` 中 `device='cuda'`

### Q3: 想换右臂做实验？
修改 `src/config.py`：
```python
SINGLE_ARM_DIMS = list(range(7, 14))  # 右臂：7~13
```
然后重新运行 `extract_features.py` 和实验脚本。

### Q4: 核心集比例想调整（如 5% 或 20%）？
修改 `src/config.py`：
```python
CORESET_RATIO = 0.05  # 或 0.2
```
然后重新运行 `run_ablation.py`、`run_coreset.py` 和 `compare_results.py`。

### Q5: 路径包含中文，运行时报错？
**排查步骤**：
1. 确认报错信息是否真的是路径问题（而非导入问题、库缺失问题）
2. 在 `scripts/*.py` 开头加入：`# -*- coding: utf-8 -*-`
3. 绝大多数情况下，中文路径**不会**导致 Python 代码报错

### Q6: 忘记激活虚拟环境，直接用系统 Python 运行报错？
**现象**：`transformers` 或 `lerobot` 导入失败，或提示 `torch` 版本不兼容。
**解决**：
```bash
# 先激活虚拟环境，再用虚拟环境的 Python 运行
cd PROJECT
venv\Scripts\activate
cd scripts
python extract_features.py
```

### Q7: v1.0 的 ResNet-18 结果在哪里？
全部归档在项目根目录的 `resnet18_legacy/` 文件夹中，包含：
- 特征文件 (`data/features/`)
- 实验结果 (`data/processed/`)
- 报告图表 (`report/`)
- 旧版特征提取代码 (`src/feature_extractor.py`, `scripts/extract_features.py`)

---

## 📚 参考文献

1. **Zhao et al., 2023.** Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT). *Robotics: Science and Systems (RSS).* https://arxiv.org/abs/2304.13705
2. **Sorscher et al., 2022.** Beyond neural scaling laws: beating power law scaling via data pruning. *Advances in Neural Information Processing Systems (NeurIPS).* https://arxiv.org/abs/2206.14486
3. **Kim et al., 2024.** OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv preprint.* https://arxiv.org/abs/2406.09246
4. **Millidge et al., 2021.** Predictive coding: a theoretical and experimental review. *arXiv preprint.* https://arxiv.org/abs/2107.12979
5. **Radford et al., 2021.** Learning Transferable Visual Models From Natural Language Supervision (CLIP). *International Conference on Machine Learning (ICML).* https://arxiv.org/abs/2103.00020
6. **He et al., 2016.** Deep Residual Learning for Image Recognition (ResNet). *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).*

---

*项目已更新至 v2.0（CLIP + 进阶算法），approaches/ 目录包含全部核心集选择实验。全局最优方法：**密度过滤 FPS**（Test MSE 0.003792）。按顺序运行 `extract_features.py` → `run_baseline.py` → `run_ablation.py` → `run_coreset.py` → `compare_results.py` 即可获得完整实验结果。祝实验顺利！*
