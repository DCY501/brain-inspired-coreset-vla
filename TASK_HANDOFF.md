# 任务转接文档 — 脑启发核心集选择 VLA 机械臂动作预测

> **转接原因**：当前实验因 K-Means 聚类超时，需切换至 kimi work 环境继续完成。
> **当前时间**：2026-06-03
> **工作目录**：`E:\本科\课业\大三下\视觉认知工程（必修）\大作业\PROJECT`

---

## 一、项目背景与目标

**课程**：视觉认知工程（必修）大作业
**题目**：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测
**核心目标**：在 ALOHA Sim Transfer Cube 数据集（50 episodes, ~20K 帧）上，用核心集选择算法从训练集中筛选 10% 高质量数据，训练 MLP 模型预测机械臂 7DoF 动作，使 Test MSE 优于随机抽样的 10% 基线。

**GitHub 仓库**：`https://github.com/DCY501/brain-inspired-coreset-vla`

---

## 二、数据与特征

| 项目 | 详情 |
|------|------|
| 数据集 | LeRobot ALOHA sim_transfer_cube_scripted |
| 规模 | 50 episodes, ~20,000 帧, 单任务（语言指令恒定） |
| 视觉特征 | CLIP ViT-B/32, 768d (`clip_visual_features.npy`) |
| 文本特征 | CLIP ViT-B/32, 512d (`clip_text_features.npy`) |
| 拼接特征 | 1280d（训练 MLP 用） |
| 动作标签 | 7DoF 单臂动作 (`actions.npy`) |
| episode 索引 | `episode_indices.npy` |
| 训练/测试划分 | 按 episode 划分，10/50 测试集，固定 seed=42 |

---

## 三、项目结构（已重构）

```
PROJECT/
├── src/                          # 公共基础设施（不可直接运行）
│   ├── config.py                 # 全局配置：CORESET_RATIO=0.1, TEST_EPISODE_RATIO=0.2 等
│   ├── data_loader.py            # FeatureDataset, 训练/测试划分
│   ├── train_mlp.py              # MLPRegressor 定义 + train_model + evaluate_model
│   └── evaluate.py               # 测试集评估
│
├── approaches/                   # 各思路独立存放（已完成重构迁入）
│   ├── redundancy_distribution/  # 思路一：v1.1 + PCA 400d K-Means
│   │   ├── src/selector_v1_1_pca_best.py
│   │   └── results/              # 实验结果 JSON + PT 权重
│   ├── temporal_awareness/       # 思路二：动作变化率事件边界（已废弃）
│   ├── autoencoder_reconstruction/ # 思路三：AE 重建误差（已废弃）
│   ├── farthest_point_sampling/  # 思路四：FPS 系列
│   │   ├── src/selector_fps.py              # 标准 FPS
│   │   ├── src/selector_fps_fast.py         # 优化版 FPS（预计算范数）
│   │   ├── src/selector_density_filtered_fps.py  # 密度过滤 FPS ✅最优
│   │   └── results/
│   ├── fusion_score_weighted/    # 融合思路 v1 + v2
│   │   ├── src/selector_score_fusion.py     # 得分加权 v1（已废弃）
│   │   ├── src/selector_hierarchical_fusion.py # 分层融合 v2.0
│   │   └── results/              # 已有 cr2.0 结果，cr3.0 待跑
│   └── README.md                 # 各思路详细说明（建议阅读）
│
├── data/features/                # CLIP 提取的 .npy 特征文件
├── venv/                         # Python 虚拟环境（已配置）
└── README.md                     # 项目 README（v1.0 版本，待更新）
```

---

## 四、关键配置参数（config.py）

```python
CORESET_RATIO = 0.1              # 筛选 10% 帧
TEST_EPISODE_RATIO = 0.2         # 20% episode 作为测试集
VISUAL_FEATURE_DIM = 768         # CLIP 视觉特征维度
BATCH_SIZE_TRAIN = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 1e-3
MLP_HIDDEN_DIMS = [512, 256, 128]
```

---

## 五、实验结果汇总（截至转接时）

按 Test MSE 从小到大排序：

| 排名 | 方法 | 思路 | MSE | MAE | 状态 |
|------|------|------|-----|-----|------|
| 🥇 | **密度过滤 FPS** | 思路四 | **0.003792** | 0.0339 | ✅ 全局最优 |
| 🥈 | **v1.1 + PCA 400d** | 思路一 | **0.003835** | — | ✅ 亚军 |
| 3 | 加权 FPS (feature_norm) | 思路四 | 0.004162 | 0.0353 | ✅ |
| 4 | 标准 FPS | 思路四 | 0.004294 | 0.0359 | ✅ |
| 5 | **分层融合 cr2.0** | 融合 v2.0 | **0.004004** | 0.0349 | ✅ 融合最佳 |
| 6 | **分层融合 cr3.0** | 融合 v2.0 | **0.004604** | 0.0384 | ✅ |
| 7 | **分层融合 cr2.0_dfps** | 融合 v2.0 | **0.005272** | 0.0376 | ❌ 密度过滤 FPS 在子空间内失效 |
| 8-11 | 得分融合 v1.0 (4组) | 融合 v1.0 | 0.0054~0.0095 | — | ❌ 均失败 |
| — | 时序事件边界 | 思路二 | 0.008871 | 0.0612 | ❌ 单任务时序耦合 |
| — | AE 重建误差 | 思路三 | ~0.007~0.008 | — | ❌ CLIP 特征太规整 |

**随机基线**：~0.007229（未正式跑，参考值）

---

## 六、当前待完成任务

### 🔴 高优先级：跑完分层融合 cr3.0 —— ✅ 已完成

**问题**：分层融合 v2.0 的 `candidate_ratio=3.0` 版本因 K-Means 聚类超时未跑出结果。

**根本原因**：
- `selector_v1_1_pca_best.py` 中 `n_clusters = max(2, target_size // 3)`
- cr3.0 时 v1.1 的 target_size = 4800（候选池大小）
- 导致 n_clusters = 1600，在 16000 点上聚 1600 簇 × n_init=10，CPU 上 300 秒超时

**解决方案**（已实施）：
修改 `selector_hierarchical_fusion.py`，给 v1.1 传一个基于**原始数据量**的固定 n_clusters：
```python
fixed_n_clusters = max(10, len(features) // 30)  # 533 簇（原 1600 簇）
candidates = v1_selector.select(features, episode_indices, n_clusters=fixed_n_clusters)
```

**cr3.0 结果**：Test MSE = **0.004604**，MAE = 0.0384

**结论**：
- cr3.0 (0.004604) **不如** cr2.0 (0.004004)
- 增大候选池（3.0× vs 2.0×）**没有**带来改善，反而略有下降
- 分层融合 v2.0 的最佳表现是 cr2.0 的 **0.004004**，但仍不如单一最优方法（密度过滤 FPS 0.003792、v1.1+PCA400 0.003835）
- **假设不成立**：增大候选池并不能让 FPS 有更大优化空间，v1.1 的粗筛本身已经引入了信息损失

### 🟡 中优先级：更新项目 README

当前 `README.md` 还是 v1.0 版本，没有反映 approaches 目录结构。需要：
1. 更新目录结构说明
2. 补充所有实验结果表格
3. 添加 approaches/ 下各思路的入口说明

### 🟢 低优先级：代码推送

- `git add` 所有新文件（approaches/ 下的代码、结果）
- `git commit` 提交当前状态
- `git push` 到 GitHub

---

## 七、关键代码文件清单

| 文件 | 作用 | 是否需要修改 |
|------|------|-------------|
| `approaches/redundancy_distribution/src/selector_v1_1_pca_best.py` | v1.1+PCA 核心集选择器 | ✅ 可能需要支持外部传 n_clusters |
| `approaches/farthest_point_sampling/src/selector_fps.py` | 标准 FPS | ❌ |
| `approaches/farthest_point_sampling/src/selector_fps_fast.py` | 优化版 FPS（预计算范数） | ❌ |
| `approaches/farthest_point_sampling/src/selector_density_filtered_fps.py` | 密度过滤 FPS ✅最优 | ❌ |
| `approaches/fusion_score_weighted/src/selector_hierarchical_fusion.py` | 分层融合 v2.0 | ✅ 需要修改 n_clusters 传参 |
| `approaches/fusion_score_weighted/src/selector_score_fusion.py` | 得分融合 v1.0（已废弃） | ❌ |
| `src/config.py` | 全局配置 | ❌ |
| `src/train_mlp.py` | MLP + 训练 | ❌ |
| `src/evaluate.py` | 评估 | ❌ |
| `src/data_loader.py` | 数据加载 | ❌ |

---

## 八、运行环境

```bash
# 激活虚拟环境（Windows）
cd PROJECT
venv\Scripts\activate

# Python 版本：3.12
# PyTorch：2.10.0+cpu
# 关键依赖：numpy, scikit-learn, torch
```

**所有实验固定随机种子**：
- `torch.manual_seed(42)`
- `np.random.RandomState(42)`
- `KMeans(random_state=42)`
- `PCA(random_state=42)`

---

## 九、核心洞察与假设

1. **密度过滤 FPS 最优（0.003792）**：标准 FPS 选到孤立噪声点，k-NN 局部密度过滤剔除 15% 最稀疏点后补齐，效果最好。
2. **得分融合 v1.0 失败**：`s_v1`（语义离群）和 `s_fps`（几何离群）都倾向选离群帧，线性加权后双重强调噪声。
3. **分层融合 v2.0 假设**：v1.1 语义粗筛生成高质量候选池，FPS 在此池内精筛。cr2.0 结果（0.004004）不如密度过滤 FPS（0.003792），但优于标准 FPS（0.004294），说明 v1.1 的候选池质量尚可。
4. **cr3.0 假设验证失败**：增大候选池到 3.0× 后 MSE 反而上升到 0.004604（vs cr2.0 的 0.004004）。说明 v1.1 粗筛的候选池并非越大越好——过大的候选池引入了更多低质量帧，FPS 精筛无法完全弥补。
5. **密度过滤 FPS 在分层融合中失效**：将 Layer 2 换成密度过滤 FPS（cr2.0_dfps），MSE 恶化到 0.005272。原因是密度过滤 FPS 的全局最优性依赖于**全局 16000 帧的 k-NN 密度估计**，在 v1.1 粗筛后的候选池（3200 帧）内：候选池点相对均匀失去区分度、FPS 覆盖点被误杀、替换进来的反而是冗余簇内点。**关键洞察**：密度过滤 FPS 的优异表现是全局密度估计 + 全局替换的结果，嵌入分层结构限制在子空间内运行反而失效。
6. **分层融合上限**：分层融合 v2.0 的最佳表现（cr2.0: 0.004004）介于 v1.1（0.003835）和标准 FPS（0.004294）之间，但无法超越单一最优方法。说明两层结构的"信息漏斗"效应：v1.1 粗筛不可避免地过滤掉了部分有用帧，FPS 只能在剩余候选池内优化，无法恢复被过滤的信息。

---

## 十、下一步操作建议

1. ✅ **修复 cr3.0 K-Means 超时** —— 已完成，通过固定 n_clusters=533
2. ✅ **运行 cr3.0 实验** —— 已完成，MSE=0.004604（不如 cr2.0 的 0.004004）
3. ✅ **对比所有方法** —— 已完成，分层融合无法超越单一最优方法
4. ✅ **更新 README.md** —— 已完成：更新目录结构、补充实验结果表格、添加 cr2.0_dfps 分析
5. ⏳ **git commit + push** —— 待完成：提交所有修改到 GitHub

---

*转接完成。cr3.0 已跑完，cr2.0_dfps（密度过滤 FPS 替换 Layer 2）实验已跑完（MSE=0.005272，失效）。结论：分层融合 v2.0 的最佳表现是 cr2.0 (MSE=0.004004)，仍不如密度过滤 FPS (0.003792) 和 v1.1+PCA400 (0.003835)。密度过滤 FPS 的全局最优性无法通过简单嵌入分层结构复现。下一步建议更新 README 并提交代码。*
