# 思路四-A：最远点采样 (Farthest Point Sampling)

> **核心思想**：不聚类，直接贪心选"离已选集合最远的点"。每次迭代选特征空间中距离当前核心集最远的帧，直到选够 target_size。FPS 是 NP-hard 覆盖问题的贪心近似算法，能保证最大最小距离下界为最优解的 1/2。

---

## 标准 FPS

### 算法

```python
selected = [随机初始帧]
min_dists = [inf, inf, ..., inf]

for step in range(target_size - 1):
    dists = ||features - features[last_selected]||_2
    min_dists = min(min_dists, dists)
    next = argmax(min_dists)
    selected.append(next)
```

### 实验结果

| 配置 | Test MSE | vs 随机基线 |
|------|---------|-----------|
| 随机基线 | 0.007229 | — |
| **标准 FPS** | **0.004294** | **↓40.6%** |

---

## 改进尝试：加权 FPS (Weighted FPS)

在标准 FPS 基础上引入质量权重，得分 = min_dist x weight，试图实现"覆盖 + 质量"双目标优化。

### 权重策略对比

| 策略 | 原理 | Test MSE | 失败原因 |
|------|------|---------|---------|
| feature_norm | 视觉特征 L2 范数 | 0.004162 | CLIP 已归一化，权重全为 1.0 |
| action_delta | 动作变化率 | 0.007225 | 时序边界在视觉空间扎堆 |
| local_density | k-NN 平均距离 | 0.005761 | 过度偏向稀疏离群点，引入噪声 |

### 失败分析

**加权 FPS 要有效，必须同时满足两个条件：**

| 条件 | feature_norm | action_delta | local_density |
|------|-------------|-------------|--------------|
| **权重有区分度** | ❌ 全为 1.0 | ✅ | ✅ |
| **与空间位置正交** | ✅ | ❌ 空间扎堆 | ❌ 过度稀疏 |

**结论**：当前数据集上，**不加权的标准 FPS（0.004294）反而是最优的纯几何方法**。任何权重要么无区分度，要么引入负面偏差。

---

## 密度过滤 FPS 结果

| 配置 | Test MSE | vs 标准 FPS | 状态 |
|------|---------|-----------|------|
| 标准 FPS | 0.004294 | — | — |
| **密度过滤 FPS** | **0.003792** | **↓11.7%** | ✅ **全局最优** |

**关键发现**：
- 标准 FPS 选出的 1600 帧中，240 帧（15%）是极端稀疏的孤立点（密度 0.055~0.19）
- 替换为全局中密度更高的典型帧（密度 0.0026~0.0041）后，效果显著提升
- **密度过滤 FPS（0.003792）> v1.1 + PCA 400d（0.003835）**

**结论**：纯几何 FPS 框架很强，但"最远"不等于"最好"。简单密度后处理剔除了噪声离群点，效果超越复杂的 K-Means + PCA 组合。

---

## 与 v1.1 的递进关系

| 方法 | 核心机制 | Test MSE |
|------|---------|---------|
| 随机基线 | 无 | 0.007229 |
| 标准 FPS | 直接覆盖 | 0.004294 |
| v1.0 | K-Means + 稀疏惩罚 | 0.004066 |
| v1.1 | K-Means + 动态配额 | 0.003961 |
| v1.1+PCA400 | + PCA 去噪 | 0.003835 |
| **密度过滤 FPS** | **FPS + 剔除孤立噪声** | **0.003792** ✅ |

FPS 证明了"覆盖"本身的价值，密度过滤进一步证明"剔除噪声"的价值。两者结合超越了 v1.1 的三层叠加（K-Means + 动态配额 + PCA）。

---

## 文件结构

```
approaches/farthest_point_sampling/
├── README.md
├── src/
│   ├── selector_fps.py              # 标准 FPS
│   └── selector_weighted_fps.py     # 加权 FPS（多种策略）
├── scripts/
│   ├── run_fps.py                   # 标准 FPS 实验
│   └── run_weighted_fps.py          # 加权 FPS 实验
└── results/
    ├── fps_result_seed42.json
    ├── weighted_fps_feature_norm_result_seed42.json
    ├── weighted_fps_action_delta_result_seed42.json
    └── weighted_fps_local_density_result_seed42.json
```
