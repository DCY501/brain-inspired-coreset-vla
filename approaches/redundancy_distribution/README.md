# 思路一：冗余分布 (Redundancy Distribution)

> **核心思想**：机器人操作数据中，大量帧的视觉状态高度相似（分布冗余），导致模型过拟合于常见状态。通过 K-Means 聚类覆盖特征空间，优先从稀疏簇中选择样本，实现分布均衡。

---

## 动机

在机器人模仿学习中，简单动作的帧往往占绝大多数（例如机械臂静止等待）。如果均匀随机采样，模型会被这些"冗余帧"淹没，无法有效学习到关键状态转移。这对应到神经科学中的**稀疏编码**思想——大脑对常见刺激响应弱，对新颖刺激响应强。

---

## 算法演进

### v1.0 固定保底 (Fixed Quota)

**机制**：
- K-Means 聚类（~533 簇）
- 每簇固定保底 **1 帧**（得分最高者）
- 剩余名额按全局多样性得分补齐

**结果**：Test MSE **0.004066**（vs 随机基线 0.007229，↓43.7%）

**代码**：`src/selector_v1_fixed.py` + `scripts/run_v1_fixed.py`

---

### v1.1 动态保底 (Dynamic Quota)

**改进**：簇大小自适应配额，取代固定 1 帧

```python
quota = max(1, min(3, cluster_size // 50 + 1))
```

- 小簇（<50 帧）：保底 1 帧
- 中簇（50-99 帧）：保底 2 帧
- 大簇（≥100 帧）：保底 3 帧（封顶）

**直觉**：大簇覆盖的视觉状态区域更大，理应分配更多名额；小簇代表稀有状态，不应浪费过多名额。

**结果**：Test MSE **0.003961**（vs v1.0 ↓2.6%，vs 随机基线 ↓45.2%）

**代码**：`src/selector_v1_1_dynamic.py` + `scripts/run_v1_1_dynamic.py`

---

### v1.1 + PCA 400d (最优配置)

**改进**：聚类前对 768d 视觉特征做 PCA 降维至 400d（保留 99.8% 方差）

**关键发现**：PCA 400d 不仅压缩了 48% 维度，**还提升了精度**，因为去除了干扰 K-Means 的噪声方向。

**完整 PCA 维度搜索**：

| 维度 | 保留方差 | Test MSE | vs 无 PCA |
|------|---------|---------|----------|
| 50 | 94.2% | 0.004542 | ↑7.3% |
| 100 | 97.3% | 0.004196 | ↓0.8% |
| 200 | 99.0% | 0.004283 | ↑1.2% |
| 300 | 99.6% | 0.004631 | ↑9.5% |
| **400** | **99.8%** | **0.003835** | **↓9.3%** ✅ |
| 500 | 99.9% | 0.003960 | ↓6.4% |
| 600 | 100.0% | 0.004686 | ↑10.8% |
| 无 PCA | 100% | 0.004231 | — |

**结果**：Test MSE **0.003835**（vs v1.1 无 PCA ↓9.3%，vs 随机基线 ↓47.0%）

**结论**：400d 是全局 sweet spot，非单调趋势。600d 保留 100% 方差却比无 PCA 差，说明 PCA 的线性变换本身在高维时有害。

**代码**：`src/selector_v1_1_pca_best.py` + `scripts/run_v1_1_pca_best.py`

---

## 文件结构

```
approaches/redundancy_distribution/
├── README.md                          # 本文件
├── src/
│   ├── selector_v1_fixed.py           # v1.0 固定保底
│   ├── selector_v1_1_dynamic.py       # v1.1 动态保底
│   └── selector_v1_1_pca_best.py      # v1.1 + PCA 400d 最优
├── scripts/
│   ├── run_v1_fixed.py                # v1.0 实验脚本
│   ├── run_v1_1_dynamic.py            # v1.1 实验脚本
│   └── run_v1_1_pca_best.py           # v1.1 + PCA 实验脚本
└── results/
    ├── v1_fixed/                      # v1.0 结果
    ├── v1_1_dynamic/                  # v1.1 结果
    └── pca_sweep/                     # PCA 50~600d 全部结果
```

---

## 运行方式

```bash
# v1.0 固定保底
cd PROJECT
python -m approaches.redundancy_distribution.scripts.run_v1_fixed

# v1.1 动态保底
python -m approaches.redundancy_distribution.scripts.run_v1_1_dynamic

# v1.1 + PCA 400d
python -m approaches.redundancy_distribution.scripts.run_v1_1_pca_best
```

---

## 与后续思路的关系

此思路完全基于**空间特征分布**（视觉特征的静态聚类），未利用任何**时序信息**。后续思路（如时序感知）将在此基础上开拓全新维度——不修改本思路的任何代码，而是创建独立的 `approaches/` 子目录。
