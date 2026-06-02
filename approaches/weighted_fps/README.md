# 思路四-A 改进：加权最远点采样 (Weighted FPS)

> 思路：在标准 FPS 的基础上引入质量权重，实现"覆盖 + 质量"双目标优化。

---

## 加权失败原因分析

### feature_norm 权重失效

**直接原因**：CLIP 视觉特征已全部归一化到单位球面上。

```
CLIP 视觉特征 L2 范数: min=1.000000, max=1.000000, std=0.000000
```

所有帧的 feature_norm 精确等于 1.0，权重无任何区分度。加权 FPS 实际退化为标准 FPS（只是初始帧从随机变成第 0 帧）。

**深层原因**：CLIP ViT-B/32 在提取视觉特征时已经做了 L2 归一化。这是 CLIP 模型的设计选择（对比学习需要归一化），不是数据特性。任何基于"特征范数"的策略在 CLIP 特征上都会失效。

### action_delta 权重失效

**直接原因**：加权 FPS 选出的帧和时序事件边界高度重叠——都是动作变化率大的"事件边界帧"。

**深层原因**：单任务数据集的时空耦合（50 个 episode 遵循同一脚本）。高变化率帧虽然在时序上分散，但在视觉空间中高度重叠。加权 FPS 用 action_delta 优先选这些帧，结果就是在相同的视觉区域反复采样，空间覆盖崩溃。

这和"时序事件边界 standalone 失败"是同一个根因。

### 加权 FPS 的有效条件

一个权重策略要有效，必须满足两个条件：

| 条件 | feature_norm | action_delta |
|------|-------------|-------------|
| **有区分度** | ❌ 全为 1.0 | ✅ 有区分度 |
| **与空间位置正交** | ✅ 完全正交 | ❌ 高变化率帧空间扎堆 |

**两个条件必须同时满足，缺一不可。**

### 如果换数据集，加权可能有效

- **多任务数据**（如 Open X-Embodiment）：不同任务的 feature_norm 可能不同（因为 CLIP 对不同任务的激活强度不同），action_delta 也可能与空间位置更正交
- **原始像素输入**（不用 CLIP）：feature_norm 自然有区分度

---

## 实验结果

### 加权 FPS 结果

| 策略 | Test MSE | vs 标准 FPS | 失败原因 |
|------|---------|-----------|---------|
| 标准 FPS | 0.004294 | — | — |
| feature_norm | 0.004162 | ↓3.1% | 权重无区分度（CLIP 已归一化） |
| action_delta | 0.007225 | ↑68.3% | 权重与空间位置不正交（时序扎堆） |

**feature_norm 的 0.004162 vs 0.004294**：差距 0.000132 纯粹是初始帧选择不同（随机 vs 第 0 帧）导致的随机波动，不是权重的效果。

---

## 结论

> **FPS 本身是好方法（0.004294），但加权策略在当前数据集上走不通。**
>
> 不是加权 FPS 的思想错了，而是**这个数据集没有提供有效的权重信号**。
>
> 后续如果要改进 FPS，应该回到纯几何角度（如 PCA+FPS、分层 FPS），而不是依赖权重。

---

## 文件结构

```
approaches/weighted_fps/
├── README.md
├── src/
│   └── selector_weighted_fps.py
├── scripts/
│   └── run_weighted_fps.py
└── results/
    ├── weighted_fps_feature_norm_result_seed42.json
    └── weighted_fps_action_delta_result_seed42.json
```

## 运行方式

```bash
cd PROJECT
python -m approaches.weighted_fps.scripts.run_weighted_fps
```
