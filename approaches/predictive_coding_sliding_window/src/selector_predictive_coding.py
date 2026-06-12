"""
滑动窗口自回归预测编码核心集选择器 (Predictive Coding Sliding Window)

思路：大脑本质上是一个预测机器。神经元持续对世界进行前向预测，
只有当感官输入打破预测时才会产生显著的信号更新。预测编码理论认为：
预测误差小的帧 = 大脑已经学会的模式 = 时序冗余。

机制：
  1. 在每个 episode 内，用滑动窗口的历史动作自回归预测当前帧动作
  2. 线性回归拟合窗口动作 → 当前动作的关系
  3. 计算每帧的预测残差（MSE）
  4. 残差小的帧 = 可预测 = 时序冗余，残差大的帧 = 打破预测 = 高价值
  5. 选择残差最大的帧构成核心集

关键设计：
  - 每个 episode 独立拟合（避免跨 episode 泄露）
  - 前 window 帧无法计算预测误差，设为高值保留
  - 线性模型容量有限，误差大小仍能反映可预测性
"""
import numpy as np
from config import *


class PredictiveCodingSelector:
    """
    滑动窗口自回归预测编码选择器
    基于预测误差大小识别时序冗余帧
    """
    def __init__(self, target_ratio: float = CORESET_RATIO,
                 window: int = 5,
                 seed: int = 42,
                 fusion_mode: str = 'single'):
        """
        Args:
            target_ratio: 目标筛选比例
            window: 滑动窗口大小
            seed: 随机种子（本方法确定性，保留接口一致性）
            fusion_mode: 融合模式
                'single' - 单尺度预测误差
                'multi_scale' - 多尺度融合（已废弃）
                'residual_delta' - 预测误差 × 动作变化率联合（方案C）
        """
        self.target_ratio = target_ratio
        self.window = window
        self.seed = seed
        self.fusion_mode = fusion_mode
    
    def _compute_residuals_for_scale(self, actions, episode_indices, window):
        """计算单一尺度的预测残差"""
        n = len(actions)
        residuals = np.zeros(n)
        
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            idx = np.where(mask)[0]
            ep_actions = actions[idx]
            n_ep = len(ep_actions)
            
            if n_ep <= window + 1:
                residuals[idx] = 1e6
                continue
            
            X_list = []
            y_list = []
            for t in range(window, n_ep):
                X_list.append(ep_actions[t - window:t].flatten())
                y_list.append(ep_actions[t])
            
            X_win = np.array(X_list)
            y_tgt = np.array(y_list)
            
            XtX = X_win.T @ X_win
            Xty = X_win.T @ y_tgt
            reg = 1e-6 * np.eye(XtX.shape[0])
            w = np.linalg.solve(XtX + reg, Xty)
            
            y_pred = X_win @ w
            errors = np.mean((y_pred - y_tgt) ** 2, axis=1)
            
            residuals[idx[:window]] = 1e6
            residuals[idx[window:]] = errors
        
        return residuals
    
    def _compute_action_deltas(self, actions: np.ndarray, episode_indices: np.ndarray):
        """计算动作变化率（跨episode边界重置为0）"""
        n = len(actions)
        deltas = np.zeros(n)
        
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            idx = np.where(mask)[0]
            if len(idx) <= 1:
                continue
            ep_actions = actions[idx]
            d = np.zeros(len(idx))
            d[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
            deltas[idx] = d
        
        return deltas
    
    def select(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        执行预测编码核心集选择
        
        Args:
            features: np.ndarray, shape=(N, D), 视觉特征（本方法不使用，保留接口）
            actions: np.ndarray, shape=(N, 7), 动作向量
            episode_indices: np.ndarray, shape=(N,), episode 编号
        Returns:
            selected: np.ndarray, 被选中的帧索引
        """
        n = len(actions)
        target_size = max(1, int(n * self.target_ratio))
        
        if self.fusion_mode == 'residual_delta':
            # 方案C：预测误差 × 动作变化率联合
            residuals = self._compute_residuals_for_scale(
                actions, episode_indices, self.window)
            deltas = self._compute_action_deltas(actions, episode_indices)
            
            print(f"[Predictive Coding] Residual range: {residuals[residuals < 1e5].min():.6f} ~ "
                  f"{residuals[residuals < 1e5].max():.6f}")
            print(f"[Predictive Coding] Delta range: {deltas.min():.6f} ~ {deltas.max():.6f}")
            
            # 归一化后相乘（避免量纲差异）
            r_norm = np.zeros_like(residuals)
            mask_r = residuals < 1e5
            if mask_r.sum() > 0:
                r_min, r_max = residuals[mask_r].min(), residuals[mask_r].max()
                r_norm[mask_r] = (residuals[mask_r] - r_min) / (r_max - r_min + 1e-8)
                r_norm[~mask_r] = 1.0
            
            d_norm = np.zeros_like(deltas)
            if deltas.max() > 0:
                d_norm = deltas / (deltas.max() + 1e-8)
            
            # 联合得分：需要同时满足"难预测"和"在变化"
            prediction_errors = r_norm * d_norm
            print(f"[Predictive Coding] Fusion mode: residual × delta (normalized)")
            
        elif self.fusion_mode == 'multi_scale':
            # 多尺度融合（已废弃，保留代码）
            all_residuals = []
            for scale in [3, 5, 10]:
                r = self._compute_residuals_for_scale(actions, episode_indices, scale)
                all_residuals.append(r)
            normalized_residuals = []
            for r in all_residuals:
                mask = r < 1e5
                if mask.sum() > 0:
                    r_min, r_max = r[mask].min(), r[mask].max()
                    r_norm = np.zeros_like(r)
                    r_norm[mask] = (r[mask] - r_min) / (r_max - r_min + 1e-8)
                    r_norm[~mask] = 1.0
                else:
                    r_norm = r
                normalized_residuals.append(r_norm)
            prediction_errors = np.maximum.reduce(normalized_residuals)
            print(f"[Predictive Coding] Multi-scale fusion (normalized max)")
        else:
            # 单尺度
            prediction_errors = self._compute_residuals_for_scale(
                actions, episode_indices, self.window)
            print(f"[Predictive Coding] Single-scale with window={self.window}")
        
        # 按得分排序，选高分帧
        selected_idx = np.argsort(prediction_errors)[-target_size:]
        selected = np.array(sorted(selected_idx), dtype=np.int64)
        
        print(f"[Predictive Coding] Selected {len(selected)} frames")
        print(f"[Predictive Coding] Score range: "
              f"{prediction_errors[selected].min():.6f} ~ {prediction_errors[selected].max():.6f}")
        print(f"[Predictive Coding] Coverage: {len(np.unique(episode_indices[selected]))}/"
              f"{len(np.unique(episode_indices))} episodes")
        
        return selected
