"""
预测编码得分计算 (Predictive Coding Scorer)

核心思想（对应大脑预测编码理论）：
    大脑持续构建内部模型来预测下一时刻的感官输入。
    当实际输入与预测一致时，神经活动被抑制；
    当预测误差大（"惊奇"）时，该区域值得更多关注 -> 信息价值高。

实现方式：
    对每个 episode，用前 W 帧的视觉特征训练一个轻量 Ridge 回归模型，
    预测当前帧的动作。预测误差（MSE）即为该帧的"surprise"得分。
    误差越大 -> 该帧的动作模式越难以被历史视觉-动作关联预测 -> 信息价值越高。
"""
import numpy as np
from sklearn.linear_model import Ridge


class PredictiveCodingScorer:
    """
    基于滑动窗口预测编码的时序重要性得分计算器
    """
    def __init__(self, window_size: int = 7, alpha: float = 1.0, mode: str = 'strict'):
        """
        Args:
            window_size: 滑动窗口大小，用前 W 帧预测当前帧
            alpha: Ridge 回归正则化强度
            mode: 'strict' = 逐帧 leave-one-out 预测（慢但更准）；
                  'fast'   = 全局 Ridge 残差（快但略有信息泄漏）
        """
        self.window_size = window_size
        self.alpha = alpha
        self.mode = mode
    
    def _build_window_features(self, features: np.ndarray, actions: np.ndarray):
        """
        将 episode 的时序数据构造成 (X_window, y_target) 样本对
        
        Args:
            features: (L, D) 视觉特征
            actions:  (L, A) 动作向量
        Returns:
            X: (N, window_size * D)  窗口化特征
            y: (N, A)                对应目标动作
            frame_indices: (N,)      每个样本对应的原始帧索引（目标帧位置）
        """
        L = len(features)
        if L < self.window_size + 1:
            return None, None, None
        
        X, y, frame_indices = [], [], []
        for t in range(self.window_size, L):
            X.append(features[t - self.window_size:t].flatten())
            y.append(actions[t])
            frame_indices.append(t)
        
        return np.stack(X), np.stack(y), np.array(frame_indices, dtype=np.int64)
    
    def compute_scores(self, features: np.ndarray, actions: np.ndarray, episode_indices: np.ndarray):
        """
        计算每个帧的预测编码 surprise 得分
        
        Args:
            features: (N, D) 视觉特征
            actions:  (N, A) 动作向量
            episode_indices: (N,) episode 编号
        Returns:
            scores: (N,) 值域 [0, 1]，越高表示预测误差越大、信息价值越高
        """
        n = len(features)
        scores = np.zeros(n, dtype=np.float32)
        
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            idx_in_ep = np.where(mask)[0]
            ep_features = features[mask]
            ep_actions = actions[mask]
            L = len(ep_features)
            
            if L < self.window_size + 1:
                # 太短的 episode：无法建立预测模型，赋予中等得分保留
                scores[idx_in_ep] = 0.5
                continue
            
            X, y, frame_idx = self._build_window_features(ep_features, ep_actions)
            if X is None:
                scores[idx_in_ep] = 0.5
                continue
            
            n_samples = len(X)
            ep_scores = np.zeros(L, dtype=np.float32)
            ep_scores[:self.window_size] = np.nan  # 占位，稍后填充
            
            if self.mode == 'fast':
                # ====== 快速模式：全局 Ridge + 残差 ======
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = np.mean((y_pred - y) ** 2, axis=1)  # (N,)
                ep_scores[self.window_size:] = residuals
            
            else:
                # ====== 严格模式：逐帧 leave-one-out（时间序列交叉验证）======
                # 对每个目标帧 t，用 [window_size, t) 的历史窗口训练，预测 t
                for i in range(n_samples):
                    t = frame_idx[i]  # 当前目标帧在 episode 中的索引
                    
                    # 训练集：当前目标帧之前的所有窗口（严格 leave-one-out）
                    if i < 3:
                        # 样本太少，跳过，后面用插值填充
                        ep_scores[t] = np.nan
                        continue
                    
                    X_train = X[:i]
                    y_train = y[:i]
                    x_test = X[i].reshape(1, -1)
                    y_test = y[i]
                    
                    model = Ridge(alpha=self.alpha, fit_intercept=True)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(x_test)[0]
                    
                    surprise = np.mean((y_pred - y_test) ** 2)
                    ep_scores[t] = surprise
                
                # 插值填充无法计算的帧（前 window_size + 前几帧）
                valid_mask = ~np.isnan(ep_scores)
                if valid_mask.any():
                    median_val = np.median(ep_scores[valid_mask])
                    ep_scores[~valid_mask] = median_val
            
            # 归一化到 [0, 1]
            valid = ep_scores[self.window_size:]
            if valid.max() > valid.min():
                ep_scores = (ep_scores - ep_scores.min()) / (ep_scores.max() - ep_scores.min() + 1e-8)
            
            scores[idx_in_ep] = ep_scores
        
        return scores
