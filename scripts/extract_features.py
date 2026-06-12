"""
CLIP 多模态特征离线提取脚本（绕过 LeRobotDataset，直接读视频+parquet）
运行方式:
    cd scripts
    python extract_features.py

输出:
    data/features/clip_visual_features.npy  -- (20000, 768)
    data/features/clip_text_features.npy    -- (20000, 512)
    data/features/actions.npy               -- (20000, 7)
    data/features/episode_indices.npy       -- (20000,)
    data/features/task_names.npy            -- (20000,)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import pandas as pd
import av
from tqdm import tqdm

from config import *
from feature_extractor import CLIPFeatureExtractor


def load_parquet_data(data_root: str):
    """直接读取所有 parquet 文件，绕开 LeRobotDataset"""
    data_dir = os.path.join(data_root, "data", "chunk-000")
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    
    # 读取任务名称
    tasks_df = pd.read_parquet(os.path.join(data_root, "meta", "tasks.parquet"))
    task_names = tasks_df['task'].tolist() if 'task' in tasks_df.columns else ["unknown"]
    
    # 提取字段
    actions = np.stack(df['action'].values).astype(np.float32)       # (N, 14)
    ep_indices = df['episode_index'].values.astype(np.int64)         # (N,)
    task_idx = df['task_index'].values.astype(np.int64)              # (N,)
    
    # 单臂降维
    if USE_SINGLE_ARM:
        actions = actions[:, SINGLE_ARM_DIMS]  # (N, 7)
    
    # 任务文本
    task_texts = [task_names[i] if i < len(task_names) else task_names[0] for i in task_idx]
    
    return actions, ep_indices, task_texts


def load_video_frames(video_path: str):
    """用 av 逐帧读取视频，返回 numpy 数组列表"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    frames = []
    for frame in tqdm(container.decode(video=0), total=total_frames, desc="Loading video"):
        img = frame.to_rgb().to_ndarray()  # (H, W, 3), uint8
        frames.append(img)
    container.close()
    return frames


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}")
    
    # ---------- 加载 parquet 数据 ----------
    print("[Extract] Loading parquet data...")
    actions, ep_indices, task_texts = load_parquet_data(DATA_ROOT)
    num_frames = len(actions)
    print(f"[Extract] {num_frames} frames loaded from parquet")
    
    # ---------- 加载视频帧 ----------
    video_path = os.path.join(DATA_ROOT, "videos", IMAGE_KEY, "chunk-000", "file-000.mp4")
    print(f"[Extract] Loading video from {video_path}...")
    frames = load_video_frames(video_path)
    assert len(frames) == num_frames, f"Frame mismatch: video={len(frames)}, parquet={num_frames}"
    
    # ---------- 初始化 CLIP ----------
    extractor = CLIPFeatureExtractor(model_name=CLIP_MODEL_NAME, device=device)
    
    # 文本只需编码一次（单任务），然后复制
    print("[Extract] Encoding text (once)...")
    text_feat = extractor.extract_text([task_texts[0]])  # (1, 512)
    text_features = np.repeat(text_feat.numpy(), num_frames, axis=0)  # (N, 512)
    
    # ---------- 批量提取视觉特征 ----------
    all_visual = []
    batch_size = BATCH_SIZE_EXTRACT
    n_batches = (num_frames + batch_size - 1) // batch_size
    
    print(f"[Extract] Extracting visual features ({num_frames} frames, batch={batch_size})...")
    for i in tqdm(range(n_batches), desc="CLIP Visual"):
        start = i * batch_size
        end = min(start + batch_size, num_frames)
        
        # 构建图像 tensor (B, 3, H, W), uint8
        batch_frames = np.stack(frames[start:end])  # (B, H, W, 3), uint8
        batch_frames = torch.from_numpy(batch_frames).permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        v_feats = extractor.extract_visual(batch_frames)  # (B, 512)
        all_visual.append(v_feats.numpy())
    
    visual_features = np.concatenate(all_visual, axis=0)  # (N, 512)
    
    print(f"[Extract] Visual features: {visual_features.shape}")
    print(f"[Extract] Text features: {text_features.shape}")
    
    # ---------- 保存 ----------
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    np.save(os.path.join(FEATURES_DIR, "clip_visual_features.npy"), visual_features)
    np.save(os.path.join(FEATURES_DIR, "clip_text_features.npy"), text_features)
    np.save(os.path.join(FEATURES_DIR, "actions.npy"), actions)
    np.save(os.path.join(FEATURES_DIR, "episode_indices.npy"), ep_indices)
    np.save(os.path.join(FEATURES_DIR, "task_names.npy"), np.array(task_texts))
    
    print(f"[Saved] All features saved to {FEATURES_DIR}/")
    print("  - clip_visual_features.npy")
    print("  - clip_text_features.npy")
    print("  - actions.npy")
    print("  - episode_indices.npy")
    print("  - task_names.npy")


if __name__ == "__main__":
    main()
