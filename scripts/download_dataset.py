"""
数据集下载脚本
调用 lerobot 库从 HuggingFace 下载 ALOHA Sim Transfer Cube 数据集
"""
import os
import sys

# 将 project/src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import DATA_ROOT, REPO_ID
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("=" * 60)
    print("Downloading ALOHA Sim Transfer Cube Dataset")
    print("=" * 60)
    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f"Target directory: {DATA_ROOT}")
    
    dataset = LeRobotDataset(repo_id=REPO_ID, root=DATA_ROOT)
    print(f"\nDataset loaded successfully!")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")


if __name__ == "__main__":
    main()
