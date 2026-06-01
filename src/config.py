"""
全局配置文件
包含路径、超参数、实验设置等常量
"""
import os

# ==================== 路径配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

for d in [DATA_ROOT, PROCESSED_DIR, FEATURES_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 数据集配置 ====================
REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"

USE_SINGLE_ARM = True
SINGLE_ARM_DIMS = list(range(7))

# ==================== CLIP 特征提取配置 ====================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VISUAL_FEATURE_DIM = 768
TEXT_FEATURE_DIM = 512
FEATURE_DIM = 1280
BATCH_SIZE_EXTRACT = 16

# ==================== MLP 模型与训练配置 ====================
MLP_HIDDEN_DIMS = [512, 256, 128]
MLP_DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE_TRAIN = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# ==================== 实验配置 ====================
RANDOM_SEED = 42
TEST_EPISODE_RATIO = 0.2
BASELINE_SAMPLE_RATIO = 0.1
CORESET_RATIO = 0.1
PCA_N_COMPONENTS = 400                    # PCA 降维维度
