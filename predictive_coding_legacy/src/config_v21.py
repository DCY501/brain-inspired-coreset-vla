"""
全局配置文件
包含路径、超参数、实验设置等常量
"""
import os

# ==================== 路径配置 ====================
# 项目根目录 (project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")          # 原始 lerobot 数据集
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")  # 模型权重、实验结果
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")    # 离线提取的视觉特征
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")                # 图表、对比报告

# 确保目录存在
for d in [DATA_ROOT, PROCESSED_DIR, FEATURES_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 数据集配置 ====================
REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
IMAGE_KEY = "observation.images.top"      # 单视角：顶视角相机
STATE_KEY = "observation.state"           # 状态向量
ACTION_KEY = "action"                     # 动作向量
TASK_KEY = "task"                         # 语言指令字段

# 动作降维配置：取前 7 维作为单臂（左臂）
USE_SINGLE_ARM = True
SINGLE_ARM_DIMS = list(range(7))          # 左臂: 0~6；若用右臂改为 list(range(7, 14))

# ==================== CLIP 特征提取配置 ====================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # CLIP 预训练模型
VISUAL_FEATURE_DIM = 768                  # CLIP ViT-B/32 视觉编码器 pooler_output 维度
TEXT_FEATURE_DIM = 512                    # CLIP 文本编码器输出维度（完整保留，与多任务VLA一致）
FEATURE_DIM = 1280                        # 总输入维度 = 768 + 512
BATCH_SIZE_EXTRACT = 16                   # 特征提取批次（CLIP 比 ResNet 大，适当减小）

# ==================== MLP 模型与训练配置 ====================
MLP_HIDDEN_DIMS = [512, 256, 128]         # 隐藏层维度
MLP_DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE_TRAIN = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15              # 验证集不改善的容忍轮数

# ==================== 实验配置 ====================
RANDOM_SEED = 42
TEST_EPISODE_RATIO = 0.2                  # 20% episode 作为固定测试集
BASELINE_SAMPLE_RATIO = 0.1               # Baseline: 随机抽 10% 训练 episode
CORESET_RATIO = 0.1                       # Coreset: 筛选 10% 训练帧

# ==================== 核心集选择算法配置 ====================
TEMPORAL_WEIGHT = 0.6                     # 时序得分权重 (Predictive Coding)
DIVERSITY_WEIGHT = 0.4                    # 分布得分权重 (RAS + 均衡)

# ==================== 预测编码配置 (v2.1) ====================
USE_PREDICTIVE_CODING = True              # 是否启用真正的滑动窗口预测编码
PREDICTIVE_CODING_WINDOW = 3              # 滑动窗口大小（帧）
PREDICTIVE_CODING_ALPHA = 1.0             # Ridge 回归正则化强度
PREDICTIVE_CODING_MODE = 'fast'           # 'strict'=逐帧 leave-one-out; 'fast'=全局残差
