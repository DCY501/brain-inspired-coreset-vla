"""
PCA 维度扫描脚本
依次运行 v1.1 + PCA [None, 50, 100, 200, 300, 400]
确保 torch.manual_seed 固定，结果可复现
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'src', 'config.py')

DIMS = [None, 50, 100, 200, 300, 400]

for dim in DIMS:
    # 修改 config.py
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if dim is None:
        new_line = "PCA_N_COMPONENTS = None                   # PCA 降维维度"
    else:
        new_line = f"PCA_N_COMPONENTS = {dim:<4d}                   # PCA 降维维度"
    
    # 替换 PCA_N_COMPONENTS 行
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'PCA_N_COMPONENTS' in line:
            lines[i] = new_line
            break
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    tag = f"pca{dim}" if dim else "nopca"
    print(f"\n{'='*60}")
    print(f"Running: v1.1 + {tag}")
    print('='*60)
    
    # 运行实验
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'run_coreset.py')],
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print(f"[ERROR] {tag} failed!")
        break

print("\n" + "="*60)
print("All PCA sweep experiments completed!")
print("="*60)
