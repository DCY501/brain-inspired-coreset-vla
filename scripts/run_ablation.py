"""
消融实验批量运行脚本
依次运行 v1(temporal_only)、v2(diversity_only)、v3(fusion)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_coreset import run_single_coreset
from config import RANDOM_SEED

MODES = ['temporal_only', 'diversity_only', 'fusion']

for mode in MODES:
    res = run_single_coreset(seed=RANDOM_SEED, mode=mode)
    print(f"\n[{mode}] Test MSE: {res['test_mse']:.6f}")

print("\n" + "=" * 60)
print("Ablation experiments completed!")
print("Run: python scripts/compare_results.py")
print("=" * 60)
