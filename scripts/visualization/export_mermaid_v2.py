"""
Mermaid 在线导出脚本 v2 - 使用 kroki.io API
支持 PNG 和 SVG 导出
"""
import os
import base64
import zlib
import urllib.request
import json

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
PHOTO_DIR = os.path.join(PROJECT_ROOT, 'report', 'photo')


def export_via_kroki(filepath, fmt='png'):
    """调用 kroki.io API 导出"""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    # kroki 使用 deflate + base64 编码
    compressed = zlib.compress(code.encode('utf-8'))
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')

    url = f"https://kroki.io/mermaid/{fmt}/{encoded}"
    out_path = filepath.replace('.mmd', f'_kroki.{fmt}')

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(out_path, 'wb') as f:
            f.write(data)
        print(f"  [OK] {os.path.basename(out_path)} ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {os.path.basename(out_path)}: {e}")
        return False


def main():
    files = [
        'figA_kmeans.mmd',
        'figB_fps.mmd',
        'figE_fusion_score.mmd',
        'figE_fusion_hierarchical.mmd',
        'figE_fusion_complementary.mmd',
        'figE_fusion_action_aware.mmd',
        'figE_fusion_semantic_density.mmd',
    ]

    print("=" * 60)
    print("Mermaid 在线导出 v2 (kroki.io) → PNG + SVG")
    print("=" * 60)

    for name in files:
        filepath = os.path.join(PHOTO_DIR, name)
        if not os.path.exists(filepath):
            print(f"\n[SKIP] 文件不存在: {name}")
            continue

        print(f"\n[{name}]")
        export_via_kroki(filepath, 'png')
        export_via_kroki(filepath, 'svg')

    print("\n" + "=" * 60)
    print("导出完成，文件保存在:", PHOTO_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
