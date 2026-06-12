#!/usr/bin/env python3
"""Convert SVG flowcharts to PNG for LaTeX inclusion."""
import os
import cairosvg

PHOTO_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'photo')

svgs = [
    'figA_kmeans.svg',
    'figB_fps.svg',
    'figC_predictive_coding.svg',
    'figE_fusion_action_aware.svg',
    'figE_fusion_complementary.svg',
    'figE_fusion_hierarchical.svg',
    'figE_fusion_score.svg',
    'figE_fusion_semantic_density.svg',
]

for svg_name in svgs:
    svg_path = os.path.join(PHOTO_DIR, svg_name)
    png_name = svg_name.replace('.svg', '.png')
    png_path = os.path.join(PHOTO_DIR, png_name)
    
    if not os.path.exists(svg_path):
        print(f"SKIP: {svg_path} not found")
        continue
    
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=1200)
    print(f"OK: {svg_name} -> {png_name}")

print("Done.")
