#!/usr/bin/env python3
"""2x2 comparison: darken, tint, hillshade+contours, hillshade only."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from process_neighborhoods import parse_neighborhoods
from process_elevation import load_elevation_from_file
from process_combined import add_combined_visualization_to_axis

styles = [
    dict(contour_style='darken', dem_path=None, elevation_linewidth=0.35, label='DARKEN'),
    dict(contour_style='tint', dem_path=None, elevation_linewidth=0.35, label='TINT'),
    dict(contour_style='darken', dem_path='data/sf_dem_10m_marin.tif', elevation_linewidth=0.35, label='HILLSHADE + CONTOURS'),
    dict(contour_style='darken', dem_path='data/sf_dem_10m_marin.tif', elevation_linewidth=0.0, label='HILLSHADE ONLY'),
]

# Load data once
with open('data/sf_neighborhoods.json', 'r') as f:
    neighborhoods = parse_neighborhoods(json.load(f))
elevation_data = load_elevation_from_file('data/sf_elevation.json')

fig, axes = plt.subplots(2, 2, figsize=(44, 40))
axes = axes.flatten()

for i, cfg in enumerate(styles):
    label = cfg.pop('label')
    print(f"Rendering {label}...")
    tmp_fig, tmp_ax = plt.subplots(figsize=(20, 18))
    add_combined_visualization_to_axis(
        tmp_ax,
        neighborhoods,
        elevation_data,
        max_colors=8,
        show_neighborhood_labels=True,
        palette='earthy',
        **cfg,
    )
    tmp_ax.set_aspect('equal')
    tmp_ax.set_xticks([])
    tmp_ax.set_yticks([])
    for spine in tmp_ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    tmp_fig.canvas.draw()
    img_array = np.asarray(tmp_fig.canvas.buffer_rgba())
    axes[i].imshow(img_array)
    axes[i].set_title(label, fontsize=40, fontweight='bold', pad=16)
    axes[i].axis('off')
    plt.close(tmp_fig)

fig.tight_layout(pad=3)
fig.savefig('images/sf_contour_style_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved images/sf_contour_style_comparison.png")
