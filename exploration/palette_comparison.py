#!/usr/bin/env python3
"""Render a high-res comparison grid of all available color palettes."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from process_neighborhoods import PALETTES, parse_neighborhoods
from process_elevation import load_elevation_from_file
from process_combined import add_combined_visualization_to_axis

# Load data once
with open('data/sf_neighborhoods.json', 'r') as f:
    neighborhoods = parse_neighborhoods(json.load(f))
elevation_data = load_elevation_from_file('data/sf_elevation.json')

palettes = list(PALETTES.keys())
n = len(palettes)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(60, rows * 18))
axes = axes.flatten()

for i, name in enumerate(palettes):
    print(f"Rendering palette: {name} ({i+1}/{n})")
    tmp_fig, tmp_ax = plt.subplots(figsize=(20, 18))
    add_combined_visualization_to_axis(
        tmp_ax,
        neighborhoods,
        elevation_data,
        max_colors=8,
        show_neighborhood_labels=True,
        palette=name,
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
    axes[i].set_title(name.upper(), fontsize=40, fontweight='bold', pad=16)
    axes[i].axis('off')
    plt.close(tmp_fig)

for j in range(n, len(axes)):
    axes[j].axis('off')

fig.tight_layout(pad=3)
fig.savefig('images/sf_palette_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved images/sf_palette_comparison.png")
