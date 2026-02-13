#!/usr/bin/env python3
"""Render a high-res comparison grid of all available color palettes."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from process_neighborhoods import (
    PALETTES, parse_neighborhoods, build_neighborhood_index,
    get_neighborhood_color_map, add_neighborhood_fills, add_smart_neighborhood_labels,
)
from process_elevation import load_elevation_from_file, add_neighborhood_contours
from process_terrain import add_filled_contours
from shapely.ops import unary_union

# Load data once
with open('data/sf_neighborhoods.json', 'r') as f:
    neighborhoods = parse_neighborhoods(json.load(f))
elevation_data = load_elevation_from_file('data/sf_elevation.json')
names, polys, tree = build_neighborhood_index(neighborhoods)
city_boundary = unary_union(polys)

palettes = list(PALETTES.keys())
n = len(palettes)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(60, rows * 18))
axes = axes.flatten()

for i, name in enumerate(palettes):
    print(f"Rendering palette: {name} ({i+1}/{n})")
    color_map, _ = get_neighborhood_color_map(neighborhoods, 8, palette=name)

    tmp_fig, tmp_ax = plt.subplots(figsize=(20, 18))
    add_filled_contours(tmp_ax, elevation_data, city_boundary)
    add_neighborhood_fills(tmp_ax, neighborhoods, color_map)
    add_neighborhood_contours(tmp_ax, elevation_data, names, polys, tree, color_map)
    add_smart_neighborhood_labels(tmp_ax, neighborhoods, list(range(len(neighborhoods))),
                                  lambda x: (0.5, 0.5, 0.5))
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
