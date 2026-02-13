#!/usr/bin/env python3
"""2x2 comparison: darken, tint, hillshade+contours, hillshade only."""

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
    parse_neighborhoods, build_neighborhood_index,
    get_neighborhood_color_map, add_neighborhood_fills,
)
from process_elevation import load_elevation_from_file, add_neighborhood_contours
from process_terrain import add_filled_contours, add_hillshade
from shapely.ops import unary_union

# Load data once
with open('data/sf_neighborhoods.json', 'r') as f:
    neighborhoods = parse_neighborhoods(json.load(f))
elevation_data = load_elevation_from_file('data/sf_elevation.json')
names, polys, tree = build_neighborhood_index(neighborhoods)
city_boundary = unary_union(polys)
color_map, _ = get_neighborhood_color_map(neighborhoods, 8, palette='earthy')

DEM = 'data/sf_dem_10m_marin.tif'

styles = [
    dict(contour_style='darken', dem_path=None, linewidth=0.35, label='DARKEN'),
    dict(contour_style='tint', dem_path=None, linewidth=0.35, label='TINT'),
    dict(contour_style='darken', dem_path=DEM, linewidth=0.35, label='HILLSHADE + CONTOURS'),
    dict(contour_style='darken', dem_path=DEM, linewidth=0.0, label='HILLSHADE ONLY'),
]

fig, axes_grid = plt.subplots(2, 2, figsize=(44, 40))
axes_flat = axes_grid.flatten()

for i, cfg in enumerate(styles):
    label = cfg['label']
    print(f"Rendering {label}...")

    tmp_fig, tmp_ax = plt.subplots(figsize=(20, 18))

    # Terrain base
    if cfg['dem_path']:
        add_hillshade(tmp_ax, cfg['dem_path'], alpha=0.45)
    else:
        add_filled_contours(tmp_ax, elevation_data, city_boundary)

    # Neighborhood fills
    add_neighborhood_fills(tmp_ax, neighborhoods, color_map)

    # Contour lines
    if cfg['linewidth'] > 0:
        add_neighborhood_contours(tmp_ax, elevation_data, names, polys, tree, color_map,
                                  contour_style=cfg['contour_style'], linewidth=cfg['linewidth'])

    tmp_ax.set_aspect('equal')
    tmp_ax.set_xticks([])
    tmp_ax.set_yticks([])
    for spine in tmp_ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    tmp_fig.canvas.draw()
    img_array = np.asarray(tmp_fig.canvas.buffer_rgba())
    axes_flat[i].imshow(img_array)
    axes_flat[i].set_title(label, fontsize=40, fontweight='bold', pad=16)
    axes_flat[i].axis('off')
    plt.close(tmp_fig)

fig.tight_layout(pad=3)
fig.savefig('images/sf_contour_style_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved images/sf_contour_style_comparison.png")
