#!/usr/bin/env python3
"""Render poster-quality versions of all palettes."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

for name in PALETTES:
    print(f"=== Rendering {name} ===")
    color_map, _ = get_neighborhood_color_map(neighborhoods, 8, palette=name)

    fig, ax = plt.subplots(figsize=(32, 28))
    add_filled_contours(ax, elevation_data, city_boundary)
    add_neighborhood_fills(ax, neighborhoods, color_map)
    add_neighborhood_contours(ax, elevation_data, names, polys, tree, color_map)
    add_smart_neighborhood_labels(ax, neighborhoods, list(range(len(neighborhoods))),
                                  lambda x: (0.5, 0.5, 0.5), font_scale=2.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    save_path = f'images/sf_combined_{name}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)

print("Done — all palettes rendered.")
