#!/usr/bin/env python3
"""Render poster-quality versions of all palettes."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from process_neighborhoods import PALETTES, parse_neighborhoods
from process_elevation import load_elevation_from_file
from process_combined import add_combined_visualization_to_axis

# Load data once
with open('data/sf_neighborhoods.json', 'r') as f:
    neighborhoods = parse_neighborhoods(json.load(f))
elevation_data = load_elevation_from_file('data/sf_elevation.json')

for name in PALETTES:
    print(f"=== Rendering {name} ===")
    fig, ax = plt.subplots(figsize=(32, 28))
    add_combined_visualization_to_axis(
        ax,
        neighborhoods,
        elevation_data,
        max_colors=8,
        show_neighborhood_labels=True,
        palette=name,
    )
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
