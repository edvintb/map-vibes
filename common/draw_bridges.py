#!/usr/bin/env python3
"""
Shared bridge-drawing utilities: catenary cable physics and side-elevation profiles.

Used by both San Francisco and New York map generators.
"""

import logging

import matplotlib.patheffects as pe
import numpy as np

logger = logging.getLogger(__name__)

# Bridge scaling constants
TOWER_RATIO = 0.1567   # peak_h / bridge_length (Bay Bridge calibration)
RODS_PER_INCH = 40     # suspender rods per rendered inch of bridge


def _cable_height(t, towers, peak_h):
    """Compute catenary cable height at fractional position t along the bridge."""
    a = 2.0
    def _catenary_rise(s, h_target):
        """Normalized cosh curve from 0 at s=0 to h_target at s=1."""
        return h_target * (np.cosh(a * s) - 1) / (np.cosh(a) - 1)
    pts = [(0.0, 0.0)] + list(towers) + [(1.0, 0.0)]
    for i in range(len(pts) - 1):
        f0, h0 = pts[i]
        f1, h1 = pts[i + 1]
        if f0 <= t <= f1:
            span = f1 - f0
            if span == 0:
                return h0
            s = (t - f0) / span
            if h0 < h1:
                return _catenary_rise(s, h1)
            elif h0 > h1:
                return _catenary_rise(1 - s, h0)
            else:
                return h0 * (2 * s - 1) ** 2
    return 0.0


def _make_perp(start, end):
    """Return unit perpendicular vector (pointing upward) from start to end."""
    d = (end - start) / np.linalg.norm(end - start)
    perp = np.array([-d[1], d[0]])
    if perp[1] < 0:
        perp = -perp
    return perp


def _draw_bridge(ax, start, end, perp, tower_fracs, peak_h, n_rods,
                 color, zorder, lw_scale):
    """Render a single bridge: deck line, catenary cables, and suspender rods."""
    vec = end - start
    towers = [(f, peak_h) for f in tower_fracs]
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, lw=1.6 * lw_scale, zorder=zorder,
            solid_capstyle="butt")
    t_arr = np.linspace(0, 1, 300)
    h_arr = np.array([_cable_height(t, towers, peak_h) for t in t_arr])
    cable_pts = (start[np.newaxis, :] +
                 np.outer(t_arr, vec) +
                 np.outer(h_arr, perp))
    ax.plot(cable_pts[:, 0], cable_pts[:, 1],
            color=color, lw=0.7 * lw_scale, zorder=zorder + 1, alpha=0.85)
    for i in range(n_rods):
        ti = (i + 1) / (n_rods + 1)
        deck_pt = start + vec * ti
        h = _cable_height(ti, towers, peak_h)
        cable_pt = deck_pt + perp * h
        ax.plot([deck_pt[0], cable_pt[0]], [deck_pt[1], cable_pt[1]],
                color=color, lw=0.25 * lw_scale, zorder=zorder + 1,
                alpha=0.5)


def draw_bridges(ax, bridges, bridge_labels=False):
    """Draw side-elevation bridge profiles from a list of bridge dicts.

    Each bridge dict should have: name, start, end, tower_fracs, color, label_color.
    Optional: label_side (+1 above, -1 below, default +1), label_offset (dx, dy).

    All visual sizes derive from the axes extent and figure dimensions
    so that changing figsize or DEM extent scales automatically.
    """
    fig = ax.get_figure()
    fig_h = fig.get_size_inches()[1]
    ylim = ax.get_ylim()
    deg_per_inch = (ylim[1] - ylim[0]) / fig_h
    lw_scale = fig_h / 18.0
    label_gap = 0.25 * deg_per_inch

    zz = 9
    label_style = dict(
        fontweight="bold", fontfamily="sans-serif",
        path_effects=[pe.withStroke(linewidth=1.0 * lw_scale,
                                    foreground="white")],
        zorder=zz + 5, linespacing=1.1,
    )

    for b in bridges:
        start = np.array(b["start"])
        end = np.array(b["end"])
        perp = _make_perp(start, end)
        length = np.linalg.norm(end - start)
        peak_h = TOWER_RATIO * length
        bridge_inches = length / deg_per_inch
        n_rods = max(6, round(RODS_PER_INCH * bridge_inches))

        _draw_bridge(ax, start, end, perp,
                     tower_fracs=b["tower_fracs"],
                     peak_h=peak_h, n_rods=n_rods,
                     color=b["color"], zorder=zz, lw_scale=lw_scale)

        if bridge_labels:
            side = b.get("label_side", 1)
            fontsize = max(3.5, min(5.5, 3.5 + 2.0 * bridge_inches)) * lw_scale
            lp = (start + end) / 2 + perp * label_gap * side
            offset = b.get("label_offset", (0.0, 0.0))
            lp[0] += offset[0]
            lp[1] += offset[1]
            va = "bottom" if side > 0 else "top"
            ax.text(lp[0], lp[1], b["name"],
                    ha="center", va=va,
                    fontsize=fontsize, color=b["label_color"],
                    **label_style)

    logger.info(f"  Added bridges{' with labels' if bridge_labels else ''}")
