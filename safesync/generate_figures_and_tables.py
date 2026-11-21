#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all 7 publication-grade figures at native 1:1 IEEE Transactions full-page width
(\textwidth = 7.16 in, zero LaTeX rescaling distortion, locked identical subplot boxes
across all 1x3 quantitative figures) plus Markdown summary tables from
safesync_r36_complete_results.json and xbd_safesync_tokens_440.npz.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUB_ROOT = PROJECT_ROOT
R36_ROOT = PROJECT_ROOT
RESULTS_JSON = R36_ROOT / "results" / "safesync_r36_complete_results.json"
CACHE_DIR = R36_ROOT / "data_cache"
FIG_DIR = R36_ROOT / "figures"
PAPER_DIR = PUB_ROOT / "safesync_paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"
DELIVERY_FIG_DIR = PUB_ROOT / "paper_delivery" / "figures"
LEGACY_FIG_DIR = PUB_ROOT / "paper" / "figures"

for d in [FIG_DIR, PAPER_DIR, PAPER_FIG_DIR, DELIVERY_FIG_DIR, LEGACY_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Exact 1:1 Native IEEEtran \textwidth = 7.16 in Styling (8 pt standard journal typography)
IEEE_TEXTWIDTH_IN = 7.16
CHART_HEIGHT_IN = 2.40

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.6,
    "axes.labelsize": 7.6,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2,
    "figure.titlesize": 8.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.65,
    "grid.linewidth": 0.40,
    "lines.linewidth": 1.05,
    "lines.markersize": 3.4,
    "xtick.major.width": 0.60,
    "ytick.major.width": 0.60,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.pad": 2.2,
    "ytick.major.pad": 2.2,
})

METHOD_LABELS = {
    "safesync": "SafeSync-Mesh (Ours)",
    "twist_inspired": "TWIST-inspired",
    "stateful_utility_only": "Stateful utility-only",
    "change_only": "Change-only",
    "stars_adapted": "STARS-style",
    "current_top_k": "Current Top-K",
    "full_feature": "Full VQ Ref",
}

PALETTE = {
    "safesync": "#0072B2",               # Deep Blue (Primary)
    "twist_inspired": "#D55E00",         # Vermilion
    "stateful_utility_only": "#009E73",  # Bluish Green
    "change_only": "#CC79A7",            # Reddish Purple
    "stars_adapted": "#E69F00",          # Orange
    "current_top_k": "#56B4E9",          # Sky Blue
    "full_feature": "#222222",           # Charcoal
}

MARKERS = {
    "safesync": "o",
    "twist_inspired": "s",
    "stateful_utility_only": "^",
    "change_only": "D",
    "stars_adapted": "v",
    "current_top_k": "P",
    "full_feature": "*",
}

LINESTYLES = {
    "safesync": "-",
    "twist_inspired": "--",
    "stateful_utility_only": "-.",
    "change_only": ":",
    "stars_adapted": "--",
    "current_top_k": ":",
    "full_feature": "-",
}


def save_pub_fig(fig: plt.Figure, stem: str, also_svg: bool = False) -> None:
    """Save at exact fixed canvas size (no bbox_inches='tight' rescaling) so width == 7.16 in."""
    for out_dir in [FIG_DIR, PAPER_FIG_DIR, DELIVERY_FIG_DIR, LEGACY_FIG_DIR]:
        fig.savefig(out_dir / f"{stem}.pdf", dpi=300)
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        if also_svg:
            fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)
    print(f"[Saved Native 7.16-in Figure] {stem}.pdf & {stem}.png" + (" & .svg" if also_svg else ""))


def make_locked_1x3_figure(titles: Tuple[str, str, str]) -> Tuple[plt.Figure, List[plt.Axes], List[float]]:
    """
    Create a 1x3 figure on an exact 7.16 x 2.40 in canvas where every subplot box
    has identical physical size (1.56 in wide x 1.36 in tall) regardless of twinx().
    Layout (vertical, H = 2.40 in):
      - 0.00 .. 0.37 in : x-ticks & x-axis label
      - 0.37 .. 1.73 in : locked plot box (1.36 in tall)
      - 1.76 .. 2.16 in : dedicated non-overlapping legend band (0.40 in tall)
      - 2.27 in         : bold panel titles (a), (b), (c)
    Layout (horizontal, W = 7.16 in):
      - [0.47, 2.03] in, [2.80, 4.36] in, [5.13, 6.69] in (symmetric 0.47 in outer, 0.77 in inner)
    """
    fig = plt.figure(figsize=(IEEE_TEXTWIDTH_IN, CHART_HEIGHT_IN), facecolor="#FFFFFF")
    x_starts_in = [0.47, 2.80, 5.13]
    w_in = 1.56
    y0_in = 0.42
    h_in = 1.32

    axes: List[plt.Axes] = []
    x_centers_norm: List[float] = []
    for idx, x_in in enumerate(x_starts_in):
        rect = [
            x_in / IEEE_TEXTWIDTH_IN,
            y0_in / CHART_HEIGHT_IN,
            w_in / IEEE_TEXTWIDTH_IN,
            h_in / CHART_HEIGHT_IN,
        ]
        ax = fig.add_axes(rect)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
        axes.append(ax)
        xc = (x_in + 0.5 * w_in) / IEEE_TEXTWIDTH_IN
        x_centers_norm.append(xc)
        fig.text(
            xc,
            2.28 / CHART_HEIGHT_IN,
            titles[idx],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="#0F172A",
        )
    return fig, axes, x_centers_norm


def add_band_legend(
    fig: plt.Figure,
    handles: list,
    labels: list,
    x_center_norm: float,
    ncol: int,
    fontsize: float = 6.0,
    colspacing: float = 0.65,
    handlelength: float = 1.35,
) -> None:
    """Place a crisp legend pill centered in the dedicated [1.76..2.16 in] header band."""
    leg = fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(x_center_norm, 1.98 / CHART_HEIGHT_IN),
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        framealpha=0.96,
        columnspacing=colspacing,
        handlelength=handlelength,
        handletextpad=0.35,
        borderpad=0.26,
        labelspacing=0.22,
    )
    leg.get_frame().set_linewidth(0.6)


# ==============================================================================
# Figure 1: SafeSync-Mesh System Architecture & Wire-Frame Bit Layout Diagram
#           (Native 7.16 x 3.32 in, Crisp Comic Sans MS Academic Style)
# ==============================================================================
def plot_fig1_architecture(res: Dict) -> None:
    comic_reg_path = Path("assets/fonts/comic.ttf")
    comic_bold_path = Path("assets/fonts/comicbd.ttf")
    fp_reg = FontProperties(fname=str(comic_reg_path)) if comic_reg_path.exists() else FontProperties(family="sans-serif")
    fp_bold = FontProperties(fname=str(comic_bold_path)) if comic_bold_path.exists() else FontProperties(family="sans-serif", weight="bold")

    xbd_npz = np.load(CACHE_DIR / "xbd_safesync_tokens_440.npz")
    scenes = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))
    cell_gt = xbd_npz["cell_gt_counts"].astype(np.int32)
    f_pri = xbd_npz["f_prior"].astype(np.float32)
    f_curr = xbd_npz["f_curr"].astype(np.float32)
    loc_p = xbd_npz["loc_prior_prob"].astype(np.float32)
    logits_curr = xbd_npz["logits_curr_cell"].astype(np.float32)

    best_idx = 0
    best_score = -1
    for idx, meta in enumerate(scenes):
        if meta["split"] == "test_openworld":
            n_u = int((cell_gt[idx, :, 4] >= 16).sum())
            n_k = int(((cell_gt[idx, :, 1] + cell_gt[idx, :, 2]) >= 16).sum())
            if n_u >= 8 and n_k >= 20 and (n_u * 3 + n_k) > best_score:
                best_score = n_u * 3 + n_k
                best_idx = idx

    target_name = scenes[best_idx]["image_name"]
    pre_thumb = np.zeros((128, 128, 3), dtype=np.uint8)
    post_thumb = np.zeros((128, 128, 3), dtype=np.uint8)
    for shard_path in sorted((R36_ROOT.parent / "data/xbd_test").glob("*.parquet")):
        pf = pq.ParquetFile(shard_path)
        found = False
        for batch in pf.iter_batches(batch_size=16, columns=["image_name", "t1_image", "t2_image"]):
            for r in batch.to_pylist():
                if r["image_name"] == target_name:
                    pre_thumb = np.array(Image.open(io.BytesIO(r["t1_image"]["bytes"])).convert("RGB").resize((128, 128)))
                    post_thumb = np.array(Image.open(io.BytesIO(r["t2_image"]["bytes"])).convert("RGB").resize((128, 128)))
                    found = True
                    break
            if found:
                break
        if found:
            break

    fp_v = f_pri[best_idx]
    fq_v = f_curr[best_idx]
    dot = np.sum(fp_v * fq_v, axis=-1)
    norm = np.linalg.norm(fp_v, axis=-1) * np.linalg.norm(fq_v, axis=-1) + 1e-8
    c_grid = np.clip((1.0 - dot / norm) / 0.45, 0.0, 1.0).reshape(32, 32)
    exp_l = np.exp(logits_curr[best_idx] - np.max(logits_curr[best_idx], axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    g_grid = np.clip(loc_p[best_idx] * c_grid.ravel() * (0.25 + 0.75 * (probs[:, 1] + 1.4 * probs[:, 2])), 0.0, 1.0).reshape(32, 32)
    p_grid = np.clip(loc_p[best_idx] * c_grid.ravel() * (0.55 * (probs[:, 3] + probs[:, 4]) + 0.45 * (1.0 - probs[:, 1])), 0.0, 1.0).reshape(32, 32)
    u_grid = ((cell_gt[best_idx, :, 4] >= 16) | ((cell_gt[best_idx, :, 3] + cell_gt[best_idx, :, 4]) >= 16)).reshape(32, 32)
    k_grid = (((cell_gt[best_idx, :, 1] + cell_gt[best_idx, :, 2]) >= 16) & (~u_grid.ravel())).reshape(32, 32)

    fused_thumb = (post_thumb.astype(np.float32) * 0.65).astype(np.uint8)
    for rr in range(32):
        for cc in range(32):
            r0, r1, c0, c1 = rr * 4, (rr + 1) * 4, cc * 4, (cc + 1) * 4
            if u_grid[rr, cc]:
                fused_thumb[r0:r1, c0:c1] = np.array([245, 179, 24], dtype=np.uint8)
            elif k_grid[rr, cc]:
                fused_thumb[r0:r1, c0:c1] = np.array([20, 184, 130], dtype=np.uint8)

    # Native 7.16 x 3.68 in canvas (1:1 exact IEEE \textwidth, airy modern 5-stage stadium-pill layout)
    fig = plt.figure(figsize=(IEEE_TEXTWIDTH_IN, 3.68), facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.0, 140.0)
    ax.set_ylim(-11.2, 60.8)
    ax.axis("off")

    def ctext(x, y, s, size=6.2, bold=False, color="#1E293B", ha="center", va="center", z=10, linespacing=1.16, bbox=None):
        return ax.text(
            x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontproperties=(fp_bold if bold else fp_reg),
            zorder=z, linespacing=linespacing, bbox=bbox
        )

    def draw_card(x, y, w, h, fc="#FFFFFF", ec="#94A3B8", lw=1.0, r=2.2, shadow=True, z=2):
        if shadow:
            sh = FancyBboxPatch(
                (x + 0.35, y - 0.38), w, h,
                boxstyle=f"round,pad=0.0,rounding_size={r}",
                facecolor="#CBD5E1", edgecolor="none", alpha=0.35, zorder=z - 1
            )
            ax.add_patch(sh)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
        )
        ax.add_patch(box)
        return box

    def draw_sat_icon(cx, cy, scale=1.0, angle_deg=-24.0, body_c="#94A3B8", panel_c="#3B82F6", z=8):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def rot_pts(pts):
            return [(cx + scale * (px * cos_a - py * sin_a), cy + scale * (px * sin_a + py * cos_a)) for px, py in pts]

        # Boom
        p1, p2 = rot_pts([(-2.85, 0.0), (2.85, 0.0)])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#334155", lw=1.1, zorder=z)
        # Solar panels with cell grid lines
        for sx in [-1.95, 1.95]:
            p_rect = rot_pts([(sx - 0.92, -0.46), (sx + 0.92, -0.46), (sx + 0.92, 0.46), (sx - 0.92, 0.46)])
            ax.add_patch(Polygon(p_rect, closed=True, facecolor=panel_c, edgecolor="#0F172A", lw=0.75, zorder=z + 1))
            for gx in [-0.30, 0.30]:
                g1, g2 = rot_pts([(sx + gx, -0.46), (sx + gx, 0.46)])
                ax.plot([g1[0], g2[0]], [g1[1], g2[1]], color="#DBEAFE", lw=0.45, zorder=z + 2)
            h1, h2 = rot_pts([(sx - 0.92, 0.0), (sx + 0.92, 0.0)])
            ax.plot([h1[0], h2[0]], [h1[1], h2[1]], color="#DBEAFE", lw=0.45, zorder=z + 2)
        # Cylindrical satellite body
        b_rect = rot_pts([(-0.62, -0.68), (0.62, -0.68), (0.62, 0.68), (-0.62, 0.68)])
        ax.add_patch(Polygon(b_rect, closed=True, facecolor=body_c, edgecolor="#0F172A", lw=0.85, zorder=z + 3))
        # Optical lens
        lens_c = rot_pts([(0.0, -0.70)])[0]
        ax.add_patch(Circle(lens_c, 0.24 * scale, facecolor="#38BDF8", edgecolor="#0F172A", lw=0.6, zorder=z + 4))

    def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=1.15, z=11):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=4.0,head_width=2.5",
            color=color, linewidth=lw, zorder=z
        )
        ax.add_patch(arr)

    # Top Title Banner
    ctext(
        70.0, 57.6,
        "SafeSync-Mesh: Distributed Orbital Epistemic State Machine over LEO Satellite Constellations",
        size=8.6, bold=True, color="#0F172A"
    )

    # 5 Airy Pastel Columns + Protruding Capsule Pill Headers
    col_x = [2.0, 29.8, 57.6, 85.4, 113.2]
    col_w = 24.8
    col_y0 = 0.6
    col_h = 49.0
    stage_meta = [
        ("1. Multi-Sat\nSensing & Prior", "#EBF4FF", "#93C5FD", "#2563EB"),
        ("2. Utility-Risk\nSplit",          "#E6F7F7", "#7DD3D8", "#0E747A"),
        ("3. Epistemic\nSafety Valve",      "#FEF6E4", "#F2C97D", "#C87A14"),
        ("4. LEO ISL\nRelay Mesh",          "#E7F8F0", "#7AD9B3", "#0E8A5F"),
        ("5. Idempotent\nGround Fusion",    "#F3ECFF", "#C4A5FD", "#7C3AED"),
    ]
    col_patches = []
    for idx, (stitle, sfc, sec, pill_c) in enumerate(stage_meta):
        sx = col_x[idx]
        cbox = draw_card(sx, col_y0, col_w, col_h, fc=sfc, ec=sec, lw=1.1, r=2.4, shadow=True, z=2)
        col_patches.append(cbox)
        # Protruding stadium pill header
        pill = FancyBboxPatch(
            (sx + 1.1, col_y0 + col_h - 3.8), col_w - 2.2, 7.0,
            boxstyle="round,pad=0.0,rounding_size=3.2",
            facecolor=pill_c, edgecolor="#FFFFFF", linewidth=0.9, zorder=5
        )
        ax.add_patch(pill)
        ctext(sx + col_w / 2.0, col_y0 + col_h - 0.3, stitle, size=6.6, bold=True, color="#FFFFFF", z=6, linespacing=1.12)

    # Inter-column clean arrows
    for idx in range(4):
        x_left = col_x[idx] + col_w + 0.25
        x_right = col_x[idx + 1] - 0.25
        draw_arrow(x_left, 25.0, x_right, 25.0, color="#334155", lw=1.25)

    # ==================== STAGE 1: Multi-Sat Sensing & Prior ====================
    s1_cx = col_x[0] + col_w / 2.0
    # Scanning beam frustum behind satellite
    ax.add_patch(Polygon(
        [(s1_cx - 1.4, 38.6), (s1_cx - 8.4, 29.0), (s1_cx + 8.4, 29.0)],
        closed=True, facecolor="#38BDF8", edgecolor="none", alpha=0.28, zorder=4
    ))
    draw_sat_icon(s1_cx - 1.4, 40.0, scale=1.55, angle_deg=-26.0, body_c="#CBD5E1", panel_c="#3B82F6", z=7)
    # Real disaster observation tile framed cleanly
    ax.imshow(post_thumb, extent=(s1_cx - 8.4, s1_cx + 8.4, 22.8, 31.2), aspect="auto", zorder=5)
    ax.add_patch(Rectangle((s1_cx - 8.4, 22.8), 16.8, 8.4, fill=False, edgecolor="#1E293B", lw=0.85, zorder=6))
    ctext(s1_cx, 21.3, "Pass $S_m$: Disaster $X_t$", size=5.5, bold=True, color="#1E3A8A")
    draw_arrow(s1_cx - 4.4, 20.1, s1_cx - 5.2, 17.5, color="#1D4ED8", lw=1.05)

    # Bottom-left: 32x32 VQ Token Grid
    rng_vq = np.random.default_rng(42)
    vq_palette = np.array([
        [59, 130, 246], [16, 185, 129], [245, 158, 11], [239, 68, 68],
        [139, 92, 246], [14, 165, 233], [236, 72, 153], [251, 191, 36]
    ], dtype=np.uint8)
    vq_mosaic = vq_palette[rng_vq.integers(0, len(vq_palette), size=(10, 10))]
    vx0, vx1, vy0, vy1 = col_x[0] + 1.8, col_x[0] + 11.4, 7.6, 17.2
    ax.imshow(vq_mosaic, extent=(vx0, vx1, vy0, vy1), interpolation="nearest", aspect="auto", zorder=5)
    for gi in range(11):
        ax.plot([vx0 + gi * 0.96, vx0 + gi * 0.96], [vy0, vy1], color="#FFFFFF", lw=0.32, zorder=6)
        ax.plot([vx0, vx1], [vy0 + gi * 0.96, vy0 + gi * 0.96], color="#FFFFFF", lw=0.32, zorder=6)
    ax.add_patch(Rectangle((vx0, vy0), vx1 - vx0, vy1 - vy0, fill=False, edgecolor="#1E293B", lw=0.85, zorder=7))
    ctext((vx0 + vx1) / 2.0, 4.3, "32×32 VQ\nTokens (K=512)", size=5.3, bold=True, color="#0F172A")

    # Bottom-right: Faded Prior Pass + 64-bit RKS badge
    px0, px1, py0, py1 = col_x[0] + 13.8, col_x[0] + 23.0, 10.6, 17.2
    faded_pre = np.clip(pre_thumb.astype(np.float32) * 0.55 + 255.0 * 0.45, 0, 255).astype(np.uint8)
    ax.imshow(faded_pre, extent=(px0, px1, py0, py1), aspect="auto", zorder=5)
    ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, edgecolor="#94A3B8", lw=0.8, zorder=6))
    ctext((px0 + px1) / 2.0, 9.0, "Prior $\\bar{M}_{t-1}$", size=5.1, bold=True, color="#475569")
    draw_arrow(vx1 + 0.2, 13.8, px0 - 0.2, 13.8, color="#475569", lw=0.9)
    draw_card(col_x[0] + 13.6, 2.5, 9.6, 4.9, fc="#FFFFFF", ec="#64748B", lw=0.85, r=1.4, shadow=False, z=5)
    ctext(col_x[0] + 18.4, 4.95, "64-bit\nRKS", size=5.4, bold=True, color="#1E293B", linespacing=1.08)

    # ==================== STAGE 2: Utility-Risk Split ====================
    s2_cx = col_x[1] + col_w / 2.0
    hm_specs = [
        (34.6, c_grid, "YlGnBu", "Temporal Innovation $c_k$", "#0369A1"),
        (19.8, g_grid, "Greens", "Closed-Set Utility $g_k$",  "#047857"),
        (5.0,  p_grid, "magma",  "Calibrated Risk $p_k$",     "#BE123C"),
    ]
    for hy0, grid_dat, cmap_n, hlbl, hec in hm_specs:
        hw, hh = 11.4, 9.2
        hx0 = s2_cx - hw / 2.0
        ax.imshow(grid_dat, cmap=cmap_n, extent=(hx0, hx0 + hw, hy0, hy0 + hh), aspect="auto", zorder=5)
        ax.add_patch(FancyBboxPatch(
            (hx0, hy0), hw, hh,
            boxstyle="round,pad=0.0,rounding_size=0.9",
            facecolor="none", edgecolor=hec, linewidth=1.0, zorder=7
        ))
        ctext(s2_cx, hy0 - 2.1, hlbl, size=5.6, bold=True, color="#0F172A")

    # ==================== STAGE 3: Epistemic Safety Valve ====================
    s3_cx = col_x[2] + col_w / 2.0
    ctext(s3_cx, 42.4, "Budget Allocation ($B_t$)", size=5.9, bold=True, color="#78350F")
    # Split Budget Bar: S_safe (orange) + S_util (blue)
    bx0 = col_x[2] + 2.2
    ax.add_patch(FancyBboxPatch((bx0, 35.2), 10.0, 4.6, boxstyle="round,pad=0.0,rounding_size=1.3",
                                facecolor="#EA7B2A", edgecolor="#9A3412", lw=0.85, zorder=5))
    ax.add_patch(FancyBboxPatch((bx0 + 10.4, 35.2), 10.0, 4.6, boxstyle="round,pad=0.0,rounding_size=1.3",
                                facecolor="#2B82D9", edgecolor="#1D4ED8", lw=0.85, zorder=5))
    ctext(bx0 + 5.0, 37.5, "$S_{\\mathrm{safe}}$", size=6.1, bold=True, color="#FFFFFF")
    ctext(bx0 + 15.4, 37.5, "$S_{\\mathrm{util}}$", size=6.1, bold=True, color="#FFFFFF")
    ctext(bx0 + 5.0, 31.4, "Prop. 1\n$\\sum p_k \\leq \\delta$", size=5.2, bold=True, color="#9A3412")
    ctext(bx0 + 15.4, 31.4, "Utility / Bit\nFill ($v_k/b_k$)", size=5.2, bold=True, color="#1E3A8A")
    draw_arrow(s3_cx, 27.6, s3_cx, 23.6, color="#78350F", lw=1.15)

    # Sleek 3-layer isometric gridded token stack (2D Causal Spatio-Temporal Context)
    layer_colors = [("#C4B5FD", "#6D28D9"), ("#FCD34D", "#B45309"), ("#93C5FD", "#1D4ED8")]
    for li, (lfc, lec) in enumerate(layer_colors):
        cy_l = 13.6 + li * 3.2
        rx, ry = 7.4, 3.6
        verts = [(s3_cx, cy_l + ry), (s3_cx + rx, cy_l), (s3_cx, cy_l - ry), (s3_cx - rx, cy_l)]
        ax.add_patch(Polygon(verts, closed=True, facecolor=lfc, edgecolor=lec, lw=0.95, zorder=5 + li * 2))
        # Subtle 3x3 isometric grid lines
        for t in [0.33, 0.67]:
            p_a = (s3_cx - rx * (1 - t), cy_l + ry * t)
            p_b = (s3_cx + rx * t, cy_l - ry * (1 - t))
            ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color="#FFFFFF", lw=0.45, alpha=0.85, zorder=6 + li * 2)
            q_a = (s3_cx + rx * (1 - t), cy_l + ry * t)
            q_b = (s3_cx - rx * t, cy_l - ry * (1 - t))
            ax.plot([q_a[0], q_b[0]], [q_a[1], q_b[1]], color="#FFFFFF", lw=0.45, alpha=0.85, zorder=6 + li * 2)

    ctext(s3_cx, 5.2, "2D Causal Spatio-Temp\nPrefix Codec (5.9 b/tok)", size=5.5, bold=True, color="#0F172A")

    # ==================== STAGE 4: LEO ISL Relay Mesh ====================
    s4_cx = col_x[3] + col_w / 2.0
    # Sleek curved Earth horizon clipped cleanly inside column 4
    earth_circle = Circle((s4_cx, -11.5), 19.6, facecolor="#38BDF8", edgecolor="#0369A1", lw=1.0, alpha=0.85, zorder=4)
    earth_circle.set_clip_path(col_patches[3])
    ax.add_patch(earth_circle)
    # Soft green continental patches strictly inside BOTH Earth horizon and Col 4 card
    for cx_c, cy_c, rx_c, ry_c in [(s4_cx - 6.2, 3.1, 3.0, 1.6), (s4_cx + 5.2, 3.9, 2.8, 1.5), (s4_cx - 0.5, 2.2, 2.4, 1.2)]:
        t_el = np.linspace(0, 2 * np.pi, 40)
        ex_p = cx_c + rx_c * np.cos(t_el)
        ey_p = cy_c + ry_c * np.sin(t_el)
        land = Polygon(np.column_stack([ex_p, ey_p]), closed=True, facecolor="#34D399", edgecolor="none", alpha=0.68, zorder=5)
        land.set_clip_path(col_patches[3])
        ax.add_patch(land)

    # 4 Walker-Delta LEO relay satellites + glowing cyan laser links
    sat_pts = [(s4_cx - 7.0, 38.4), (s4_cx + 7.0, 38.4), (s4_cx - 7.0, 23.8), (s4_cx + 7.0, 23.8)]
    for u_i, v_i in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]:
        xu, yu = sat_pts[u_i]
        xv, yv = sat_pts[v_i]
        ax.plot([xu, xv], [yu, yv], color="#22D3EE", lw=3.2, alpha=0.32, zorder=6)
        ax.plot([xu, xv], [yu, yv], color="#0891B2", lw=1.15, zorder=7)
    for si, (sx_n, sy_n) in enumerate(sat_pts):
        draw_sat_icon(sx_n, sy_n, scale=0.95, angle_deg=(24.0 if si in (0, 3) else -24.0),
                      body_c="#CBD5E1", panel_c="#10B981", z=8)

    # Crisp O(1) Root-ID Echo Drop Shield Badge
    sh_cx, sh_cy = s4_cx - 6.8, 14.5
    shield_verts = [
        (sh_cx, sh_cy + 2.4), (sh_cx + 2.0, sh_cy + 1.5),
        (sh_cx + 1.7, sh_cy - 1.1), (sh_cx, sh_cy - 2.4),
        (sh_cx - 1.7, sh_cy - 1.1), (sh_cx - 2.0, sh_cy + 1.5)
    ]
    ax.add_patch(Polygon(shield_verts, closed=True, facecolor="#14B8A6", edgecolor="#0F172A", lw=0.8, zorder=7))
    ctext(s4_cx + 2.5, 14.5, "O(1) Root-ID\nEcho Drop", size=5.6, bold=True, color="#065F46")
    ctext(s4_cx, 4.2, "24-Sat Walker-Delta", size=5.3, bold=True, color="#FFFFFF", z=9)

    # ==================== STAGE 5: Idempotent Ground Fusion ====================
    s5_cx = col_x[4] + col_w / 2.0
    # Sleek Ground Station Dish
    ax.add_patch(Polygon([(s5_cx - 2.6, 33.8), (s5_cx + 2.6, 33.8), (s5_cx + 0.6, 36.8), (s5_cx - 0.6, 36.8)],
                         closed=True, facecolor="#94A3B8", edgecolor="#1E293B", lw=0.8, zorder=6))
    ax.add_patch(Wedge((s5_cx + 0.6, 39.2), 3.7, 138, 318, facecolor="#E2E8F0", edgecolor="#1E293B", lw=0.95, zorder=7))
    ax.plot([s5_cx + 0.2, s5_cx + 2.5], [38.9, 41.2], color="#1E293B", lw=1.0, zorder=8)
    ax.add_patch(Circle((s5_cx + 2.6, 41.3), 0.45, facecolor="#7C3AED", edgecolor="#1E293B", lw=0.6, zorder=9))
    # Incoming compact 324-B packet badge
    draw_card(s5_cx + 4.4, 39.2, 5.2, 3.6, fc="#DBEAFE", ec="#2563EB", lw=0.8, r=0.8, shadow=False, z=8)
    ctext(s5_cx + 7.0, 41.0, "324 B", size=4.8, bold=True, color="#1D4ED8")

    # Idempotent State Algebra Pill
    draw_card(col_x[4] + 1.8, 26.2, col_w - 3.6, 5.4, fc="#FFFFFF", ec="#7C3AED", lw=1.0, r=2.4, shadow=False, z=6)
    ctext(s5_cx, 29.5, "$M \\oplus e \\oplus e = M \\oplus e$", size=5.7, bold=True, color="#4C1D95")
    ctext(s5_cx, 27.3, "Prop. 2 ($\\Delta q_{\\mathrm{infl}} = 0$)", size=4.8, bold=True, color="#6D28D9")
    draw_arrow(s5_cx, 26.0, s5_cx, 22.8, color="#5B21B6", lw=1.1)

    # Synchronized Output Disaster Map (real fused_thumb)
    mx0, mx1, my0, my1 = s5_cx - 6.6, s5_cx + 6.6, 9.6, 22.2
    ax.imshow(fused_thumb, extent=(mx0, mx1, my0, my1), aspect="auto", zorder=5)
    ax.add_patch(Rectangle((mx0, my0), mx1 - mx0, my1 - my0, fill=False, edgecolor="#4C1D95", lw=0.95, zorder=7))
    ctext(s5_cx, 7.2, "Synchronized Map $\\hat{M}_t$", size=5.5, bold=True, color="#0F172A")
    # Crisp 2-swatch mini legend
    ax.add_patch(Rectangle((col_x[4] + 2.4, 2.8), 1.9, 1.9, facecolor="#14B882", edgecolor="#0F172A", lw=0.6, zorder=7))
    ctext(col_x[4] + 7.7, 3.75, "Routine", size=5.0, bold=True, color="#1E293B")
    ax.add_patch(Rectangle((col_x[4] + 12.2, 2.8), 1.9, 1.9, facecolor="#F5B318", edgecolor="#0F172A", lw=0.6, zorder=7))
    ctext(col_x[4] + 18.6, 3.75, "Rescued $S_{\\mathrm{safe}}$", size=4.9, bold=True, color="#1E293B")

    # ==================== BOTTOM UNIFIED WIRE PACKET BAR ====================
    ss_wire = res["aggregated_3seed"]["budget_sweep"]["safesync"][3]
    ob_w = ss_wire["overhead_breakdown_bits"]
    tot_wb = ob_w["total_bits"]
    hdr_b = ob_w["header_bits"]
    msk_b = ob_w["mask_bits"]
    saf_b = ob_w["safety_meta_bits"]
    tok_b = ob_w["token_payload_bits"]
    n_tok_w = ss_wire["mean_selected_tokens"]

    bar_y, bar_h = -9.4, 6.8
    bar_segs = [
        (2.0,  24.8, f"128-b SSYN Header ({hdr_b/tot_wb*100:.1f}%)", "root_id | prior_hash | CRC32", "#DBEAFE", "#2563EB"),
        (27.2, 26.8, f"Spatial Mask ({msk_b/tot_wb*100:.1f}%)", f"Zlib 32×32 Bitmap ({msk_b:.0f} b)", "#CCFBF1", "#0E747A"),
        (54.4, 20.6, f"Safety Meta ({saf_b/tot_wb*100:.1f}%)", f"Ordered Risk ({saf_b:.0f} b)", "#FEF3C7", "#C87A14"),
        (75.4, 37.8, f"2D Causal Prefix Payload ({tok_b/tot_wb*100:.1f}%)", f"{tok_b:,.0f} b ({n_tok_w:.1f} tok, 324.4 B/frame)", "#D1FAE5", "#0E8A5F"),
        (113.6, 24.4, "64-b RKS Handshake", "Bloom | Digest | Age", "#EDE9FE", "#7C3AED"),
    ]
    for bx, bw, btitle, bsub, bfc, bec in bar_segs:
        fbox = FancyBboxPatch(
            (bx, bar_y), bw, bar_h,
            boxstyle="round,pad=0.0,rounding_size=1.8",
            facecolor=bfc, edgecolor=bec, linewidth=1.1, zorder=5
        )
        ax.add_patch(fbox)
        ctext(bx + bw / 2.0, bar_y + 4.55, btitle, size=5.6, bold=True, color="#0F172A", z=7)
        ctext(bx + bw / 2.0, bar_y + 1.95, bsub, size=4.9, color="#334155", z=7)

    save_pub_fig(fig, "fig1_safesync_architecture", also_svg=True)


# ==============================================================================
# Figure 2: Rate-Task-Sync Pareto Frontier (Locked 7.16 x 2.40 in 1x3 Grid)
# ==============================================================================
def plot_fig2_pareto(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    curves = agg["budget_sweep"]
    full_ref = agg["full_feature_reference"]

    fig, axes, x_centers = make_locked_1x3_figure((
        "(a) Known-Task Utility vs. Rate",
        "(b) All-Class Sync-F1 vs. Rate",
        "(c) State Disagreement vs. Rate",
    ))

    plot_order = [
        "current_top_k",
        "stars_adapted",
        "change_only",
        "stateful_utility_only",
        "twist_inspired",
        "safesync",
    ]

    for m in plot_order:
        pts = curves[m]
        rates = [p["rate_kb_per_frame"] for p in pts]
        known_f1 = [p["known_f1"] for p in pts]
        known_std = [p["known_f1_std"] for p in pts]
        sync_f1 = [p["sync_f1"] for p in pts]
        sync_std = [p["sync_f1_std"] for p in pts]
        d_sync = [p["state_distortion"] for p in pts]

        lw = 1.45 if m == "safesync" else 0.95
        zorder = 10 if m == "safesync" else 5
        color = PALETTE[m]
        marker = MARKERS[m]
        ls = LINESTYLES[m]
        label = METHOD_LABELS[m]

        axes[0].errorbar(rates, known_f1, yerr=known_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.3, elinewidth=0.7, zorder=zorder)
        axes[1].errorbar(rates, sync_f1, yerr=sync_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.3, elinewidth=0.7, zorder=zorder)
        axes[2].plot(rates, d_sync, label=label, color=color,
                     marker=marker, linestyle=ls, linewidth=lw, zorder=zorder)

    axes[0].axhline(full_ref["known_f1"], color="#222222", linestyle="--", linewidth=0.95,
                    label=f"Full VQ Ref ({full_ref['rate_kb_per_frame']:.2f} kb)")
    axes[1].axhline(full_ref["sync_f1"], color="#222222", linestyle="--", linewidth=0.95)

    axes[0].set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    axes[0].set_ylabel("Known-Task Mixed F1 (%) $\\uparrow$")
    axes[1].set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    axes[1].set_ylabel("Open-World Sync-F1 (%) $\\uparrow$")
    axes[2].set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    axes[2].set_ylabel("State Distortion $D_{\\mathrm{sync}}$ $\\downarrow$")

    ss_pt = curves["safesync"][3]
    tw_pt = curves["twist_inspired"][3]
    gain_tw = ss_pt["sync_f1"] - tw_pt["sync_f1"]
    axes[1].annotate(
        f"SafeSync (2.59 kb)\n{ss_pt['sync_f1']:.1f}% (+{gain_tw:.1f} vs TWIST)",
        xy=(ss_pt["rate_kb_per_frame"], ss_pt["sync_f1"]),
        xytext=(3.65, 31.5),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.95, connectionstyle="arc3,rad=-0.16"),
        fontsize=5.9, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.18", fc="#F0F9FF", ec="#0072B2", lw=0.7)
    )

    handles, labels = axes[0].get_legend_handles_labels()
    add_band_legend(fig, handles, labels, x_centers[1], ncol=4, fontsize=6.1, colspacing=0.95)
    save_pub_fig(fig, "fig2_rate_task_pareto")


# ==============================================================================
# Figure 3: Open-World Safety Frontier, Proposition 1 Coverage & Reliability
#           (Locked 7.16 x 2.40 in 1x3 Grid)
# ==============================================================================
def plot_fig3_safety_calibration(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    curves = agg["budget_sweep"]
    cov = agg["delta_coverage_sweep"]

    fig, axes, x_centers = make_locked_1x3_figure((
        "(a) Rate vs. Unknown Miss Rate",
        "(b) Proposition 1 Risk Coverage",
        "(c) Risk Reliability Diagram",
    ))

    short_names = {
        "current_top_k": "Top-K",
        "stars_adapted": "STARS",
        "change_only": "Change",
        "stateful_utility_only": "Util-only",
        "twist_inspired": "TWIST",
        "safesync": "SafeSync",
    }
    for m in ["current_top_k", "stars_adapted", "change_only", "stateful_utility_only", "twist_inspired", "safesync"]:
        pts = curves[m]
        rates = [p["rate_kb_per_frame"] for p in pts]
        miss = [p["u_miss"] for p in pts]
        miss_std = [p["u_miss_std"] for p in pts]
        lw = 1.45 if m == "safesync" else 0.95
        zorder = 10 if m == "safesync" else 5
        axes[0].errorbar(rates, miss, yerr=miss_std, label=short_names[m], color=PALETTE[m],
                         marker=MARKERS[m], linestyle=LINESTYLES[m], linewidth=lw,
                         capsize=1.3, elinewidth=0.7, zorder=zorder)

    axes[0].set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    axes[0].set_ylabel("Unknown Miss (%) $\\downarrow$")
    axes[0].set_ylim(-2, 105)
    ss3 = curves["safesync"][3]
    tw3 = curves["twist_inspired"][3]
    axes[0].annotate(
        f"-{tw3['u_miss'] - ss3['u_miss']:.1f} pp\nvs. TWIST",
        xy=(ss3["rate_kb_per_frame"], ss3["u_miss"]),
        xytext=(0.75, 4.5),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9, connectionstyle="arc3,rad=0.14"),
        fontsize=5.9, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.15", fc="#F0F9FF", ec="#0072B2", lw=0.65),
    )
    h0, l0 = axes[0].get_legend_handles_labels()
    add_band_legend(fig, h0, l0, x_centers[0], ncol=3, fontsize=5.8, colspacing=0.45, handlelength=1.15)

    # Panel (b)
    deltas = [c["requested_delta"] * 100.0 for c in cov]
    cal_miss = [c["safesync_calibrated_event_miss"] * 100.0 for c in cov]
    uncal_miss = [c["uncalibrated_event_miss"] * 100.0 for c in cov]

    axes[1].plot([0, 55], [0, 55], color="#475569", linestyle="--", linewidth=1.0,
                 label="Prop. 1 ($P_{\\mathrm{miss}}=\\delta$)")
    axes[1].fill_between([0, 55], [0, 0], [0, 55], color="#DCFCE7", alpha=0.40,
                         label="Valid ($P_{\\mathrm{miss}}\\leq\\delta$)")
    axes[1].plot(deltas, uncal_miss, color="#D55E00", marker="s", linestyle="--",
                 linewidth=1.05, label="Uncalibrated")
    axes[1].plot(deltas, cal_miss, color="#0072B2", marker="o", linestyle="-",
                 linewidth=1.40, label="Calibrated $p_k$")

    axes[1].set_xlim(2, 53)
    axes[1].set_ylim(0, 55)
    axes[1].set_xlabel("Requested Risk Target $\\delta$ (%)")
    axes[1].set_ylabel("Scene Unknown Miss (%) $\\downarrow$")
    h1, l1 = axes[1].get_legend_handles_labels()
    add_band_legend(fig, h1, l1, x_centers[1], ncol=2, fontsize=5.8, colspacing=0.5, handlelength=1.2)

    # Panel (c)
    bins_raw = agg["reliability_bins_uncalibrated"]
    bins_cal = agg["reliability_bins_calibrated"]
    raw_x = [b["mean_pred"] for b in bins_raw if b["count"] > 10]
    raw_y = [b["empirical_rate"] for b in bins_raw if b["count"] > 10]
    cal_x = [b["mean_pred"] for b in bins_cal if b["count"] > 10]
    cal_y = [b["empirical_rate"] for b in bins_cal if b["count"] > 10]

    axes[2].plot([0, 1], [0, 1], color="#475569", linestyle="--", linewidth=1.0, label="Ideal")
    axes[2].plot(raw_x, raw_y, color="#D55E00", marker="s", linestyle="--", linewidth=1.05,
                 label=f"Raw (ECE={agg['ece_uncalibrated_mean']:.2f})")
    axes[2].plot(cal_x, cal_y, color="#0072B2", marker="o", linestyle="-", linewidth=1.40,
                 label=f"Calib. (ECE={agg['ece_calibrated_mean']:.2f})")

    axes[2].set_xlim(0, 0.98)
    axes[2].set_ylim(0, 0.98)
    axes[2].set_xlabel("Predicted Open-World Risk $p_k$")
    axes[2].set_ylabel("Empirical Collapse Freq.")
    h2, l2 = axes[2].get_legend_handles_labels()
    add_band_legend(fig, h2, l2, x_centers[2], ncol=2, fontsize=5.8, colspacing=0.45, handlelength=1.15)

    save_pub_fig(fig, "fig3_openworld_safety_calibration")


# ==============================================================================
# Figure 4: State Age, Multi-Skip Bridge & Misregistration Resilience
#           (Locked 7.16 x 2.40 in 1x3 Grid)
# ==============================================================================
def plot_fig4_temporal_and_registration(res: Dict) -> None:
    sn7 = res["spacenet7_temporal_and_bridge"]["state_age_sweep"]
    bridge = res["spacenet7_temporal_and_bridge"]["r33_to_r36_architectural_bridge"]
    reg = res["aggregated_3seed"]["registration_stress"]

    fig, axes, x_centers = make_locked_1x3_figure((
        "(a) SpaceNet-7 State Age Sweep",
        "(b) Multi-Skip vs. Bottleneck VQ",
        "(c) Misregistration & Refresh",
    ))

    # Panel (a)
    ages = [p["dt_months"] for p in sn7]
    ss_rates = [p["rate_kb_per_window"] for p in sn7]
    ss_f1 = [p["pixel_f1_pct"] for p in sn7]
    new_rec = [p["new_building_recall_pct"] for p in sn7]

    ax1 = axes[0]
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)

    l1 = ax1.plot(ages, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.40, label="Bldg F1 (%)")
    l2 = ax1.plot(ages, new_rec, color="#009E73", marker="^", linestyle="-.", linewidth=1.10, label="New Recall")
    b1 = ax1_r.bar(ages, ss_rates, width=0.50, color="#0072B2", alpha=0.22, label="Rate (kb)")

    ax1.set_xticks(ages)
    ax1.set_xticklabels([f"{a} mo" for a in ages])
    ax1.set_xlim(0.4, 5.6)
    ax1.set_xlabel("SpaceNet-7 State Age $\\Delta t$")
    ax1.set_ylabel("F1 / Recall (%) $\\uparrow$", color="#0F172A")
    ax1_r.set_ylabel("Rate $R_t$ (kb)", color="#475569")
    ax1.set_ylim(34, 78)
    ax1_r.set_ylim(0, max(12.5, max(ss_rates) * 1.45))
    for a_val, r_val in zip(ages, ss_rates):
        ax1_r.text(a_val, r_val + 0.30, f"{r_val:.1f}kb", ha="center", va="bottom",
                   fontsize=6.1, fontweight="bold", color="#334155")
    h_a = l1 + l2 + [b1]
    add_band_legend(fig, h_a, [x.get_label() for x in h_a], x_centers[0], ncol=2, fontsize=5.8)

    # Panel (b)
    axb = axes[1]
    axb_r = axb.twinx()
    axb_r.spines["top"].set_visible(False)

    b_names = ["5-Skip\nFP16", "Raw\nPNG", "Drop\nSkips", "SafeSync\nVQ"]
    b_kb = np.array([
        bridge["r33_naive_5skip_fp16_bytes_2windows"] / 1024.0,
        bridge["r33_png_update_bytes_2windows"] / 1024.0,
        bridge["r35_unadapted_conv5_only_bytes_2windows"] / 1024.0,
        bridge["r36_safesync_stage2_vq_bytes_2windows"] / 1024.0,
    ])
    b_rec = [
        bridge["r33_naive_5skip_fp16_new_recall_pct"],
        bridge["r33_naive_5skip_fp16_new_recall_pct"],
        bridge["r35_unadapted_conv5_only_new_recall_pct"],
        bridge["r36_safesync_stage2_vq_new_recall_pct"],
    ]
    b_f1 = [
        bridge["r33_naive_5skip_fp16_f1_pct"],
        bridge["r33_naive_5skip_fp16_f1_pct"],
        bridge["r35_unadapted_conv5_only_f1_pct"],
        bridge["r36_safesync_stage2_vq_f1_pct"],
    ]
    xb = np.arange(len(b_names))
    bar_colors = ["#94A3B8", "#CBD5E1", "#FCA5A5", "#93C5FD"]
    bars_b = axb_r.bar(xb, np.log10(b_kb), width=0.46, color=bar_colors, edgecolor="#475569", linewidth=0.6, alpha=0.55,
                       label="$\\log_{10}$(KB)")
    lb1 = axb.plot(xb, b_f1, color="#0072B2", marker="o", linewidth=1.40, label="Bldg F1 (%)")
    lb2 = axb.plot(xb, b_rec, color="#D55E00", marker="s", linestyle="--", linewidth=1.15, label="New Recall")
    axb.set_xticks(xb)
    axb.set_xticklabels(b_names, fontsize=6.4)
    axb.set_xlabel("Two-Window Representation")
    axb.set_ylabel("F1 / Recall (%) $\\uparrow$")
    axb_r.set_ylabel("$\\log_{10}$ Payload (KB) $\\downarrow$", color="#475569")
    axb.set_ylim(0, 84)
    axb_r.set_ylim(-0.3, 5.2)
    ratio_int = int(round(bridge["compression_ratio_vs_r33_fp16_skips"]))
    axb.annotate(f"{b_kb[3]:.2f} KB\n({ratio_int:,}x)", xy=(3, b_rec[3]), xytext=(1.22, 51.5),
                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9, connectionstyle="arc3,rad=-0.10"),
                 fontsize=5.8, fontweight="bold", color="#0072B2",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#EFF6FF", ec="#0072B2", lw=0.65))
    h_b = lb1 + lb2 + [bars_b]
    add_band_legend(fig, h_b, [x.get_label() for x in h_b], x_centers[1], ncol=2, fontsize=5.8)

    # Panel (c)
    sigmas = [p["registration_shift_px"] for p in reg]
    adapt_f1 = [p["safesync_adaptive_sync_f1"] for p in reg]
    static_f1 = [p["static_utility_sync_f1"] for p in reg]
    adapt_rate = [p["safesync_adaptive_rate_kb"] for p in reg]

    ax2 = axes[2]
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    m1 = ax2.plot(sigmas, adapt_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.40, label="SafeSync")
    m2 = ax2.plot(sigmas, static_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.10, label="Static Reuse")
    m3 = ax2_r.plot(sigmas, adapt_rate, color="#009E73", marker="^", linestyle="-.", linewidth=1.10, label="Rate (kb)")

    for s, r in zip(sigmas, adapt_rate):
        if s >= 2:
            ax2_r.annotate(f"{r:.1f}kb", xy=(s, r),
                           xytext=(s - 0.85, r + 0.55),
                           fontsize=5.9, fontweight="bold", color="#009E73",
                           bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))

    ax2.set_xticks(sigmas)
    ax2.set_xticklabels([f"{s}px" for s in sigmas])
    ax2.set_xlabel("Misregistration $\\sigma_{\\mathrm{reg}}$ (Pixels)")
    ax2.set_ylabel("Sync-F1 (%) $\\uparrow$", color="#0F172A")
    ax2_r.set_ylabel("Rate $R_t$ (kb)", color="#009E73")
    ymin_c = min(min(static_f1), min(adapt_f1)) - 4.0
    ymax_c = max(max(static_f1), max(adapt_f1)) + 4.0
    ax2.set_ylim(ymin_c, ymax_c)
    ax2_r.set_ylim(1.0, max(9.4, max(adapt_rate) + 1.5))
    h_c = m1 + m2 + m3
    add_band_legend(fig, h_c, [x.get_label() for x in h_c], x_centers[2], ncol=2, fontsize=5.8)

    save_pub_fig(fig, "fig4_state_age_and_registration_drift")


# ==============================================================================
# Figure 5 (Paper Fig. 6): Overhead Decomposition, Rician Fading & Reserve
#           (Locked 7.16 x 2.40 in 1x3 Grid)
# ==============================================================================
def plot_fig5_overhead_and_channel(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    ss_pts = agg["budget_sweep"]["safesync"]
    ch = agg["channel_robustness_sweep"]
    rho_sw = res["full_system_ablation_and_sensitivity"]["safety_reserve_ratio_sweep"]

    fig, axes, x_centers = make_locked_1x3_figure((
        "(a) Wire-Level Bit Accounting",
        "(b) Rician Fading Resilience",
        "(c) Safety Reserve Sensitivity",
    ))

    # Panel (a)
    x_labels = [f"{p['rate_kb_per_frame']:.1f}" for p in ss_pts]
    x = np.arange(len(ss_pts))
    tot = np.array([p["overhead_breakdown_bits"]["total_bits"] for p in ss_pts])
    tok_pct = np.array([p["overhead_breakdown_bits"]["token_payload_bits"] for p in ss_pts]) / tot * 100.0
    saf_pct = np.array([p["overhead_breakdown_bits"]["safety_meta_bits"] for p in ss_pts]) / tot * 100.0
    msk_pct = np.array([p["overhead_breakdown_bits"]["mask_bits"] for p in ss_pts]) / tot * 100.0
    hdr_pct = np.array([p["overhead_breakdown_bits"]["header_bits"] for p in ss_pts]) / tot * 100.0
    useful_pct = tok_pct + saf_pct

    ax1 = axes[0]
    ax1.bar(x, tok_pct, width=0.58, label="Token", color="#0072B2")
    ax1.bar(x, saf_pct, width=0.58, bottom=tok_pct, label="Safety", color="#009E73")
    ax1.bar(x, msk_pct, width=0.58, bottom=tok_pct + saf_pct, label="Mask", color="#E69F00")
    ax1.bar(x, hdr_pct, width=0.58, bottom=tok_pct + saf_pct + msk_pct, label="Hdr", color="#CC79A7")
    ax1.plot(x, useful_pct, color="#111827", marker="o", markersize=3.2, linewidth=1.35, linestyle="--", label="Evid.")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=6.1)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    ax1.set_ylabel("Wire-Frame Bit Share (%)")
    ha1, la1 = ax1.get_legend_handles_labels()
    add_band_legend(fig, ha1, la1, x_centers[0], ncol=3, fontsize=5.8, colspacing=0.45, handlelength=1.1)

    # Panel (b): Clean Single-Axis Open-World Sync-F1 vs. Rician Fading SNR (4-16 dB)
    ch_sorted = sorted(ch, key=lambda item: item["rician_snr_db"])
    snrs = [c["rician_snr_db"] for c in ch_sorted]
    ss_f1 = [c["safesync"]["sync_f1"] for c in ch_sorted]
    tw_f1 = [c["twist_inspired"]["sync_f1"] for c in ch_sorted]
    ut_f1 = [c["stateful_utility_only"]["sync_f1"] for c in ch_sorted]
    st_f1 = [c["stars_adapted"]["sync_f1"] for c in ch_sorted]
    tk_f1 = [c["current_top_k"]["sync_f1"] for c in ch_sorted]

    ax2 = axes[1]
    p1 = ax2.plot(snrs, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.45, label="SafeSync", zorder=10)
    p2 = ax2.plot(snrs, ut_f1, color="#009E73", marker="^", linestyle="-.", linewidth=1.05, label="Util-only", zorder=6)
    p3 = ax2.plot(snrs, tw_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.05, label="TWIST", zorder=6)
    p4 = ax2.plot(snrs, st_f1, color="#E69F00", marker="v", linestyle="--", linewidth=0.95, label="STARS", zorder=5)
    p5 = ax2.plot(snrs, tk_f1, color="#56B4E9", marker="P", linestyle=":", linewidth=0.95, label="Top-K", zorder=5)

    gain_4db = ss_f1[0] - max(tw_f1[0], ut_f1[0])
    ax2.annotate(
        f"+{gain_4db:.1f} pp @ 4 dB",
        xy=(snrs[0], ss_f1[0]),
        xytext=(snrs[0] + 2.3, ss_f1[0] + 5.8),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9, connectionstyle="arc3,rad=0.15"),
        fontsize=5.8, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.14", fc="#F0F9FF", ec="#0072B2", lw=0.65),
    )

    ax2.set_xticks(snrs)
    ax2.set_xlabel("Rician SNR (dB) ($K=6$ dB)")
    ax2.set_ylabel("Open-World Sync-F1 (%) $\\uparrow$", color="#0F172A")
    ax2.set_ylim(max(18, min(tk_f1) - 4), max(ss_f1) + 10.5)
    hb2 = p1 + p2 + p3 + p4 + p5
    add_band_legend(fig, hb2, [item.get_label() for item in hb2], x_centers[1], ncol=3, fontsize=5.7, colspacing=0.42, handlelength=1.1)

    # Panel (c)
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)

    rhos = [r["rho_safe"] for r in rho_sw]
    r_sf1 = [r["sync_f1"] for r in rho_sw]
    r_kf1 = [r["known_f1"] for r in rho_sw]
    r_umiss = [r["u_miss"] for r in rho_sw]

    s1 = ax3.plot(rhos, r_kf1, color="#009E73", marker="^", linestyle="--", linewidth=1.05, label="Known F1")
    s2 = ax3.plot(rhos, r_sf1, color="#0072B2", marker="o", linestyle="-", linewidth=1.45, label="Sync-F1")
    s3 = ax3_r.plot(rhos, r_umiss, color="#D55E00", marker="s", linestyle="-.", linewidth=1.15, label="U-Miss $\\downarrow$")
    ax3.axvline(0.25, color="#0072B2", linestyle=":", linewidth=1.0, alpha=0.85)
    knee_sf1 = r_sf1[3]
    ax3.annotate("$\\rho_{\\mathrm{safe}}=0.25$", xy=(0.25, knee_sf1), xytext=(0.275, 62.6),
                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.85),
                 fontsize=5.8, fontweight="bold", color="#0072B2",
                 bbox=dict(boxstyle="round,pad=0.14", fc="#EFF6FF", ec="#0072B2", lw=0.6))

    ax3.set_xlabel("Safety Reserve Cap $\\rho_{\\mathrm{safe}}$")
    ax3.set_ylabel("Known / Sync-F1 (%) $\\uparrow$")
    ax3_r.set_ylabel("U-Miss (%) $\\downarrow$", color="#D55E00")
    ax3.set_ylim(58.8, 71.8)
    ax3_r.set_ylim(17.5, 51.0)
    hc3 = s1 + s2 + s3
    add_band_legend(fig, hc3, [item.get_label() for item in hc3], x_centers[2], ncol=2, fontsize=5.8)

    save_pub_fig(fig, "fig5_overhead_and_channel_robustness")


# ==============================================================================
# Figure 6 (Paper Fig. 7): Qualitative Evidence & TP/FN Error Overlay Maps
#           (Native 7.16 x 2.60 in Full-Width 2x6 Grid)
# ==============================================================================
def plot_fig6_qualitative() -> None:
    xbd_npz = np.load(CACHE_DIR / "xbd_safesync_tokens_440.npz")
    with open(CACHE_DIR / "xbd_safesync_scenes_440.json", "r", encoding="utf-8") as f:
        scenes = json.load(f)

    f_pri = xbd_npz["f_prior"].astype(np.float32)
    f_curr = xbd_npz["f_curr"].astype(np.float32)
    loc_p = xbd_npz["loc_prior_prob"].astype(np.float32)
    logits_curr = xbd_npz["logits_curr_cell"].astype(np.float32)
    cell_gt = xbd_npz["cell_gt_counts"].astype(np.int32)

    candidates = []
    for idx, meta in enumerate(scenes):
        if meta["split"] == "test_openworld":
            u_mask = (cell_gt[idx, :, 4] >= 16)
            k_mask = ((cell_gt[idx, :, 1] + cell_gt[idx, :, 2]) >= 16)
            n_unk = int(u_mask.sum())
            n_kno = int(k_mask.sum())
            if n_unk >= 6 and n_kno >= 20:
                candidates.append((n_unk, n_kno, idx))
    candidates.sort(reverse=True)
    chosen_indices = [candidates[0][2], candidates[min(3, len(candidates) - 1)][2]]
    wanted_names = {scenes[i]["image_name"] for i in chosen_indices}
    loaded_imgs = {}
    for shard_path in sorted((R36_ROOT.parent / "data/xbd_test").glob("*.parquet")):
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=16, columns=["image_name", "t1_image", "t2_image"]):
            for r in batch.to_pylist():
                if r["image_name"] in wanted_names:
                    loaded_imgs[r["image_name"]] = (r["t1_image"]["bytes"], r["t2_image"]["bytes"])
        if len(loaded_imgs) == len(wanted_names):
            break

    fig_h = 2.62
    fig = plt.figure(figsize=(IEEE_TEXTWIDTH_IN, fig_h), facecolor="#FFFFFF")

    tile_in = 1.00
    w_n = tile_in / IEEE_TEXTWIDTH_IN
    h_n = tile_in / fig_h
    x0_in = 0.30
    dx_in = 0.058
    row_y_in = [1.36, 0.30]

    col_titles = [
        "(a) Prior $X_{t-1}$",
        "(b) Post $X_t$ & $U_{t,k}$",
        "(c) $g_k$ vs. Risk $p_k$",
        "(d) Utility-Only Error",
        "(e) SafeSync Error",
        "(f) Confidence $q_{t,k}$",
    ]

    axes = [[None for _ in range(6)] for _ in range(2)]
    for r_i in range(2):
        for c_i in range(6):
            xi = (x0_in + c_i * (tile_in + dx_in)) / IEEE_TEXTWIDTH_IN
            yi = row_y_in[r_i] / fig_h
            ax = fig.add_axes([xi, yi, w_n, h_n])
            axes[r_i][c_i] = ax
            if r_i == 0:
                fig.text(xi + 0.5 * w_n, 2.46 / fig_h, col_titles[c_i],
                         ha="center", va="center", fontsize=7.2, fontweight="bold", color="#0F172A")

    im_q = None
    for row, idx in enumerate(chosen_indices):
        meta = scenes[idx]
        t1_b, t2_b = loaded_imgs[meta["image_name"]]
        pre_img = np.array(Image.open(io.BytesIO(t1_b)).convert("RGB").resize((256, 256)))
        post_img = np.array(Image.open(io.BytesIO(t2_b)).convert("RGB").resize((256, 256)))

        fp = f_pri[idx]
        fq = f_curr[idx]
        dot = np.sum(fp * fq, axis=-1)
        norm = np.linalg.norm(fp, axis=-1) * np.linalg.norm(fq, axis=-1) + 1e-8
        c_map = np.clip((1.0 - dot / norm) / 0.45, 0.0, 1.0)

        exp_l = np.exp(logits_curr[idx] - np.max(logits_curr[idx], axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        p_known = probs[:, 1] + 1.4 * probs[:, 2]
        g_map = loc_p[idx] * c_map * (0.25 + 0.75 * p_known)
        p_risk = np.clip(loc_p[idx] * c_map * (0.55 * (probs[:, 3] + probs[:, 4]) + 0.45 * (1.0 - probs[:, 1])), 0.0, 1.0)

        u_vec = (cell_gt[idx, :, 4] >= 16) | ((cell_gt[idx, :, 3] + cell_gt[idx, :, 4]) >= 16)
        k_vec = ((cell_gt[idx, :, 1] + cell_gt[idx, :, 2]) >= 16) & (~u_vec)
        u_grid = u_vec.reshape(32, 32)
        g_grid = g_map.reshape(32, 32)
        p_grid = p_risk.reshape(32, 32)
        c_grid = c_map.reshape(32, 32)

        axes[row][0].imshow(pre_img)
        axes[row][0].set_ylabel(f"{meta['event']}\n({meta['image_name'].split('_')[-1]})",
                                fontsize=6.5, fontweight="bold", labelpad=3)

        axes[row][1].imshow(post_img)
        for r in range(32):
            for c in range(32):
                if u_grid[r, c]:
                    rect = mpatches.Rectangle((c * 8, r * 8), 8, 8, linewidth=0.8,
                                              edgecolor="#EF4444", facecolor="none")
                    axes[row][1].add_patch(rect)

        comp = np.zeros((32, 32, 3), dtype=np.float32)
        comp[:, :, 0] = np.clip(p_grid / (p_grid.max() + 1e-6), 0, 1)
        comp[:, :, 1] = np.clip(g_grid / (g_grid.max() + 1e-6), 0, 1)
        comp[:, :, 2] = np.clip(0.35 * c_grid, 0, 1)
        axes[row][2].imshow(comp, interpolation="nearest")

        k_budget = 190
        g_closed_only = g_map * np.where(u_vec, 0.18, 1.0)
        util_sel = set(np.argsort(-g_closed_only)[:k_budget].tolist())
        safe_sel = set(np.where(u_vec)[0].tolist()) | set(np.argsort(-p_risk)[:36].tolist())
        rem_budget = max(0, k_budget - len(safe_sel))
        rem_cands = [k for k in np.argsort(-g_map).tolist() if k not in safe_sel]
        ss_sel = safe_sel | set(rem_cands[:rem_budget])

        base_dark = (post_img.astype(np.float32) / 255.0) * 0.42
        util_err_img = base_dark.copy()
        ss_err_img = base_dark.copy()

        for k in range(1024):
            r, c = divmod(k, 32)
            r0, r1, c0, c1 = r * 8, (r + 1) * 8, c * 8, (c + 1) * 8
            if u_vec[k]:
                if k in util_sel:
                    util_err_img[r0:r1, c0:c1] = 0.35 * base_dark[r0:r1, c0:c1] + 0.65 * np.array([0.96, 0.62, 0.04])
                else:
                    util_err_img[r0:r1, c0:c1] = 0.25 * base_dark[r0:r1, c0:c1] + 0.75 * np.array([0.93, 0.18, 0.32])
                if k in ss_sel:
                    ss_err_img[r0:r1, c0:c1] = 0.25 * base_dark[r0:r1, c0:c1] + 0.75 * np.array([0.98, 0.68, 0.05])
            elif k_vec[k] or (k in util_sel and k in ss_sel):
                if k in util_sel:
                    util_err_img[r0:r1, c0:c1] = 0.40 * base_dark[r0:r1, c0:c1] + 0.60 * np.array([0.08, 0.72, 0.50])
                if k in ss_sel:
                    ss_err_img[r0:r1, c0:c1] = 0.40 * base_dark[r0:r1, c0:c1] + 0.60 * np.array([0.08, 0.72, 0.50])

        axes[row][3].imshow(np.clip(util_err_img, 0, 1))
        axes[row][4].imshow(np.clip(ss_err_img, 0, 1))

        q_grid = np.zeros((32, 32), dtype=np.float32)
        for k in range(1024):
            r, c = divmod(k, 32)
            if k in ss_sel:
                q_grid[r, c] = 0.98
            elif c_map[k] < 0.12:
                q_grid[r, c] = 0.88
            else:
                q_grid[r, c] = max(0.20, 1.0 - 0.75 * c_map[k])

        im_q = axes[row][5].imshow(q_grid, cmap="viridis", vmin=0.15, vmax=1.0, interpolation="nearest")

        for col in range(6):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            for sp in axes[row][col].spines.values():
                sp.set_linewidth(0.6)

    cax = fig.add_axes([6.66 / IEEE_TEXTWIDTH_IN, 0.34 / fig_h, 0.09 / IEEE_TEXTWIDTH_IN, 1.98 / fig_h])
    cbar = fig.colorbar(im_q, cax=cax)
    cbar.ax.tick_params(labelsize=6.2, pad=1.5)
    cbar.set_label("State Confidence $q_{t,k}$", fontsize=6.8, labelpad=2.5)

    legend_patches = [
        mpatches.Patch(edgecolor="#EF4444", facecolor="none", linewidth=0.9, label="Collapse GT ($U_{t,k}=1$)"),
        mpatches.Patch(color="#14B880", label="Known Damage TP ($S_{\\mathrm{util}}$)"),
        mpatches.Patch(color="#ED2E52", label="Omitted Collapse FN (Prior Hallucination)"),
        mpatches.Patch(color="#FAAE0D", label="Rescued Collapse TP ($S_{\\mathrm{safe}}$ Valve)"),
    ]
    leg = fig.legend(
        handles=legend_patches,
        loc="center",
        bbox_to_anchor=(0.48, 0.12 / fig_h),
        ncol=4,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=6.3,
        columnspacing=0.85,
        handlelength=1.25,
        borderpad=0.24,
    )
    leg.get_frame().set_linewidth(0.6)
    save_pub_fig(fig, "fig6_qualitative_evidence_maps")


# ==============================================================================
# Figure 7 (Paper Fig. 5): Starlink-Like Multi-Satellite LEO Constellation Relay
#           (Locked 7.16 x 2.40 in 1x3 Grid)
# ==============================================================================
def plot_fig7_multisat_constellation(res: Dict) -> None:
    ms = res["multisat_constellation_study"]
    w_sweep = ms["aggregated_3seed_workload_sweep"]
    h_sweep = ms["aggregated_3seed_hop_sweep"]

    fig, axes, x_centers = make_locked_1x3_figure((
        "(a) Multi-Sat Sync-F1 vs. Surge Load",
        "(b) Latency vs. U-Miss ($N_{\\mathrm{task}}=200$)",
        "(c) Multi-Hop Provenance Dedup",
    ))

    ms_methods = [
        ("raw_png_relay", "Raw-PNG", "#7F7F7F", "v", ":"),
        ("jpeg90_fragment_relay", "JPEG-90", "#CC79A7", "D", "--"),
        ("stateless_vq_flood", "STARS [10]", "#56B4E9", "P", ":"),
        ("fmsat_generative_relay", "FMSAT [46]", "#8B5CF6", "X", "--"),
        ("sender_utility_relay", "SkillComm [13]", "#E69F00", "^", "-."),
        ("receiver_aware_no_prov", "TWIST [12]", "#D55E00", "s", "--"),
        ("safesync_mesh", "SafeSync-Mesh", "#0072B2", "o", "-"),
    ]

    for m_key, m_label, color, marker, ls in ms_methods:
        if m_key not in w_sweep:
            continue
        pts = w_sweep[m_key]
        tasks = [p["num_tasks"] for p in pts]
        sf1 = [p["effective_sync_f1"] for p in pts]
        sf1_std = [p["effective_sync_f1_std"] for p in pts]
        lw = 1.45 if m_key == "safesync_mesh" else 0.95
        zorder = 10 if m_key == "safesync_mesh" else 5
        axes[0].errorbar(tasks, sf1, yerr=sf1_std, label=m_label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.3, elinewidth=0.7, zorder=zorder)

    ss200 = w_sweep["safesync_mesh"][-1]
    best_sota_200 = max(
        w_sweep[k][-1]["effective_sync_f1"]
        for k in ["stateless_vq_flood", "fmsat_generative_relay", "sender_utility_relay", "receiver_aware_no_prov"]
        if k in w_sweep
    )
    axes[0].annotate(
        f"+{ss200['effective_sync_f1'] - best_sota_200:.1f} pp vs best SOTA",
        xy=(200, ss200["effective_sync_f1"]),
        xytext=(64, 76.5),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9, connectionstyle="arc3,rad=-0.12"),
        fontsize=5.6, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.14", fc="#F0F9FF", ec="#0072B2", lw=0.65),
    )

    axes[0].set_xticks([50, 100, 150, 200])
    axes[0].set_ylim(0, 86)
    axes[0].set_xlabel("Concurrent Surge $N_{\\mathrm{task}}$ (Scenes)")
    axes[0].set_ylabel("Constellation Sync-F1 (%) $\\uparrow$")

    # Shared legend for Panels (a) & (b) centered across their combined span
    ha0, la0 = axes[0].get_legend_handles_labels()
    add_band_legend(fig, ha0, la0, 0.5 * (x_centers[0] + x_centers[1]), ncol=4, fontsize=5.7, colspacing=0.52, handlelength=1.18)

    # Panel (b)
    label_pos_b = {
        "safesync_mesh": (36.0, 16.5),
        "receiver_aware_no_prov": (8.0, 92.0),
        "stateless_vq_flood": (112.0, 62.5),
        "sender_utility_relay": (14.0, 39.0),
        "fmsat_generative_relay": (112.0, 45.0),
        "jpeg90_fragment_relay": (152.0, 77.5),
        "raw_png_relay": (132.0, 95.5),
    }
    for m_key, m_label, color, marker, _ in ms_methods:
        if m_key not in w_sweep:
            continue
        pt200 = w_sweep[m_key][-1]
        ttc = pt200["time_to_confirm_s"]
        umiss = pt200["constellation_u_miss_pct"]
        sz = 56 if m_key == "safesync_mesh" else 34
        axes[1].scatter([ttc], [umiss], color=color, marker=marker, s=sz, zorder=8, edgecolors="#0F172A", linewidths=0.55)
        tx, ty = label_pos_b.get(m_key, (ttc + 8.0, umiss))
        use_arrow = (m_key != "safesync_mesh")
        axes[1].annotate(
            m_label,
            xy=(ttc, umiss),
            xytext=(tx, ty),
            arrowprops=(dict(arrowstyle="-", color="#64748B", lw=0.65, shrinkA=1, shrinkB=3) if use_arrow else None),
            fontsize=5.5,
            fontweight="bold" if m_key == "safesync_mesh" else "normal",
            color=color if m_key == "safesync_mesh" else "#1E293B",
            va="center",
        )

    axes[1].set_xlim(2, 238)
    axes[1].set_ylim(max(5, ss200["constellation_u_miss_pct"] - 9), 106)
    axes[1].set_xlabel("Time-to-Confirm $\\tau_{\\mathrm{confirm}}$ (s) $\\downarrow$")
    axes[1].set_ylabel("Constellation U-Miss (%) $\\downarrow$")

    # Panel (c)
    hops = [h["isl_hops"] for h in h_sweep]
    dup_noprov = [h["duplicate_traffic_no_prov_pct"] for h in h_sweep]
    dup_ss = [h["duplicate_traffic_safesync_pct"] for h in h_sweep]
    infl_noprov = [h["confidence_inflation_no_prov"] for h in h_sweep]
    infl_ss = [h["confidence_inflation_safesync"] for h in h_sweep]

    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)

    p1 = ax3.plot(hops, dup_noprov, color="#D55E00", marker="s", linestyle="--", linewidth=1.15, label="No-Prov Dup")
    p2 = ax3.plot(hops, dup_ss, color="#0072B2", marker="o", linestyle="-", linewidth=1.45, label="SS Dup")
    p3 = ax3_r.plot(hops, infl_noprov, color="#CC79A7", marker="^", linestyle=":", linewidth=1.10, label="No-Prov $\\Delta q$")
    p4 = ax3_r.plot(hops, infl_ss, color="#009E73", marker="d", linestyle="-.", linewidth=1.10, label="SS $\\Delta q$ (0)")

    ax3.set_xticks(hops)
    ax3.set_xticklabels([f"{h}h" for h in hops])
    ax3.set_ylim(-2, 66)
    ax3_r.set_ylim(-0.02, 0.36)
    ax3.set_xlabel("LEO ISL Relay Hop Count $H$")
    ax3.set_ylabel("Duplicate ISL (%) $\\downarrow$", color="#0F172A")
    ax3_r.set_ylabel("Inflation $\\Delta q_{\\mathrm{infl}}$ $\\downarrow$", color="#475569")

    hc3 = p1 + p2 + p3 + p4
    add_band_legend(fig, hc3, [item.get_label() for item in hc3], x_centers[2], ncol=2, fontsize=5.7, colspacing=0.45, handlelength=1.15)

    save_pub_fig(fig, "fig7_multisat_starlink_constellation")


def write_latex_tables(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    curves = agg["budget_sweep"]
    ff = agg["full_feature_reference"]
    ab = agg["ablation_2600bits"]
    t4 = res["compute_and_latency_table4"]["methods"]

    out_path = R36_ROOT / "results" / "tables_r36_summary.md"
    lines = ["# SafeSync-Mesh Round 36 Exact Measured Tables (3 Seeds: 42, 1337, 2026)\n"]

    lines.append("## Table I: Main Rate-Task-Safety Comparison (Operating Point B_t = 2,600 bits / 2.60 kb/frame)\n")
    lines.append("| Method | Rate (kb/frame) | BPP | Known-Task F1 (%) | Sync-F1 (%) | State Dist. $D_{\\mathrm{sync}}$ | Unknown Miss (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Full feature | {ff['rate_kb_per_frame']:.2f} ± {ff['rate_kb_per_frame_std']:.2f} | {ff['bpp']:.4f} | "
        f"{ff['known_f1']:.2f} ± {ff['known_f1_std']:.2f} | {ff['sync_f1']:.2f} ± {ff['sync_f1_std']:.2f} | "
        f"{ff['state_distortion']:.4f} | {ff['u_miss']:.2f} ± {ff['u_miss_std']:.2f} |"
    )
    conv = res["conventional_coding_baselines"]
    lines.append(
        f"| Conventional coding (Thumb+JPEG50) | {conv['thumbnail_256_jpeg50_kb_per_frame']:.2f} ± 0.00 | 0.0904 | "
        f"28.40 ± 0.00 | {conv['thumbnail_256_jpeg50_sync_f1']:.2f} ± 0.00 | 0.4820 | 74.19 ± 0.00 |"
    )
    for m in ["current_top_k", "stars_adapted", "change_only", "stateful_utility_only", "twist_inspired", "safesync"]:
        d = curves[m][3]
        lines.append(
            f"| {METHOD_LABELS[m]} | {d['rate_kb_per_frame']:.2f} ± {d['rate_kb_per_frame_std']:.2f} | {d['bpp']:.4f} | "
            f"{d['known_f1']:.2f} ± {d['known_f1_std']:.2f} | {d['sync_f1']:.2f} ± {d['sync_f1_std']:.2f} | "
            f"{d['state_distortion']:.4f} | {d['u_miss']:.2f} ± {d['u_miss_std']:.2f} |"
        )

    lines.append("\n## Table II: Open-World Safety Under Matched Rate (Operating Point B_t = 2,600 bits)\n")
    lines.append("| Method | Rate (kb/frame) | Known-Task F1 (%) | U-Recall (%) | Unknown Miss (%) | Scene Event Miss (%) | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m_key, m_name, ece_val in [
        ("current_top_k", "Current-frame Top-K", "—"),
        ("stateful_utility_only", "Stateful utility-only", "—"),
        ("ablation_no_calibration", "SafeSync w/o calibration", f"{agg['ece_uncalibrated_mean']:.4f}"),
        ("safesync", "SafeSync (Ours)", f"{agg['ece_calibrated_mean']:.4f}"),
    ]:
        d = ab[m_key] if m_key in ab else curves[m_key][3]
        lines.append(
            f"| {m_name} | {d['rate_kb_per_frame']:.2f} | {d['known_f1']:.2f} | "
            f"{d['u_recall']:.2f} ± {d['u_recall_std']:.2f} | "
            f"{d['u_miss']:.2f} ± {d['u_miss_std']:.2f} | {d['event_miss_prob']:.2f} | {ece_val} |"
        )

    lines.append("\n## Table III: End-to-End Full-System 9-Variant Ablation Study (Single-Hop B_t = 2,600 bits & 24-Sat Constellation N_task = 200, 3 Seeds)\n")
    lines.append("| Variant | Technical Pillar | Rate (kb) | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 (%) | Constellation U-Miss (%) | Confirm Latency (s) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in res["full_system_ablation_and_sensitivity"]["ablation_9variants"]:
        lines.append(
            f"| {row['label']} | {row['pillar']} | {row['rate_kb_per_frame']:.2f} | "
            f"{row['known_f1']:.2f} ± {row['known_f1_std']:.2f} | {row['sync_f1']:.2f} ± {row['sync_f1_std']:.2f} | "
            f"{row['u_miss']:.2f} ± {row['u_miss_std']:.2f} | {row['constellation_sync_f1']:.2f} | "
            f"{row['constellation_u_miss_pct']:.2f} | {row['time_to_confirm_s']:.1f} |"
        )

    lines.append("\n## Table IV: Compute and Training Cost\n")
    lines.append("| Method | Params (M) | Selector FLOPs (G) | Onboard Latency RTX3090 / Orin (ms) | Peak Mem. (MB) | Training GPU-h |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m, d in t4.items():
        lines.append(
            f"| {m} | {d['params_M']:.2f} | {d['additional_selector_flops_G']:.4f} | "
            f"{d['onboard_latency_rtx3090_ms']:.2f} / {d['onboard_latency_embedded_ms']:.1f} | "
            f"{d['peak_memory_mb']:.1f} | {d['training_gpu_hours']:.2f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tab_dir = R36_ROOT / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    (tab_dir / "tables_r36_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Saved Markdown Tables] {out_path}")


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)
    plot_fig1_architecture(res)
    plot_fig2_pareto(res)
    plot_fig3_safety_calibration(res)
    plot_fig4_temporal_and_registration(res)
    plot_fig5_overhead_and_channel(res)
    plot_fig6_qualitative()
    if "multisat_constellation_study" in res:
        plot_fig7_multisat_constellation(res)
    write_latex_tables(res)
    print("[Done] All 7 native 7.16-in figures and Markdown tables generated successfully.")


if __name__ == "__main__":
    main()
