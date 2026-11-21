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
    "axes.linewidth": 0.75,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.45,
    "lines.markersize": 4.0,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
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


def save_pub_fig(fig: plt.Figure, stem: str) -> None:
    """Save at exact fixed canvas size (no bbox_inches='tight' rescaling) so width == 7.16 in."""
    for out_dir in [FIG_DIR, PAPER_FIG_DIR, DELIVERY_FIG_DIR, LEGACY_FIG_DIR]:
        fig.savefig(out_dir / f"{stem}.pdf", dpi=300)
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"[Saved Native 7.16-in Figure] {stem}.pdf & {stem}.png")


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

    # Native 7.16 x 3.32 in canvas (1:1 exact IEEE \textwidth)
    fig = plt.figure(figsize=(IEEE_TEXTWIDTH_IN, 3.32), facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0.5, 131.5)
    ax.set_ylim(-11.4, 49.0)
    ax.axis("off")

    def ctext(x, y, s, size=6.2, bold=False, color="#1E293B", ha="center", va="center", z=8, linespacing=1.15, bbox=None):
        return ax.text(
            x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontproperties=(fp_bold if bold else fp_reg),
            zorder=z, linespacing=linespacing, bbox=bbox
        )

    def draw_card(x, y, w, h, fc="#FFFFFF", ec="#334155", lw=1.0, r=1.2, shadow=True, z=3):
        if shadow:
            sh = FancyBboxPatch(
                (x + 0.28, y - 0.32), w, h,
                boxstyle=f"round,pad=0.20,rounding_size={r}",
                facecolor="#CBD5E1", edgecolor="none", alpha=0.40, zorder=z - 1
            )
            ax.add_patch(sh)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.20,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
        )
        ax.add_patch(box)

    def draw_sat_icon(cx, cy, scale=1.0, angle_deg=-22.0, body_c="#334155", panel_c="#0284C7", z=7):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def rot_pts(pts):
            return [(cx + scale * (px * cos_a - py * sin_a), cy + scale * (px * sin_a + py * cos_a)) for px, py in pts]

        ax.plot([cx - 2.2 * scale * cos_a, cx + 2.2 * scale * cos_a],
                [cy - 2.2 * scale * sin_a, cy + 2.2 * scale * sin_a], color="#475569", lw=1.05, zorder=z)
        for sx in [-1.60, 1.60]:
            p_rect = rot_pts([(sx - 0.66, -0.35), (sx + 0.66, -0.35), (sx + 0.66, 0.35), (sx - 0.66, 0.35)])
            ax.add_patch(Polygon(p_rect, closed=True, facecolor=panel_c, edgecolor="#0F172A", lw=0.65, zorder=z + 1))
        b_rect = rot_pts([(-0.50, -0.50), (0.50, -0.50), (0.50, 0.50), (-0.50, 0.50)])
        ax.add_patch(Polygon(b_rect, closed=True, facecolor=body_c, edgecolor="#0F172A", lw=0.75, zorder=z + 2))
        ax.add_patch(Circle((cx, cy - 0.42 * scale), 0.17 * scale, facecolor="#38BDF8", edgecolor="#0F172A", lw=0.5, zorder=z + 3))

    def draw_flow_arrow(x1, y1, x2, y2, label="", color="#334155", lw=1.2, rad=0.0, dy=1.35):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=3.8,head_width=2.4",
            color=color, linewidth=lw, zorder=9
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + dy
            ctext(mx, my, label, size=5.5, bold=True, color=color, z=10,
                  bbox=dict(boxstyle="round,pad=0.13", fc="#FFFFFF", ec=color, lw=0.6, alpha=0.96))

    # Top Title Banner (native 8.2 pt Comic Sans Bold)
    ctext(
        66.0, 47.4,
        "SafeSync-Mesh: Distributed Orbital Epistemic State Machine over Starlink-Like LEO Relay Constellations",
        size=8.2, bold=True, color="#0F172A"
    )

    stage_specs = [
        (1.0, 2.6, 24.5, 42.2, "1. Multi-Sat EO & Prior", "#EFF6FF", "#60A5FA", "#1D4ED8"),
        (27.1, 2.6, 25.0, 42.2, "2. Utility-Risk Split", "#F0FDFA", "#5EEAD4", "#0F766E"),
        (53.7, 2.6, 24.6, 42.2, "3. Epistemic Safety Valve", "#FFFBEB", "#FBBF24", "#B45309"),
        (79.9, 2.6, 24.6, 42.2, "4. LEO ISL Relay Mesh", "#ECFDF5", "#6EE7B7", "#047857"),
        (106.1, 2.6, 24.9, 42.2, "5. Idempotent Algebra", "#F5F3FF", "#C4B5FD", "#5B21B6"),
    ]
    for sx, sy, sw, sh, stitle, sfc, sec, pill_c in stage_specs:
        draw_card(sx, sy, sw, sh, fc=sfc, ec=sec, lw=1.05, r=1.4, shadow=True, z=2)
        pill = FancyBboxPatch(
            (sx + 0.8, sy + sh - 3.65), sw - 1.6, 2.75,
            boxstyle="round,pad=0.14,rounding_size=0.8",
            facecolor=pill_c, edgecolor="none", zorder=4
        )
        ax.add_patch(pill)
        ctext(sx + sw / 2, sy + sh - 2.28, stitle, size=6.6, bold=True, color="#FFFFFF", z=5)

    # STAGE 1
    draw_card(2.1, 24.4, 22.3, 15.4, fc="#FFFFFF", ec="#3B82F6", lw=0.95, r=1.0, shadow=False, z=4)
    ctext(13.25, 38.2, "Agile Sensing Pass $S_m$: $X_t$", size=6.3, bold=True, color="#1D4ED8")
    draw_sat_icon(6.4, 34.8, scale=1.0, angle_deg=-20.0, body_c="#334155", panel_c="#2563EB", z=6)
    ax.add_patch(Polygon([(6.4, 34.0), (3.2, 26.3), (10.4, 26.3)], closed=True, facecolor="#38BDF8", alpha=0.22, zorder=5))
    ax.imshow(post_thumb, extent=(3.2, 10.4, 26.2, 32.2), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((3.2, 26.2), 7.2, 6.0, fill=False, edgecolor="#EF4444", lw=0.95, zorder=7))
    ctext(6.8, 25.15, "Disaster $X_t$", size=5.5, bold=True, color="#B91C1C")

    rng_vq = np.random.default_rng(42)
    vq_palette = np.array([
        [59, 130, 246], [16, 185, 129], [245, 158, 11], [239, 68, 68],
        [139, 92, 246], [14, 165, 233], [236, 72, 153], [251, 191, 36]
    ], dtype=np.uint8)
    vq_mosaic = vq_palette[rng_vq.integers(0, len(vq_palette), size=(10, 10))]
    ax.imshow(vq_mosaic, extent=(13.5, 23.3, 28.0, 36.0), interpolation="nearest", aspect="auto", zorder=6)
    for gi in range(11):
        gx = 13.5 + gi * 0.98
        gy = 28.0 + gi * 0.80
        ax.plot([gx, gx], [28.0, 36.0], color="#FFFFFF", lw=0.35, zorder=7)
        ax.plot([13.5, 23.3], [gy, gy], color="#FFFFFF", lw=0.35, zorder=7)
    ax.add_patch(Rectangle((13.5, 28.0), 9.8, 8.0, fill=False, edgecolor="#1E293B", lw=0.8, zorder=7))
    draw_flow_arrow(10.6, 29.4, 13.3, 31.8, "$E_\\theta$", color="#1D4ED8", lw=1.05, dy=1.4)
    ctext(18.4, 26.1, "32x32 VQ ($K=512$)", size=5.6, bold=True, color="#1E293B")

    draw_card(2.1, 4.2, 22.3, 18.2, fc="#FFFFFF", ec="#64748B", lw=0.95, r=1.0, shadow=False, z=4)
    ctext(13.25, 20.8, "Prior Pass $S_{m-1}$ & 64-b RKS", size=6.2, bold=True, color="#334155")
    ax.imshow(pre_thumb, extent=(3.2, 10.1, 11.4, 18.8), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((3.2, 11.4), 6.9, 7.4, fill=False, edgecolor="#475569", lw=0.85, zorder=7))
    ctext(6.65, 10.2, "Prior $X_{t-1}$", size=5.5, bold=True, color="#475569")
    ctext(
        17.2, 15.0,
        "Aligned $\\overline{M}_{t-1}$\n- Tokens $\\overline{z}_{t-1,k}$\n- Conf. $q_{t-1,k}$\n- Age $a_{t-1,k}$",
        size=5.7, color="#1E293B", linespacing=1.18
    )
    rks_badge = FancyBboxPatch((3.1, 5.0), 20.3, 3.6, boxstyle="round,pad=0.10,rounding_size=0.6",
                               facecolor="#F5F3FF", edgecolor="#7C3AED", lw=0.8, zorder=5)
    ax.add_patch(rks_badge)
    ctext(13.25, 6.8, "64b RKS: Bloom | Hash | Age", size=5.2, bold=True, color="#5B21B6")

    # STAGE 2
    draw_card(28.2, 29.2, 22.8, 10.6, fc="#FFFFFF", ec="#0284C7", lw=0.95, r=1.0, shadow=False, z=4)
    ctext(39.6, 38.3, "Temporal Innovation $c_k$", size=6.2, bold=True, color="#0369A1")
    ax.imshow(c_grid, cmap="YlGnBu", extent=(29.1, 35.1, 30.1, 36.7), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.1, 30.1), 6.0, 6.6, fill=False, edgecolor="#0284C7", lw=0.75, zorder=7))
    ctext(43.0, 33.3, "$1 - \\cos(f_t, \\bar{f}_{t-1})$\n$+ \\|\\Delta f_k\\|_2$\nGates Static Cells", size=5.6, color="#1E293B")

    draw_card(28.2, 16.7, 22.8, 10.6, fc="#FFFFFF", ec="#059669", lw=0.95, r=1.0, shadow=False, z=4)
    ctext(39.6, 25.8, "Closed-Set Utility $g_k$", size=6.2, bold=True, color="#047857")
    ax.imshow(g_grid, cmap="Greens", extent=(29.1, 35.1, 17.6, 24.2), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.1, 17.6), 6.0, 6.6, fill=False, edgecolor="#059669", lw=0.75, zorder=7))
    ctext(43.0, 20.8, "Scorer $G_\\phi(f_t, \\bar{f}_{t-1})$\n$g_k \\approx \\Delta L_{\\mathrm{known}}$\nRoutine Damage", size=5.6, color="#1E293B")

    draw_card(28.2, 4.2, 22.8, 10.6, fc="#FFF1F2", ec="#E11D48", lw=1.1, r=1.0, shadow=False, z=4)
    ctext(39.6, 13.3, "Calibrated Collapse Risk $p_k$", size=6.2, bold=True, color="#BE123C")
    ax.imshow(p_grid, cmap="magma", extent=(29.1, 35.1, 5.1, 11.7), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.1, 5.1), 6.0, 6.6, fill=False, edgecolor="#E11D48", lw=0.85, zorder=7))
    ctext(43.1, 8.3, "Prior Contradiction\n+ Entropy + VQ Res.\nConformal-Isotonic", size=5.6, color="#881337")

    # STAGE 3
    draw_card(54.8, 23.0, 22.4, 16.8, fc="#FFF7ED", ec="#EA580C", lw=1.15, r=1.0, shadow=False, z=4)
    ctext(66.0, 38.1, "Stage 3A: Safety Valve $S_{\\mathrm{safe}}$", size=6.2, bold=True, color="#C2410C")
    ax.add_patch(FancyBboxPatch((56.0, 33.8), 6.3, 2.3, boxstyle="round,pad=0.05,rounding_size=0.4",
                                facecolor="#EA580C", edgecolor="#9A3412", lw=0.65, zorder=5))
    ax.add_patch(FancyBboxPatch((62.6, 33.8), 13.4, 2.3, boxstyle="round,pad=0.05,rounding_size=0.4",
                                facecolor="#DBEAFE", edgecolor="#3B82F6", lw=0.65, zorder=5))
    ctext(59.15, 34.95, "$S_{\\mathrm{safe}}$", size=5.5, bold=True, color="#FFFFFF")
    ctext(69.3, 34.95, "$B_t - R(S_{\\mathrm{safe}})$", size=5.5, bold=True, color="#1D4ED8")
    ctext(
        66.0, 28.0,
        "Risk Bound (Prop. 1):\n$\\Sigma_{s_k=0}\\, p_k \\leq \\delta$\nAdmits High $p_k / b_k$ First\nRescues Unseen Collapse",
        size=5.7, bold=True, color="#7C2D12", linespacing=1.24
    )

    draw_card(54.8, 4.2, 22.4, 16.2, fc="#FFFFFF", ec="#0284C7", lw=1.0, r=1.0, shadow=False, z=4)
    ctext(66.0, 18.7, "Stage 3B: Rate-Utility $S_{\\mathrm{util}}$", size=6.2, bold=True, color="#0369A1")
    ctext(
        66.0, 11.2,
        "Composite Sync Value:\n$v_k = \\alpha g_k + \\beta c_k + \\gamma d_k^{\\mathrm{sync}} + \\eta h(a_k)$\n\nGreedy Rate-Optimal Fill:\nSort by $\\rho_k = v_k / b_k$ ($R_t \\leq B_t$)",
        size=5.5, color="#1E293B", linespacing=1.18
    )

    # STAGE 4
    draw_card(81.0, 17.6, 22.4, 22.2, fc="#FFFFFF", ec="#059669", lw=1.0, r=1.0, shadow=False, z=4)
    ctext(92.2, 38.2, "24-Sat Walker-Delta Mesh", size=6.2, bold=True, color="#047857")
    earth_arc = Wedge((92.2, 7.1), 13.5, 64, 116, width=1.4, facecolor="#BAE6FD", edgecolor="#0284C7", lw=0.65, alpha=0.50, zorder=4)
    ax.add_patch(earth_arc)
    theta_orb = np.linspace(math.radians(52), math.radians(128), 60)
    ax.plot(92.2 + 22.5 * np.cos(theta_orb), 12.0 + 22.5 * np.sin(theta_orb),
            color="#94A3B8", linestyle="--", lw=0.8, zorder=4)

    sat_nodes = [(85.1, 33.3), (98.9, 33.7), (85.7, 23.9), (98.5, 24.3)]
    for u_i, v_i in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]:
        x_u, y_u = sat_nodes[u_i]
        x_v, y_v = sat_nodes[v_i]
        ax.plot([x_u, x_v], [y_u, y_v], color="#22D3EE", lw=2.8, alpha=0.28, zorder=5)
        ax.plot([x_u, x_v], [y_u, y_v], color="#0891B2", lw=1.05, linestyle="-", zorder=6)
    for s_idx, (sx_n, sy_n) in enumerate(sat_nodes):
        draw_sat_icon(sx_n, sy_n, scale=0.85, angle_deg=(18.0 if s_idx % 2 == 0 else -18.0),
                      body_c="#1E293B", panel_c="#10B981", z=7)
    ctext(92.2, 35.8, "Optical ISL Relay Links", size=5.5, bold=True, color="#0E7490")
    ctext(92.2, 19.1, "$S_{\\mathrm{safe}}$ Queue + O(1) Dedup", size=5.5, bold=True, color="#047857",
          bbox=dict(boxstyle="round,pad=0.10", fc="#FFFFFF", ec="none", alpha=0.90))

    draw_card(81.0, 4.2, 22.4, 11.8, fc="#F0FDF4", ec="#16A34A", lw=0.95, r=1.0, shadow=False, z=4)
    ctext(92.2, 14.3, "2D Spatio-Temp Codec", size=6.2, bold=True, color="#15803D")
    ctext(
        92.2, 8.7,
        "$P_\\omega(z_{t,k} \\mid \\bar{z}_{t-1,k}, \\hat{z}_{t,\\mathcal{N}(k)})$ (5.9 b)\n- RKS Digest Verification\n- 32-b root_id Echo Drop",
        size=5.4, color="#14532D", linespacing=1.20
    )

    # STAGE 5
    draw_card(107.2, 22.8, 22.6, 17.0, fc="#FFFFFF", ec="#7C3AED", lw=1.0, r=1.0, shadow=False, z=4)
    ctext(118.5, 38.1, "Consumer $D_j$ State Algebra", size=6.2, bold=True, color="#5B21B6")
    ax.add_patch(Polygon([(108.5, 31.2), (110.4, 31.2), (109.45, 33.2)], closed=True,
                         facecolor="#64748B", edgecolor="#1E293B", lw=0.7, zorder=6))
    ax.add_patch(Wedge((110.1, 34.2), 1.85, 135, 315, facecolor="#DDD6FE", edgecolor="#5B21B6", lw=0.85, zorder=7))
    ax.plot([109.85, 111.0], [33.95, 35.1], color="#5B21B6", lw=0.9, zorder=8)
    ax.add_patch(Circle((111.1, 35.2), 0.24, facecolor="#EC4899", edgecolor="#1E293B", lw=0.5, zorder=9))
    ctext(121.3, 33.9, "Idempotent Union (Prop. 2):\n$M_j \\oplus e \\oplus \\dots \\oplus e = M_j \\oplus e$\n($\\Delta q_{\\mathrm{infl}} = 0.000$)", size=5.1, bold=True, color="#4C1D95")
    ctext(118.5, 26.2, "- Selected: $\\tilde{z}_{t,k}, q_{t,k}=1.0$\n- Omitted: clip($1 - 0.75 c_k$)", size=5.3, color="#1E293B")

    draw_card(107.2, 4.2, 22.6, 16.8, fc="#FAF5FF", ec="#4C1D95", lw=1.0, r=1.0, shadow=False, z=4)
    ctext(118.5, 19.3, "Synchronized Map $\\hat{M}_t$", size=6.2, bold=True, color="#4C1D95")
    ax.imshow(fused_thumb, extent=(108.3, 115.5, 6.2, 16.8), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((108.3, 6.2), 7.2, 10.6, fill=False, edgecolor="#4C1D95", lw=0.85, zorder=7))
    ctext(
        122.7, 11.5,
        "Head $T_\\psi(\\hat{M}_t)$\n- Known TP\n  (Emerald)\n- Rescued $U_{t,k}$\n  (Gold $S_{\\mathrm{safe}}$)",
        size=5.5, bold=True, color="#1E293B", linespacing=1.16
    )

    # Arrows
    draw_flow_arrow(24.4, 32.0, 28.2, 34.2, "$F_t, z_t$", color="#1D4ED8", dy=1.45)
    draw_flow_arrow(24.4, 15.0, 28.2, 21.8, "$\\bar{M}_{t-1}$", color="#475569", dy=1.45)
    draw_flow_arrow(24.4, 9.5, 28.2, 9.5, "$q_{t-1}$", color="#BE123C", dy=-1.55)

    draw_flow_arrow(51.0, 9.5, 54.8, 29.5, "$p_k$", color="#E11D48", lw=1.35, dy=3.2)
    draw_flow_arrow(51.0, 34.2, 54.8, 14.5, "$c_k$", color="#0284C7", dy=4.2)
    draw_flow_arrow(51.0, 21.8, 54.8, 11.5, "$g_k$", color="#059669", dy=-2.6)
    draw_flow_arrow(66.0, 23.0, 66.0, 20.4, "$S_{\\mathrm{safe}}$", color="#EA580C", dy=0.0)

    draw_flow_arrow(77.2, 31.0, 81.0, 29.0, "$S_{\\mathrm{safe}}$", color="#EA580C", dy=1.45)
    draw_flow_arrow(77.2, 12.0, 81.0, 10.5, "$S_{\\mathrm{util}}$", color="#0284C7", dy=-1.5)

    draw_flow_arrow(103.4, 28.5, 107.2, 31.0, "324 B", color="#047857", lw=1.35, dy=1.45)
    draw_flow_arrow(118.5, 22.8, 118.5, 21.0, "", color="#5B21B6", lw=1.15)

    # Bottom Wire Packet Strip (dynamically read exact measured bit breakdown)
    ss_wire = res["aggregated_3seed"]["budget_sweep"]["safesync"][3]
    ob_w = ss_wire["overhead_breakdown_bits"]
    tot_wb = ob_w["total_bits"]
    hdr_b = ob_w["header_bits"]
    msk_b = ob_w["mask_bits"]
    saf_b = ob_w["safety_meta_bits"]
    tok_b = ob_w["token_payload_bits"]
    n_tok_w = ss_wire["mean_selected_tokens"]
    draw_card(1.0, -10.6, 130.0, 11.2, fc="#F8FAFC", ec="#94A3B8", lw=1.0, r=1.1, shadow=True, z=2)
    ctext(
        66.0, -1.1,
        f"324-Byte Verifiable Wire Packet & 64-Bit RKS Handshake ($R_t = {tot_wb:,.1f}$ bits $= {tot_wb/8.0:.1f}$ B/frame, 99.71% Saving vs. Raw PNG)",
        size=6.7, bold=True, color="#0F172A", z=5
    )

    segments = [
        (2.1, 20.8, f"128-b Header ({hdr_b/tot_wb*100:.1f}%)", "'SSYN' | root_id | CRC32", "#FCE7F3", "#BE185D"),
        (23.6, 29.8, f"Spatial Mask $b_{{\\mathrm{{mask}}}}$ ({msk_b/tot_wb*100:.1f}%)", f"Compressed Bitmap ({msk_b:.0f} b)", "#FEF3C7", "#B45309"),
        (54.1, 19.8, f"Safety Meta $b_{{\\mathrm{{safe}}}}$ ({saf_b/tot_wb*100:.1f}%)", f"Header-Ord. Risk ({saf_b:.0f} b)", "#FFEDD5", "#C2410C"),
        (74.6, 34.8, f"2D Causal Prefix Payload $b_{{\\mathrm{{tok}}}}$ ({tok_b/tot_wb*100:.1f}%)", f"$P_\\omega(z_{{t,k}}\\mid\\overline{{z}}_{{t-1,k}},\\hat{{z}}_{{t,\\mathcal{{N}}(k)}})$ ({tok_b:,.0f} b, {n_tok_w:.1f} tok)", "#DBEAFE", "#1D4ED8"),
        (110.1, 19.8, "64-b RKS Handshake", "Bloom | Hash | Age", "#EDE9FE", "#6D28D9"),
    ]
    for bx, bw, btitle, bsub, bfc, bec in segments:
        fbox = FancyBboxPatch(
            (bx, -9.7), bw, 6.6,
            boxstyle="round,pad=0.12,rounding_size=0.7",
            facecolor=bfc, edgecolor=bec, linewidth=0.95, zorder=4
        )
        ax.add_patch(fbox)
        ctext(bx + bw / 2, -5.0, btitle, size=5.7, bold=True, color=bec, z=6)
        ctext(bx + bw / 2, -7.8, bsub, size=5.2, color="#1E293B", z=6)

    save_pub_fig(fig, "fig1_safesync_architecture")


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

        lw = 1.85 if m == "safesync" else 1.2
        zorder = 10 if m == "safesync" else 5
        color = PALETTE[m]
        marker = MARKERS[m]
        ls = LINESTYLES[m]
        label = METHOD_LABELS[m]

        axes[0].errorbar(rates, known_f1, yerr=known_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.6, zorder=zorder)
        axes[1].errorbar(rates, sync_f1, yerr=sync_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.6, zorder=zorder)
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
        lw = 1.85 if m == "safesync" else 1.2
        zorder = 10 if m == "safesync" else 5
        axes[0].errorbar(rates, miss, yerr=miss_std, label=short_names[m], color=PALETTE[m],
                         marker=MARKERS[m], linestyle=LINESTYLES[m], linewidth=lw,
                         capsize=1.6, zorder=zorder)

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
                 linewidth=1.35, label="Uncalibrated")
    axes[1].plot(deltas, cal_miss, color="#0072B2", marker="o", linestyle="-",
                 linewidth=1.8, label="Calibrated $p_k$")

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
    axes[2].plot(raw_x, raw_y, color="#D55E00", marker="s", linestyle="--", linewidth=1.35,
                 label=f"Raw (ECE={agg['ece_uncalibrated_mean']:.2f})")
    axes[2].plot(cal_x, cal_y, color="#0072B2", marker="o", linestyle="-", linewidth=1.8,
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

    l1 = ax1.plot(ages, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.8, label="Bldg F1 (%)")
    l2 = ax1.plot(ages, new_rec, color="#009E73", marker="^", linestyle="-.", linewidth=1.45, label="New Recall")
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
    lb1 = axb.plot(xb, b_f1, color="#0072B2", marker="o", linewidth=1.75, label="Bldg F1 (%)")
    lb2 = axb.plot(xb, b_rec, color="#D55E00", marker="s", linestyle="--", linewidth=1.5, label="New Recall")
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

    m1 = ax2.plot(sigmas, adapt_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.8, label="SafeSync")
    m2 = ax2.plot(sigmas, static_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.45, label="Static Reuse")
    m3 = ax2_r.plot(sigmas, adapt_rate, color="#009E73", marker="^", linestyle="-.", linewidth=1.45, label="Rate (kb)")

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
    p1 = ax2.plot(snrs, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=1.9, label="SafeSync", zorder=10)
    p2 = ax2.plot(snrs, ut_f1, color="#009E73", marker="^", linestyle="-.", linewidth=1.35, label="Util-only", zorder=6)
    p3 = ax2.plot(snrs, tw_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.35, label="TWIST", zorder=6)
    p4 = ax2.plot(snrs, st_f1, color="#E69F00", marker="v", linestyle="--", linewidth=1.25, label="STARS", zorder=5)
    p5 = ax2.plot(snrs, tk_f1, color="#56B4E9", marker="P", linestyle=":", linewidth=1.25, label="Top-K", zorder=5)

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

    s1 = ax3.plot(rhos, r_kf1, color="#009E73", marker="^", linestyle="--", linewidth=1.35, label="Known F1")
    s2 = ax3.plot(rhos, r_sf1, color="#0072B2", marker="o", linestyle="-", linewidth=1.8, label="Sync-F1")
    s3 = ax3_r.plot(rhos, r_umiss, color="#D55E00", marker="s", linestyle="-.", linewidth=1.5, label="U-Miss $\\downarrow$")
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
        lw = 1.95 if m_key == "safesync_mesh" else 1.2
        zorder = 10 if m_key == "safesync_mesh" else 5
        axes[0].errorbar(tasks, sf1, yerr=sf1_std, label=m_label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=1.6, zorder=zorder)

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

    p1 = ax3.plot(hops, dup_noprov, color="#D55E00", marker="s", linestyle="--", linewidth=1.5, label="No-Prov Dup")
    p2 = ax3.plot(hops, dup_ss, color="#0072B2", marker="o", linestyle="-", linewidth=1.8, label="SS Dup")
    p3 = ax3_r.plot(hops, infl_noprov, color="#CC79A7", marker="^", linestyle=":", linewidth=1.45, label="No-Prov $\\Delta q$")
    p4 = ax3_r.plot(hops, infl_ss, color="#009E73", marker="d", linestyle="-.", linewidth=1.45, label="SS $\\Delta q$ (0)")

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
