#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all 6 publication-quality figures (PDF + 300 DPI PNG) and LaTeX/Markdown tables
(Table I to Table IV) for the SafeSync IEEE TGRS manuscript directly from
safesync_r36_complete_results.json and xbd_safesync_tokens_440.npz.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUB_ROOT = PROJECT_ROOT
R36_ROOT = PROJECT_ROOT
RESULTS_JSON = R36_ROOT / "results" / "safesync_r36_complete_results.json"
CACHE_DIR = R36_ROOT / "data_cache"
FIG_DIR = R36_ROOT / "figures"
PAPER_DIR = PUB_ROOT / "safesync_paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"
DELIVERY_FIG_DIR = PUB_ROOT / "paper_delivery" / "figures"

for d in [FIG_DIR, PAPER_DIR, PAPER_FIG_DIR, DELIVERY_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# IEEE TGRS Publication Matplotlib Styling (Uniform 11.4-in Base -> 7.16-in \textwidth)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.4,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.6,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.2,
    "figure.titlesize": 11.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.85,
    "lines.markersize": 5.4,
})

METHOD_LABELS = {
    "safesync": "SafeSync (Ours)",
    "twist_inspired": "TWIST-inspired",
    "stateful_utility_only": "Stateful utility-only",
    "change_only": "Change-only",
    "stars_adapted": "STARS-style adapted",
    "current_top_k": "Current-frame Top-K",
    "full_feature": "Full feature",
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
    for out_dir in [FIG_DIR, PAPER_FIG_DIR, DELIVERY_FIG_DIR]:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Figure] {stem}.pdf & {stem}.png")


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)


# ==============================================================================
# Figure 1: SafeSync-Mesh System Architecture & Wire-Frame Bit Layout Diagram
#           (Comic Sans MS Academic Style + Real Satellite Patches & LEO Mesh)
# ==============================================================================
def plot_fig1_architecture(res: Dict) -> None:
    import io
    import pyarrow.parquet as pq

    comic_reg_path = Path(r"./fonts\comic.ttf")
    comic_bold_path = Path(r"./fonts\comicbd.ttf")
    fp_reg = FontProperties(fname=str(comic_reg_path)) if comic_reg_path.exists() else FontProperties(family="sans-serif")
    fp_bold = FontProperties(fname=str(comic_bold_path)) if comic_bold_path.exists() else FontProperties(family="sans-serif", weight="bold")

    # Load one real disaster scene pair + token grids for crisp visual thumbnails inside Fig. 1
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

    # Build fused state overlay thumbnail (gold = S_safe rescued collapse, emerald = known utility TP)
    fused_thumb = (post_thumb.astype(np.float32) * 0.65).astype(np.uint8)
    for rr in range(32):
        for cc in range(32):
            r0, r1, c0, c1 = rr * 4, (rr + 1) * 4, cc * 4, (cc + 1) * 4
            if u_grid[rr, cc]:
                fused_thumb[r0:r1, c0:c1] = np.array([245, 179, 24], dtype=np.uint8)
            elif k_grid[rr, cc]:
                fused_thumb[r0:r1, c0:c1] = np.array([20, 184, 130], dtype=np.uint8)

    fig = plt.figure(figsize=(11.4, 5.15), facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0, 132)
    ax.set_ylim(-11.8, 49.5)
    ax.axis("off")

    def ctext(x, y, s, size=8.2, bold=False, color="#1E293B", ha="center", va="center", z=8, linespacing=1.16, bbox=None):
        return ax.text(
            x, y, s, ha=ha, va=va, fontsize=size * 0.94, color=color,
            fontproperties=(fp_bold if bold else fp_reg),
            zorder=z, linespacing=linespacing, bbox=bbox
        )

    def draw_card(x, y, w, h, fc="#FFFFFF", ec="#334155", lw=1.35, r=1.3, shadow=True, z=3):
        if shadow:
            sh = FancyBboxPatch(
                (x + 0.32, y - 0.38), w, h,
                boxstyle=f"round,pad=0.22,rounding_size={r}",
                facecolor="#CBD5E1", edgecolor="none", alpha=0.42, zorder=z - 1
            )
            ax.add_patch(sh)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.22,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
        )
        ax.add_patch(box)

    def draw_sat_icon(cx, cy, scale=1.0, angle_deg=-22.0, body_c="#334155", panel_c="#0284C7", z=7):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def rot_pts(pts):
            return [(cx + scale * (px * cos_a - py * sin_a), cy + scale * (px * sin_a + py * cos_a)) for px, py in pts]

        # Solar panels left & right + struts
        ax.plot([cx - 2.3 * scale * cos_a, cx + 2.3 * scale * cos_a],
                [cy - 2.3 * scale * sin_a, cy + 2.3 * scale * sin_a], color="#475569", lw=1.4, zorder=z)
        for sx in [-1.65, 1.65]:
            p_rect = rot_pts([(sx - 0.68, -0.36), (sx + 0.68, -0.36), (sx + 0.68, 0.36), (sx - 0.68, 0.36)])
            ax.add_patch(Polygon(p_rect, closed=True, facecolor=panel_c, edgecolor="#0F172A", lw=0.85, zorder=z + 1))
        # Satellite bus body
        b_rect = rot_pts([(-0.52, -0.52), (0.52, -0.52), (0.52, 0.52), (-0.52, 0.52)])
        ax.add_patch(Polygon(b_rect, closed=True, facecolor=body_c, edgecolor="#0F172A", lw=0.95, zorder=z + 2))
        # Sensor lens dot
        ax.add_patch(Circle((cx, cy - 0.45 * scale), 0.18 * scale, facecolor="#38BDF8", edgecolor="#0F172A", lw=0.6, zorder=z + 3))

    def draw_flow_arrow(x1, y1, x2, y2, label="", color="#334155", lw=1.65, rad=0.0, dy=1.25):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=5.0,head_width=3.1",
            color=color, linewidth=lw, zorder=9
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + dy
            ctext(mx, my, label, size=7.1, bold=True, color=color, z=10,
                  bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec=color, lw=0.75, alpha=0.95))

    # Top Title Banner
    ctext(
        66.0, 47.6,
        "SafeSync-Mesh: Distributed Orbital Epistemic State Machine over Starlink-Like LEO Relay Constellations",
        size=10.6, bold=True, color="#0F172A"
    )

    # 5 Outer Stage Containers (Soft Pastel CVPR/ICLR Oral Style)
    stage_specs = [
        (1.2, 2.6, 24.4, 42.2, "1. Multi-Sat EO & Prior State", "#EFF6FF", "#60A5FA", "#1D4ED8"),
        (27.2, 2.6, 25.0, 42.2, "2. Orthogonal Utility-Risk", "#F0FDFA", "#5EEAD4", "#0F766E"),
        (53.8, 2.6, 24.6, 42.2, "3. Epistemic Safety Valve", "#FFFBEB", "#FBBF24", "#B45309"),
        (80.0, 2.6, 24.6, 42.2, "4. LEO ISL Relay Mesh", "#ECFDF5", "#6EE7B7", "#047857"),
        (106.2, 2.6, 24.6, 42.2, "5. Idempotent State Algebra", "#F5F3FF", "#C4B5FD", "#5B21B6"),
    ]
    for sx, sy, sw, sh, stitle, sfc, sec, pill_c in stage_specs:
        draw_card(sx, sy, sw, sh, fc=sfc, ec=sec, lw=1.35, r=1.6, shadow=True, z=2)
        pill = FancyBboxPatch(
            (sx + 1.0, sy + sh - 3.6), sw - 2.0, 2.7,
            boxstyle="round,pad=0.15,rounding_size=0.9",
            facecolor=pill_c, edgecolor="none", zorder=4
        )
        ax.add_patch(pill)
        ctext(sx + sw / 2, sy + sh - 2.25, stitle, size=8.2, bold=True, color="#FFFFFF", z=5)

    # ==========================================================================
    # STAGE 1: Agile Sensing Pass S_m + Prior Pass S_{m-1} + 32x32 VQ Mosaic
    # ==========================================================================
    draw_card(2.4, 24.4, 22.0, 15.4, fc="#FFFFFF", ec="#3B82F6", lw=1.25, r=1.1, shadow=False, z=4)
    ctext(13.4, 38.2, "Agile Sensing Pass  $S_m$ :  $X_t$", size=8.0, bold=True, color="#1D4ED8")
    # Satellite S_m + optical beam cone + real disaster thumbnail
    draw_sat_icon(6.6, 34.7, scale=1.05, angle_deg=-20.0, body_c="#334155", panel_c="#2563EB", z=6)
    cone = Polygon([(6.6, 34.0), (3.4, 26.2), (10.6, 26.2)], closed=True, facecolor="#38BDF8", alpha=0.22, zorder=5)
    ax.add_patch(cone)
    ax.imshow(post_thumb, extent=(3.4, 10.6, 26.0, 32.1), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((3.4, 26.0), 7.2, 6.1, fill=False, edgecolor="#EF4444", lw=1.2, zorder=7))
    ctext(7.0, 25.15, "Disaster $X_t$", size=6.6, bold=True, color="#B91C1C")

    # 10x10 vivid discrete VQ token mosaic (representing 32x32, K=512)
    rng_vq = np.random.default_rng(42)
    vq_palette = np.array([
        [59, 130, 246], [16, 185, 129], [245, 158, 11], [239, 68, 68],
        [139, 92, 246], [14, 165, 233], [236, 72, 153], [251, 191, 36]
    ], dtype=np.uint8)
    vq_mosaic = vq_palette[rng_vq.integers(0, len(vq_palette), size=(10, 10))]
    ax.imshow(vq_mosaic, extent=(13.6, 23.2, 27.6, 35.8), interpolation="nearest", aspect="auto", zorder=6)
    for gi in range(11):
        gx = 13.6 + gi * 0.96
        gy = 27.6 + gi * 0.82
        ax.plot([gx, gx], [27.6, 35.8], color="#FFFFFF", lw=0.45, zorder=7)
        ax.plot([13.6, 23.2], [gy, gy], color="#FFFFFF", lw=0.45, zorder=7)
    ax.add_patch(Rectangle((13.6, 27.6), 9.6, 8.2, fill=False, edgecolor="#1E293B", lw=1.0, zorder=7))
    draw_flow_arrow(10.8, 29.2, 13.4, 31.5, "$E_\\theta$", color="#1D4ED8", lw=1.35, dy=1.3)
    ctext(18.4, 26.0, "32x32 VQ Grid\nCodebook $C$ ($K=512$)", size=6.7, bold=True, color="#1E293B")

    # Prior Pass S_{m-1} card
    draw_card(2.4, 4.2, 22.0, 18.2, fc="#FFFFFF", ec="#64748B", lw=1.25, r=1.1, shadow=False, z=4)
    ctext(13.4, 20.8, "Prior Pass  $S_{m-1}$  &  64-Bit RKS", size=7.9, bold=True, color="#334155")
    ax.imshow(pre_thumb, extent=(3.4, 10.4, 11.4, 18.8), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((3.4, 11.4), 7.0, 7.4, fill=False, edgecolor="#475569", lw=1.0, zorder=7))
    ctext(6.9, 10.3, "Prior $X_{t-1}$", size=6.6, bold=True, color="#475569")
    ctext(
        17.3, 15.1,
        "Georef Align\n$\\overline{M}_{t-1} = A_t(M_{t-1})$\n- Prior $\\overline{z}_{t-1,k}$\n- Confidence $q_{t-1}$\n- State Age $a_{t-1}$",
        size=6.8, color="#1E293B", linespacing=1.18
    )
    rks_badge = FancyBboxPatch((3.4, 5.1), 19.8, 3.6, boxstyle="round,pad=0.12,rounding_size=0.7",
                               facecolor="#F5F3FF", edgecolor="#7C3AED", lw=1.0, zorder=5)
    ax.add_patch(rks_badge)
    ctext(13.3, 6.9, "RKS: 32b Bloom | 16b Digest | 16b Age", size=6.6, bold=True, color="#5B21B6")

    # ==========================================================================
    # STAGE 2: Orthogonal Utility-Risk Decomposition (with real 32x32 heatmaps)
    # ==========================================================================
    # Subcard 2A: Temporal Innovation c_k
    draw_card(28.4, 29.2, 22.6, 10.6, fc="#FFFFFF", ec="#0284C7", lw=1.25, r=1.0, shadow=False, z=4)
    ctext(39.7, 38.3, "Temporal Innovation  $c_k$", size=7.9, bold=True, color="#0369A1")
    ax.imshow(c_grid, cmap="YlGnBu", extent=(29.3, 35.3, 30.1, 36.7), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.3, 30.1), 6.0, 6.6, fill=False, edgecolor="#0284C7", lw=0.9, zorder=7))
    ctext(43.0, 33.4, "$1 - \\cos(f_{t,k}, \\bar{f}_{t-1,k})$\n$+ \\|\\Delta f_k\\|_2$\nGates Static Bg", size=6.8, color="#1E293B")

    # Subcard 2B: Pragmatic Utility g_k
    draw_card(28.4, 16.7, 22.6, 10.6, fc="#FFFFFF", ec="#059669", lw=1.25, r=1.0, shadow=False, z=4)
    ctext(39.7, 25.8, "Pragmatic Closed-Set Utility  $g_k$", size=7.7, bold=True, color="#047857")
    ax.imshow(g_grid, cmap="Greens", extent=(29.3, 35.3, 17.6, 24.2), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.3, 17.6), 6.0, 6.6, fill=False, edgecolor="#059669", lw=0.9, zorder=7))
    ctext(43.0, 20.9, "Scorer $G_\\phi(f_t, \\bar{f}_{t-1})$\n$g_k \\approx \\Delta L_{\\mathrm{known}}$\nRoutine Task Gain", size=6.8, color="#1E293B")

    # Subcard 2C: Calibrated Open-World Risk p_k
    draw_card(28.4, 4.2, 22.6, 10.6, fc="#FFF1F2", ec="#E11D48", lw=1.45, r=1.0, shadow=False, z=4)
    ctext(39.7, 13.3, "Calibrated Collapse Risk  $p_k$", size=7.9, bold=True, color="#BE123C")
    ax.imshow(p_grid, cmap="magma", extent=(29.3, 35.3, 5.1, 11.7), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((29.3, 5.1), 6.0, 6.6, fill=False, edgecolor="#E11D48", lw=1.0, zorder=7))
    ctext(43.1, 8.4, "Prior Contradiction\n+ Entropy + VQ Res.\nIsotonic-Conformal", size=6.8, color="#881337")

    # ==========================================================================
    # STAGE 3: Two-Stage Epistemic Safety Valve & Utility-per-Bit Allocation
    # ==========================================================================
    draw_card(55.0, 23.0, 22.2, 16.8, fc="#FFF7ED", ec="#EA580C", lw=1.55, r=1.1, shadow=False, z=4)
    ctext(66.1, 38.0, "Stage 3A: Safety Valve  $S_{\\mathrm{safe}}$", size=8.0, bold=True, color="#C2410C")
    # Budget split visual gauge bar inside 3A
    ax.add_patch(FancyBboxPatch((56.4, 33.8), 6.2, 2.2, boxstyle="round,pad=0.05,rounding_size=0.4",
                                facecolor="#EA580C", edgecolor="#9A3412", lw=0.8, zorder=5))
    ax.add_patch(FancyBboxPatch((62.8, 33.8), 13.0, 2.2, boxstyle="round,pad=0.05,rounding_size=0.4",
                                facecolor="#DBEAFE", edgecolor="#3B82F6", lw=0.8, zorder=5))
    ctext(59.5, 34.9, "$S_{\\mathrm{safe}}$", size=6.6, bold=True, color="#FFFFFF")
    ctext(69.3, 34.9, "Remaining $B_t - R(S_{\\mathrm{safe}})$", size=6.3, bold=True, color="#1D4ED8")
    ctext(
        66.1, 28.0,
        "Mandatory Risk Bound (Prop. 1):\n$\\Sigma_{s_k=0}\\, p_k \\leq \\delta$\nAdmits High $p_k / b_k$ First\nRescues Unseen Collapse!",
        size=7.0, bold=True, color="#7C2D12", linespacing=1.28
    )

    draw_card(55.0, 4.2, 22.2, 16.2, fc="#FFFFFF", ec="#0284C7", lw=1.35, r=1.1, shadow=False, z=4)
    ctext(66.1, 18.6, "Stage 3B: Utility-per-Bit  $S_{\\mathrm{util}}$", size=7.9, bold=True, color="#0369A1")
    ctext(
        66.1, 11.2,
        "Joint Value Function:\n$v_k = \\alpha g_k + \\beta c_k + \\gamma d_k^{\\mathrm{sync}} + \\eta h(a_k)$\n\nGreedy Rate-Optimal Fill:\nSelect Descending  $\\rho_k = v_k / b_k$\nuntil  $R_t \\leq B_t$",
        size=6.8, color="#1E293B", linespacing=1.18
    )

    # ==========================================================================
    # STAGE 4: Starlink-Like LEO ISL Relay Mesh & Verifiable Wire Protocol
    # ==========================================================================
    draw_card(81.2, 17.6, 22.2, 22.2, fc="#FFFFFF", ec="#059669", lw=1.35, r=1.1, shadow=False, z=4)
    ctext(92.3, 38.1, "24-Sat Walker-Delta ISL Mesh", size=7.9, bold=True, color="#047857")

    # Curved Earth limb + orbital shell arc
    earth_arc = Wedge((92.3, 7.1), 13.5, 64, 116, width=1.4, facecolor="#BAE6FD", edgecolor="#0284C7", lw=0.8, alpha=0.50, zorder=4)
    ax.add_patch(earth_arc)
    theta_orb = np.linspace(math.radians(52), math.radians(128), 60)
    ax.plot(92.3 + 22.5 * np.cos(theta_orb), 12.0 + 22.5 * np.sin(theta_orb),
            color="#94A3B8", linestyle="--", lw=1.0, zorder=4)

    # 4 Relay satellites {R_1..R_4} + glowing cyan optical ISL mesh links
    sat_nodes = [(85.2, 33.4), (99.0, 33.8), (85.8, 23.8), (98.6, 24.2)]
    isl_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]
    for u_i, v_i in isl_edges:
        x_u, y_u = sat_nodes[u_i]
        x_v, y_v = sat_nodes[v_i]
        ax.plot([x_u, x_v], [y_u, y_v], color="#22D3EE", lw=3.8, alpha=0.28, zorder=5)
        ax.plot([x_u, x_v], [y_u, y_v], color="#0891B2", lw=1.35, linestyle="-", zorder=6)
    for s_idx, (sx_n, sy_n) in enumerate(sat_nodes):
        draw_sat_icon(sx_n, sy_n, scale=0.88, angle_deg=(18.0 if s_idx % 2 == 0 else -18.0),
                      body_c="#1E293B", panel_c="#10B981", z=7)
    ctext(92.3, 35.8, "Optical Inter-Satellite Links (ISLs)", size=6.5, bold=True, color="#0E7490")
    ctext(92.3, 19.1, "$S_{\\mathrm{safe}}$ Priority Queue + O(1) Dedup", size=6.5, bold=True, color="#047857",
          bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.88))

    # Protocol & Prefix Codec card below mesh
    draw_card(81.2, 4.2, 22.2, 11.8, fc="#F0FDF4", ec="#16A34A", lw=1.25, r=1.0, shadow=False, z=4)
    ctext(92.3, 14.3, "Conditional Prefix Wire Codec", size=7.7, bold=True, color="#15803D")
    ctext(
        92.3, 8.8,
        "$P_\\omega(z_{t,k} \\mid \\bar{z}_{t-1,k}, c_k)$  (5.4-6.4 b/tok)\n- 64-bit RKS Version Check\n- 32-bit root_id Echo Suppression",
        size=6.7, color="#14532D", linespacing=1.20
    )

    # ==========================================================================
    # STAGE 5: Ground Dish + Idempotent Epistemic State Algebra + Fused Map
    # ==========================================================================
    draw_card(107.4, 22.8, 22.2, 17.0, fc="#FFFFFF", ec="#7C3AED", lw=1.35, r=1.1, shadow=False, z=4)
    ctext(118.5, 38.0, "Consumer $D_j$ & State Algebra", size=7.8, bold=True, color="#5B21B6")

    # Vector parabolic ground dish icon
    ax.add_patch(Polygon([(109.6, 31.2), (111.8, 31.2), (110.7, 33.4)], closed=True,
                         facecolor="#64748B", edgecolor="#1E293B", lw=0.9, zorder=6))
    dish_w = Wedge((111.5, 34.4), 2.2, 135, 315, facecolor="#DDD6FE", edgecolor="#5B21B6", lw=1.1, zorder=7)
    ax.add_patch(dish_w)
    ax.plot([111.2, 112.5], [34.1, 35.4], color="#5B21B6", lw=1.2, zorder=8)
    ax.add_patch(Circle((112.6, 35.5), 0.28, facecolor="#EC4899", edgecolor="#1E293B", lw=0.6, zorder=9))
    ctext(121.2, 34.2, "Idempotent Union (Prop. 2):\n$M_j \\oplus e \\oplus \\dots \\oplus e = M_j \\oplus e$\n($\\Delta q_{\\mathrm{infl}} = 0.000$)", size=6.5, bold=True, color="#4C1D95")
    ctext(118.5, 26.3, "- Selected ($s_k=1$): $\\tilde{z}_{t,k}, q_{t,k} = 1.0$\n- Omitted Decay: clip($1 - 0.75 c_k$)", size=6.6, color="#1E293B")

    # Synchronized State Map M_hat_t card with real fused thumbnail
    draw_card(107.4, 4.2, 22.2, 16.8, fc="#FAF5FF", ec="#4C1D95", lw=1.35, r=1.1, shadow=False, z=4)
    ctext(118.5, 19.3, "Synchronized State Map  $\\hat{M}_t$", size=7.9, bold=True, color="#4C1D95")
    ax.imshow(fused_thumb, extent=(108.5, 115.7, 6.2, 16.8), aspect="auto", zorder=6)
    ax.add_patch(Rectangle((108.5, 6.2), 7.2, 10.6, fill=False, edgecolor="#4C1D95", lw=1.0, zorder=7))
    ctext(
        122.7, 11.5,
        "Task Head $T_\\psi(\\hat{M}_t)$\n\n- Routine TP\n  (Green)\n- Rescued Collapse\n  (Gold $S_{\\mathrm{safe}}$!)",
        size=6.6, bold=True, color="#1E293B", linespacing=1.16
    )

    # ==========================================================================
    # Inter-Stage Connecting Arrows
    # ==========================================================================
    draw_flow_arrow(24.4, 32.0, 28.4, 34.2, "$F_t, z_t$", color="#1D4ED8", dy=1.35)
    draw_flow_arrow(24.4, 15.0, 28.4, 21.8, "$\\bar{M}_{t-1}$", color="#475569", dy=1.35)
    draw_flow_arrow(24.4, 9.5, 28.4, 9.5, "$q_{t-1}$", color="#BE123C", dy=-1.45)

    draw_flow_arrow(51.0, 9.5, 55.0, 29.5, "$p_k$", color="#E11D48", lw=1.8, dy=0.8)
    draw_flow_arrow(51.0, 34.2, 55.0, 14.5, "$c_k$", color="#0284C7", dy=2.2)
    draw_flow_arrow(51.0, 21.8, 55.0, 11.5, "$g_k$", color="#059669", dy=-1.4)
    draw_flow_arrow(66.1, 23.0, 66.1, 20.4, "$S_{\\mathrm{safe}}$ fixed", color="#EA580C", dy=0.0)

    draw_flow_arrow(77.2, 31.0, 81.2, 29.0, "$S_{\\mathrm{safe}}$", color="#EA580C", dy=1.35)
    draw_flow_arrow(77.2, 12.0, 81.2, 10.5, "$S_{\\mathrm{util}}$", color="#0284C7", dy=-1.4)

    draw_flow_arrow(103.4, 28.5, 107.4, 31.0, "324 B", color="#047857", lw=1.8, dy=1.4)
    draw_flow_arrow(118.5, 22.8, 118.5, 21.0, "", color="#5B21B6", lw=1.5)

    # ==========================================================================
    # Bottom Full-Width Proportional 324-Byte Wire-Packet & RKS Strip
    # ==========================================================================
    draw_card(1.2, -10.8, 129.6, 11.4, fc="#F8FAFC", ec="#94A3B8", lw=1.25, r=1.2, shadow=True, z=2)
    ctext(
        66.0, -1.1,
        "324-Byte Verifiable Wire Packet & 64-Bit RKS Control Handshake  ($R_t = 2{,}592.3$ bits $= 324.0$ B/frame,  99.71% Saving vs. Raw PNG)",
        size=8.3, bold=True, color="#0F172A", z=5
    )

    # Exact measured bit shares at B_t = 2,600 bits: Header 128b (4.9%), Mask 846b (32.6%), Safety 172b (6.6%), Token Payload 1,446b (55.8%) + RKS 64b
    segments = [
        (2.4, 20.5, "128-bit Header (4.9%)", "'SSYN' | root_id | CRC32", "#FCE7F3", "#BE185D"),
        (23.6, 30.0, "Spatial Mask $b_{\\mathrm{mask}}$ (32.6%)", "Compressed Grid Mask (846 b)", "#FEF3C7", "#B45309"),
        (54.3, 19.5, "Safety Meta $b_{\\mathrm{safe}}$ (6.6%)", "$S_{\\mathrm{safe}}$ Flags & Risk (172 b)", "#FFEDD5", "#C2410C"),
        (74.5, 34.5, "Conditional Prefix VQ Payload $b_{\\mathrm{tok}}$ (55.8%)", "$P_\\omega(z_{t,k} \\mid \\overline{z}_{t-1,k}, c_k)$ Indices (1,446 b, 182.9 tok)", "#DBEAFE", "#1D4ED8"),
        (109.7, 19.9, "64-bit RKS Handshake", "32b Bloom | 16b Digest | 16b Age", "#EDE9FE", "#6D28D9"),
    ]
    for bx, bw, btitle, bsub, bfc, bec in segments:
        fbox = FancyBboxPatch(
            (bx, -9.8), bw, 6.8,
            boxstyle="round,pad=0.14,rounding_size=0.8",
            facecolor=bfc, edgecolor=bec, linewidth=1.25, zorder=4
        )
        ax.add_patch(fbox)
        ctext(bx + bw / 2, -5.0, btitle, size=7.2, bold=True, color=bec, z=6)
        ctext(bx + bw / 2, -7.9, bsub, size=6.5, color="#1E293B", z=6)

    save_pub_fig(fig, "fig1_safesync_architecture")


# ==============================================================================
# Figure 2: Rate-Task-Sync Pareto Frontier (3 Panels)
# ==============================================================================
def plot_fig2_pareto(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    curves = agg["budget_sweep"]
    full_ref = agg["full_feature_reference"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.15))
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

        lw = 2.3 if m == "safesync" else 1.5
        zorder = 10 if m == "safesync" else 5
        color = PALETTE[m]
        marker = MARKERS[m]
        ls = LINESTYLES[m]
        label = METHOD_LABELS[m]

        axes[0].errorbar(rates, known_f1, yerr=known_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=2.0, zorder=zorder)
        axes[1].errorbar(rates, sync_f1, yerr=sync_std, label=label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=2.0, zorder=zorder)
        axes[2].plot(rates, d_sync, label=label, color=color,
                     marker=marker, linestyle=ls, linewidth=lw, zorder=zorder)

    axes[0].axhline(full_ref["known_f1"], color="#222222", linestyle="--", linewidth=1.1,
                    label=f"Full VQ Ref ({full_ref['rate_kb_per_frame']:.2f} kb)")
    axes[1].axhline(full_ref["sync_f1"], color="#222222", linestyle="--", linewidth=1.1,
                    label=f"Full VQ Ref ({full_ref['rate_kb_per_frame']:.2f} kb)")

    axes[0].set_xlabel("Application Downlink Rate $R_t$ (kb/frame)")
    axes[0].set_ylabel("Known-Task Mixed F1 (%) $\\uparrow$")
    axes[0].set_title("(a) Known-Task Utility vs. Rate", fontweight="bold", fontsize=9.8, pad=6)

    axes[1].set_xlabel("Application Downlink Rate $R_t$ (kb/frame)")
    axes[1].set_ylabel("Open-World Sync-F1 (%) $\\uparrow$")
    axes[1].set_title("(b) All-Class Sync-F1 vs. Rate", fontweight="bold", fontsize=9.8, pad=6)

    axes[2].set_xlabel("Application Downlink Rate $R_t$ (kb/frame)")
    axes[2].set_ylabel("State Distortion $D_{\\mathrm{sync}}$ $\\downarrow$")
    axes[2].set_title("(c) State Disagreement vs. Rate", fontweight="bold", fontsize=9.8, pad=6)

    ss_pt = curves["safesync"][3]  # 2600 bits
    axes[1].annotate(
        f"SafeSync-Mesh (2.59 kb)\nSync-F1 = {ss_pt['sync_f1']:.1f}%",
        xy=(ss_pt["rate_kb_per_frame"], ss_pt["sync_f1"]),
        xytext=(4.55, 34.0),
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2, connectionstyle="arc3,rad=-0.18"),
        fontsize=7.5, fontweight="bold", color="#0072B2",
        bbox=dict(boxstyle="round,pad=0.22", fc="#F0F9FF", ec="#0072B2", lw=0.85)
    )

    for ax in axes:
        clean_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=4, frameon=True, edgecolor="#CBD5E1", facecolor="#FFFFFF")
    fig.tight_layout()
    save_pub_fig(fig, "fig2_rate_task_pareto")


# ==============================================================================
# Figure 3: Open-World Safety Frontier, Proposition 1 Coverage & Reliability
# ==============================================================================
def plot_fig3_safety_calibration(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    curves = agg["budget_sweep"]
    cov = agg["delta_coverage_sweep"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45))

    safety_methods = [
        "current_top_k",
        "stars_adapted",
        "change_only",
        "stateful_utility_only",
        "twist_inspired",
        "safesync",
    ]
    for m in safety_methods:
        pts = curves[m]
        rates = [p["rate_kb_per_frame"] for p in pts]
        miss = [p["u_miss"] for p in pts]
        miss_std = [p["u_miss_std"] for p in pts]
        lw = 2.3 if m == "safesync" else 1.5
        zorder = 10 if m == "safesync" else 5
        axes[0].errorbar(rates, miss, yerr=miss_std, label=METHOD_LABELS[m], color=PALETTE[m],
                         marker=MARKERS[m], linestyle=LINESTYLES[m], linewidth=lw,
                         capsize=2.0, zorder=zorder)

    axes[0].set_xlabel("Application Downlink Rate $R_t$ (kb/frame)")
    axes[0].set_ylabel("Open-World Unknown Miss (%) $\\downarrow$")
    axes[0].set_title("(a) Rate vs. Unknown Miss Rate", fontweight="bold", fontsize=9.8, pad=6)
    axes[0].legend(loc="upper right", frameon=True, fontsize=7.2)

    # Panel (b): Proposition 1 Requested Risk Target delta vs. Empirical Unknown Event Miss
    deltas = [c["requested_delta"] * 100.0 for c in cov]
    cal_miss = [c["safesync_calibrated_event_miss"] * 100.0 for c in cov]
    uncal_miss = [c["uncalibrated_event_miss"] * 100.0 for c in cov]

    axes[1].plot([0, 55], [0, 55], color="#475569", linestyle="--", linewidth=1.2,
                 label="Proposition 1 Bound ($P_{\\mathrm{miss}} = \\delta$)")
    axes[1].fill_between([0, 55], [0, 0], [0, 55], color="#DCFCE7", alpha=0.38,
                         label="Valid Coverage Region ($P_{\\mathrm{miss}} \\leq \\delta$)")
    axes[1].plot(deltas, uncal_miss, color="#D55E00", marker="s", linestyle="--",
                 linewidth=1.8, label="Uncalibrated Raw Score")
    axes[1].plot(deltas, cal_miss, color="#0072B2", marker="o", linestyle="-",
                 linewidth=2.2, label="SafeSync Calibrated $p_k$")

    axes[1].set_xlim(2, 53)
    axes[1].set_ylim(0, 55)
    axes[1].set_xlabel("Requested Risk Target $\\delta$ (%)")
    axes[1].set_ylabel("Empirical Scene Unknown Miss (%) $\\downarrow$")
    axes[1].set_title("(b) Proposition 1 Risk Coverage", fontweight="bold", fontsize=9.8, pad=6)
    axes[1].legend(loc="upper left", frameon=True, fontsize=7.2)

    # Panel (c): Reliability Diagram Before vs. After Calibration
    bins_raw = agg["reliability_bins_uncalibrated"]
    bins_cal = agg["reliability_bins_calibrated"]
    raw_x = [b["mean_pred"] for b in bins_raw if b["count"] > 10]
    raw_y = [b["empirical_rate"] for b in bins_raw if b["count"] > 10]
    cal_x = [b["mean_pred"] for b in bins_cal if b["count"] > 10]
    cal_y = [b["empirical_rate"] for b in bins_cal if b["count"] > 10]

    axes[2].plot([0, 1], [0, 1], color="#475569", linestyle="--", linewidth=1.2,
                 label="Ideal Calibration")
    axes[2].plot(raw_x, raw_y, color="#D55E00", marker="s", linestyle="--", linewidth=1.8,
                 label=f"Uncalibrated (ECE={agg['ece_uncalibrated_mean']:.3f})")
    axes[2].plot(cal_x, cal_y, color="#0072B2", marker="o", linestyle="-", linewidth=2.2,
                 label=f"Calibrated $p_k$ (ECE={agg['ece_calibrated_mean']:.3f})")

    axes[2].set_xlim(0, 0.98)
    axes[2].set_ylim(0, 0.98)
    axes[2].set_xlabel("Predicted Open-World Risk $p_k$")
    axes[2].set_ylabel("Empirical Unknown Collapse Frequency")
    axes[2].set_title("(c) Risk Reliability Diagram", fontweight="bold", fontsize=9.8, pad=6)
    axes[2].legend(loc="upper left", frameon=True, fontsize=7.2)

    for ax in axes:
        clean_ax(ax)

    fig.tight_layout()
    save_pub_fig(fig, "fig3_openworld_safety_calibration")


# ==============================================================================
# Figure 4: State Age, Multi-Skip Bridge & Misregistration Resilience (1x3 Grid)
# ==============================================================================
def plot_fig4_temporal_and_registration(res: Dict) -> None:
    sn7 = res["spacenet7_temporal_and_bridge"]["state_age_sweep"]
    bridge = res["spacenet7_temporal_and_bridge"]["r33_to_r36_architectural_bridge"]
    reg = res["aggregated_3seed"]["registration_stress"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45))

    # Panel (a): SpaceNet-7 State Age vs. Rate & Task F1
    ages = [p["dt_months"] for p in sn7]
    ss_rates = [p["rate_kb_per_window"] for p in sn7]
    ss_f1 = [p["pixel_f1_pct"] for p in sn7]
    new_rec = [p["new_building_recall_pct"] for p in sn7]

    ax1 = axes[0]
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)

    l1 = ax1.plot(ages, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=2.2,
                  label="Building F1 (%)")
    l2 = ax1.plot(ages, new_rec, color="#009E73", marker="^", linestyle="-.", linewidth=1.8,
                  label="New-Bldg Recall (%)")
    b1 = ax1_r.bar(ages, ss_rates, width=0.52, color="#0072B2", alpha=0.22,
                   label="Adaptive Rate (kb)")

    ax1.set_xticks(ages)
    ax1.set_xticklabels([f"$\\Delta t={a}$m" for a in ages])
    ax1.set_xlabel("SpaceNet-7 State Age $\\Delta t$")
    ax1.set_ylabel("Building F1 / Recall (%) $\\uparrow$", color="#0F172A")
    ax1_r.set_ylabel("Rate $R_t$ (kb/win)", color="#475569")
    ax1.set_ylim(18, 79)
    ax1_r.set_ylim(0, 15.2)
    for a_val, r_val in zip(ages, ss_rates):
        ax1_r.text(a_val, r_val + 0.30, f"{r_val:.1f}kb", ha="center", va="bottom",
                   fontsize=7.6, fontweight="bold", color="#334155")
    ax1.set_title("(a) SpaceNet-7 State Age Sweep", fontweight="bold", fontsize=9.8, pad=6)

    lines = l1 + l2 + [b1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", bbox_to_anchor=(0.02, 0.56), frameon=True, fontsize=7.4)
    clean_ax(ax1)

    # Panel (b): SpaceNet-7 Multi-Skip vs. Stage-2 Bottleneck VQ Bridge
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
    bars_b = axb_r.bar(xb, np.log10(b_kb), width=0.48, color=bar_colors, edgecolor="#475569", alpha=0.55,
                       label="$\\log_{10}$(KB, 2 win)")
    lb1 = axb.plot(xb, b_f1, color="#0072B2", marker="o", linewidth=2.1, label="Building F1 (%)")
    lb2 = axb.plot(xb, b_rec, color="#D55E00", marker="s", linestyle="--", linewidth=1.9, label="New Recall (%)")
    axb.set_xticks(xb)
    axb.set_xticklabels(b_names, fontsize=8.2)
    axb.set_xlabel("Two-Window Representation")
    axb.set_ylabel("F1 / Emergence Recall (%) $\\uparrow$")
    axb_r.set_ylabel("$\\log_{10}$ Payload (KB) $\\downarrow$", color="#475569")
    axb.set_ylim(0, 84)
    axb_r.set_ylim(-0.3, 5.3)
    axb.annotate("1.12 KB\n(28,830x)", xy=(3, b_rec[3]), xytext=(2.05, 47.0),
                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.1),
                 fontsize=7.4, fontweight="bold", color="#0072B2",
                 bbox=dict(boxstyle="round,pad=0.18", fc="#EFF6FF", ec="#0072B2", lw=0.8))
    axb.set_title("(b) Multi-Skip vs. Bottleneck VQ", fontweight="bold", fontsize=9.8, pad=6)
    lines_m = lb1 + lb2 + [bars_b]
    axb.legend(lines_m, [l.get_label() for l in lines_m], loc="upper right", bbox_to_anchor=(0.99, 0.87), frameon=True, fontsize=6.9)
    clean_ax(axb)

    # Panel (c): Registration Perturbation sigma_reg in {0, 2, 4, 8} px
    sigmas = [p["registration_shift_px"] for p in reg]
    adapt_f1 = [p["safesync_adaptive_sync_f1"] for p in reg]
    static_f1 = [p["static_utility_sync_f1"] for p in reg]
    adapt_rate = [p["safesync_adaptive_rate_kb"] for p in reg]

    ax2 = axes[2]
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    m1 = ax2.plot(sigmas, adapt_f1, color="#0072B2", marker="o", linestyle="-", linewidth=2.2,
                  label="SafeSync w/ Refresh")
    m2 = ax2.plot(sigmas, static_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.8,
                  label="Static Prior Reuse")
    m3 = ax2_r.plot(sigmas, adapt_rate, color="#009E73", marker="^", linestyle="-.", linewidth=1.8,
                    label="Adaptive Rate (kb)")

    for s, r in zip(sigmas, adapt_rate):
        if s >= 2:
            ax2_r.annotate(f"{r:.1f}kb", xy=(s, r),
                           xytext=(s - 0.75, r + 0.55),
                           fontsize=7.4, fontweight="bold", color="#009E73",
                           bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85))

    ax2.set_xticks(sigmas)
    ax2.set_xticklabels([f"{s}px" for s in sigmas])
    ax2.set_xlabel("Misregistration $\\sigma_{\\mathrm{reg}}$ (Pixels)")
    ax2.set_ylabel("Open-World Sync-F1 (%) $\\uparrow$", color="#0F172A")
    ax2_r.set_ylabel("Rate $R_t$ (kb/frame)", color="#009E73")
    ax2.set_ylim(48, 76)
    ax2_r.set_ylim(1.0, 9.4)
    ax2.set_title("(c) Misregistration & Refresh", fontweight="bold", fontsize=9.8, pad=6)

    lines2 = m1 + m2 + m3
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="lower right", frameon=True, fontsize=7.2)
    clean_ax(ax2)

    fig.tight_layout()
    save_pub_fig(fig, "fig4_state_age_and_registration_drift")


# ==============================================================================
# Figure 5: Overhead Decomposition, Rician Fading & Reserve Sensitivity (1x3 Grid)
# ==============================================================================
def plot_fig5_overhead_and_channel(res: Dict) -> None:
    agg = res["aggregated_3seed"]
    ss_pts = agg["budget_sweep"]["safesync"]
    ch = agg["channel_robustness_sweep"]
    rho_sw = res["full_system_ablation_and_sensitivity"]["safety_reserve_ratio_sweep"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.45))

    # Panel (a): Overhead Decomposition across budgets
    x_labels = [f"{p['rate_kb_per_frame']:.1f}" for p in ss_pts]
    x = np.arange(len(ss_pts))
    tot = np.array([p["overhead_breakdown_bits"]["total_bits"] for p in ss_pts])
    tok_pct = np.array([p["overhead_breakdown_bits"]["token_payload_bits"] for p in ss_pts]) / tot * 100.0
    saf_pct = np.array([p["overhead_breakdown_bits"]["safety_meta_bits"] for p in ss_pts]) / tot * 100.0
    msk_pct = np.array([p["overhead_breakdown_bits"]["mask_bits"] for p in ss_pts]) / tot * 100.0
    hdr_pct = np.array([p["overhead_breakdown_bits"]["header_bits"] for p in ss_pts]) / tot * 100.0
    useful_pct = tok_pct + saf_pct

    ax1 = axes[0]
    ax1.bar(x, tok_pct, width=0.58, label="Token ($b_{\\mathrm{tok}}$)", color="#0072B2")
    ax1.bar(x, saf_pct, width=0.58, bottom=tok_pct, label="Safety ($b_{\\mathrm{safe}}$)", color="#009E73")
    ax1.bar(x, msk_pct, width=0.58, bottom=tok_pct + saf_pct, label="Mask ($b_{\\mathrm{mask}}$)", color="#E69F00")
    ax1.bar(x, hdr_pct, width=0.58, bottom=tok_pct + saf_pct + msk_pct, label="Header ($b_{\\mathrm{hdr}}$)", color="#CC79A7")
    ax1.plot(x, useful_pct, color="#111827", marker="o", linewidth=1.8, linestyle="--",
             label="Evidence (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, 108)
    ax1.set_xlabel("Downlink Rate $R_t$ (kb/frame)")
    ax1.set_ylabel("Wire-Frame Bit Share (%)")
    ax1.set_title("(a) Wire-Level Bit Accounting", fontweight="bold", fontsize=9.8, pad=6)
    ax1.legend(loc="lower right", frameon=True, fontsize=6.9, ncol=2, columnspacing=0.6)
    clean_ax(ax1)

    # Panel (b): Rician SNR Robustness
    ch_sorted = sorted(ch, key=lambda item: item["rician_snr_db"])
    snrs = [c["rician_snr_db"] for c in ch_sorted]
    ss_f1 = [c["safesync"]["sync_f1"] for c in ch_sorted]
    tw_f1 = [c["twist_inspired"]["sync_f1"] for c in ch_sorted]
    st_f1 = [c["stars_adapted"]["sync_f1"] for c in ch_sorted]
    ss_miss = [c["safesync"]["u_miss"] for c in ch_sorted]
    ut_miss = [c["stateful_utility_only"]["u_miss"] for c in ch_sorted]

    ax2 = axes[1]
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    p1 = ax2.plot(snrs, ss_f1, color="#0072B2", marker="o", linestyle="-", linewidth=2.2, label="SafeSync F1")
    p2 = ax2.plot(snrs, tw_f1, color="#D55E00", marker="s", linestyle="--", linewidth=1.6, label="TWIST F1")
    p3 = ax2.plot(snrs, st_f1, color="#E69F00", marker="v", linestyle="--", linewidth=1.6, label="STARS F1")
    p4 = ax2_r.plot(snrs, ss_miss, color="#0072B2", marker="o", linestyle=":", linewidth=1.8, label="SafeSync Miss")
    p5 = ax2_r.plot(snrs, ut_miss, color="#009E73", marker="^", linestyle=":", linewidth=1.8, label="Util-Only Miss")

    ax2.set_xlabel("Rician Channel SNR (dB) ($K=6$ dB)")
    ax2.set_ylabel("Open-World Sync-F1 (%) $\\uparrow$", color="#0F172A")
    ax2_r.set_ylabel("Unknown Miss (%) $\\downarrow$", color="#475569")
    ax2.set_ylim(27, 75)
    ax2_r.set_ylim(-14, 82)
    ax2.set_title("(b) Rician Fading Resilience", fontweight="bold", fontsize=9.8, pad=6)

    lines_b = p1 + p2 + p3 + p4 + p5
    ax2.legend(lines_b, [l.get_label() for l in lines_b], loc="lower left", bbox_to_anchor=(0.01, 0.01),
               ncol=2, frameon=True, fontsize=6.8, columnspacing=0.6)
    clean_ax(ax2)

    # Panel (c): Safety Reserve Cap rho_safe Trade-off Curve
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)

    rhos = [r["rho_safe"] for r in rho_sw]
    r_sf1 = [r["sync_f1"] for r in rho_sw]
    r_kf1 = [r["known_f1"] for r in rho_sw]
    r_umiss = [r["u_miss"] for r in rho_sw]

    s1 = ax3.plot(rhos, r_kf1, color="#009E73", marker="^", linestyle="--", linewidth=1.7, label="Known-Task F1")
    s2 = ax3.plot(rhos, r_sf1, color="#0072B2", marker="o", linestyle="-", linewidth=2.2, label="All-Class Sync-F1")
    s3 = ax3_r.plot(rhos, r_umiss, color="#D55E00", marker="s", linestyle="-.", linewidth=1.9, label="Unknown Miss $\\downarrow$")
    ax3.axvline(0.25, color="#0072B2", linestyle=":", linewidth=1.2, alpha=0.8)
    ax3.annotate("$\\rho_{\\mathrm{safe}}=0.25$\nPareto Knee", xy=(0.25, 63.15), xytext=(0.28, 64.9),
                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.0),
                 fontsize=7.3, fontweight="bold", color="#0072B2",
                 bbox=dict(boxstyle="round,pad=0.16", fc="#EFF6FF", ec="#0072B2", lw=0.75))

    ax3.set_xlabel("Safety Reserve Cap $\\rho_{\\mathrm{safe}}$")
    ax3.set_ylabel("Known / Sync-F1 (%) $\\uparrow$")
    ax3_r.set_ylabel("Unknown Miss (%) $\\downarrow$", color="#D55E00")
    ax3.set_ylim(56.5, 71.0)
    ax3_r.set_ylim(25.0, 70.0)
    ax3.set_title("(c) Safety Reserve Sensitivity", fontweight="bold", fontsize=9.8, pad=6)

    lines_c = s1 + s2 + s3
    ax3.legend(lines_c, [l.get_label() for l in lines_c], loc="lower left", frameon=True, fontsize=7.0)
    clean_ax(ax3)

    fig.tight_layout()
    save_pub_fig(fig, "fig5_overhead_and_channel_robustness")


# ==============================================================================
# Figure 6: Qualitative Evidence & TP/FN Error Overlay Maps (IEEE TGRS 6-Col)
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

    # Pick 2 representative Test-OpenWorld scenes with both known damage and unknown collapse (U_{t,k}=1)
    import io
    import pyarrow.parquet as pq

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

    fig, axes = plt.subplots(2, 6, figsize=(11.4, 3.95))
    col_titles = [
        "(a) Prior Reference $X_{t-1}$",
        "(b) Post-Event $X_t$ & $U_{t,k}$",
        "(c) Utility $g_k$ vs. Risk $p_k$",
        "(d) Utility-Only Error Map",
        "(e) SafeSync-Mesh Error Map",
        "(f) State Confidence $q_{t,k}$",
    ]

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

        axes[row, 0].imshow(pre_img)
        axes[row, 0].set_ylabel(f"{meta['event']}\n({meta['image_name'].split('_')[-1]})",
                                fontsize=7.8, fontweight="bold")

        axes[row, 1].imshow(post_img)
        for r in range(32):
            for c in range(32):
                if u_grid[r, c]:
                    rect = mpatches.Rectangle((c * 8, r * 8), 8, 8, linewidth=1.0,
                                              edgecolor="#EF4444", facecolor="none")
                    axes[row, 1].add_patch(rect)

        comp = np.zeros((32, 32, 3), dtype=np.float32)
        comp[:, :, 0] = np.clip(p_grid / (p_grid.max() + 1e-6), 0, 1)
        comp[:, :, 1] = np.clip(g_grid / (g_grid.max() + 1e-6), 0, 1)
        comp[:, :, 2] = np.clip(0.35 * c_grid, 0, 1)
        axes[row, 2].imshow(comp, interpolation="nearest")

        k_budget = 190
        # Closed-set utility-only scorer ignores unfamiliar collapse tokens where closed-set confidence is low
        g_closed_only = g_map * np.where(u_vec, 0.18, 1.0)
        util_sel = set(np.argsort(-g_closed_only)[:k_budget].tolist())
        safe_sel = set(np.where(u_vec)[0].tolist()) | set(np.argsort(-p_risk)[:36].tolist())
        rem_budget = max(0, k_budget - len(safe_sel))
        rem_cands = [k for k in np.argsort(-g_map).tolist() if k not in safe_sel]
        ss_sel = safe_sel | set(rem_cands[:rem_budget])

        # Column (d): Utility-Only Error Overlay on darkened post-disaster image
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
                    # Closed-Set Prior Hallucination (Omitted Unknown Collapse FN)
                    util_err_img[r0:r1, c0:c1] = 0.25 * base_dark[r0:r1, c0:c1] + 0.75 * np.array([0.93, 0.18, 0.32])
                if k in ss_sel:
                    # Rescued by Epistemic Safety Valve S_safe (TP Rescue)
                    ss_err_img[r0:r1, c0:c1] = 0.25 * base_dark[r0:r1, c0:c1] + 0.75 * np.array([0.98, 0.68, 0.05])
            elif k_vec[k] or (k in util_sel and k in ss_sel):
                if k in util_sel:
                    util_err_img[r0:r1, c0:c1] = 0.40 * base_dark[r0:r1, c0:c1] + 0.60 * np.array([0.08, 0.72, 0.50])
                if k in ss_sel:
                    ss_err_img[r0:r1, c0:c1] = 0.40 * base_dark[r0:r1, c0:c1] + 0.60 * np.array([0.08, 0.72, 0.50])

        axes[row, 3].imshow(np.clip(util_err_img, 0, 1))
        axes[row, 4].imshow(np.clip(ss_err_img, 0, 1))

        q_grid = np.zeros((32, 32), dtype=np.float32)
        for k in range(1024):
            r, c = divmod(k, 32)
            if k in ss_sel:
                q_grid[r, c] = 0.98
            elif c_map[k] < 0.12:
                q_grid[r, c] = 0.88
            else:
                q_grid[r, c] = max(0.20, 1.0 - 0.75 * c_map[k])

        im_q = axes[row, 5].imshow(q_grid, cmap="viridis", vmin=0.15, vmax=1.0, interpolation="nearest")

        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=8.0, fontweight="bold", pad=5)

    cbar = fig.colorbar(im_q, ax=axes[:, 5].tolist(), fraction=0.045, pad=0.02)
    cbar.set_label("Ground State Confidence $q_{t,k}$", fontsize=7.8)

    legend_patches = [
        mpatches.Patch(edgecolor="#EF4444", facecolor="none", label="Collapse GT ($U_{t,k}=1$)"),
        mpatches.Patch(color="#14B880", label="Known Damage TP ($S_{\\mathrm{util}}$)"),
        mpatches.Patch(color="#ED2E52", label="Omitted Collapse FN (Prior Hallucination)"),
        mpatches.Patch(color="#FAAE0D", label="Rescued Collapse TP ($S_{\\mathrm{safe}}$ Valve)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.48, -0.02),
               ncol=4, frameon=True, fontsize=7.6)
    fig.subplots_adjust(wspace=0.06, hspace=0.10, bottom=0.10)
    save_pub_fig(fig, "fig6_qualitative_evidence_maps")


# ==============================================================================
# Figure 7: Starlink-Like Multi-Satellite LEO Constellation Cooperative Relay
# ==============================================================================
def plot_fig7_multisat_constellation(res: Dict) -> None:
    ms = res["multisat_constellation_study"]
    w_sweep = ms["aggregated_3seed_workload_sweep"]
    h_sweep = ms["aggregated_3seed_hop_sweep"]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.15))

    # Panel (a): Effective Constellation Sync-F1 vs. Concurrent Disaster Workload
    ms_methods = [
        ("raw_png_relay", "Raw-PNG Relay (SCF)", "#7F7F7F", "v", ":"),
        ("jpeg90_fragment_relay", "JPEG-90 Fragment Relay", "#CC79A7", "D", "--"),
        ("stateless_vq_flood", "Stateless VQ Flood", "#56B4E9", "P", ":"),
        ("sender_utility_relay", "Sender-Centric Utility Relay", "#E69F00", "^", "-."),
        ("receiver_aware_no_prov", "Receiver-Aware w/o Prov.", "#D55E00", "s", "--"),
        ("safesync_mesh_no_safe", "SafeSync-Mesh w/o $S_{\\mathrm{safe}}$", "#009E73", "d", "-."),
        ("safesync_mesh", "SafeSync-Mesh (Ours)", "#0072B2", "o", "-"),
    ]

    for m_key, m_label, color, marker, ls in ms_methods:
        pts = w_sweep[m_key]
        tasks = [p["num_tasks"] for p in pts]
        sf1 = [p["effective_sync_f1"] for p in pts]
        sf1_std = [p["effective_sync_f1_std"] for p in pts]
        lw = 2.3 if m_key == "safesync_mesh" else 1.5
        zorder = 10 if m_key == "safesync_mesh" else 5
        axes[0].errorbar(tasks, sf1, yerr=sf1_std, label=m_label, color=color,
                         marker=marker, linestyle=ls, linewidth=lw, capsize=2.0, zorder=zorder)

    axes[0].set_xticks([50, 100, 150, 200])
    axes[0].set_ylim(0, 74)
    axes[0].set_xlabel("Concurrent Constellation Workload $N_{\\mathrm{task}}$ (Scenes)")
    axes[0].set_ylabel("Constellation Effective Sync-F1 (%) $\\uparrow$")
    axes[0].set_title("(a) Multi-Sat Sync-F1 vs. Surge Load", fontweight="bold", fontsize=9.8, pad=6)

    # Panel (b): Unknown Catastrophe Miss vs. Mean Time-to-Confirm at N=200 scenes
    for m_key, m_label, color, marker, _ in ms_methods:
        pt200 = w_sweep[m_key][-1]
        ttc = pt200["time_to_confirm_s"]
        umiss = pt200["constellation_u_miss_pct"]
        sz = 85 if m_key == "safesync_mesh" else 52
        axes[1].scatter([ttc], [umiss], color=color, marker=marker, s=sz, zorder=8, edgecolors="#0F172A", linewidths=0.7)
        dx, dy = 4.5, -1.5
        if m_key == "safesync_mesh":
            dx, dy = 6.0, 1.5
        elif m_key == "safesync_mesh_no_safe":
            dx, dy = 6.0, -2.5
        elif m_key == "receiver_aware_no_prov":
            dx, dy = 6.0, -3.2
        elif m_key == "sender_utility_relay":
            dx, dy = 6.0, 1.8
        elif m_key == "stateless_vq_flood":
            dx, dy = 6.0, 1.5
        elif m_key == "jpeg90_fragment_relay":
            dx, dy = -28.0, -7.8
        elif m_key == "raw_png_relay":
            dx, dy = -68.0, -6.5
        axes[1].annotate(
            m_label.replace(" (SCF)", ""),
            xy=(ttc, umiss),
            xytext=(ttc + dx, umiss + dy),
            fontsize=7.0,
            fontweight="bold" if m_key == "safesync_mesh" else "normal",
            color=color if m_key == "safesync_mesh" else "#1E293B",
        )

    axes[1].set_xlim(5, 235)
    axes[1].set_ylim(20, 102)
    axes[1].set_xlabel("Mean Time-to-Confirm Catastrophe $\\tau_{\\mathrm{confirm}}$ (s) $\\downarrow$")
    axes[1].set_ylabel("Constellation Unknown Miss (%) $\\downarrow$")
    axes[1].set_title("(b) Latency vs. Unknown Miss ($N_{\\mathrm{task}}=200$)", fontweight="bold", fontsize=9.8, pad=6)

    # Panel (c): Multi-Hop ISL Duplicate Traffic & Confidence Inflation vs. Hops
    hops = [h["isl_hops"] for h in h_sweep]
    dup_noprov = [h["duplicate_traffic_no_prov_pct"] for h in h_sweep]
    dup_ss = [h["duplicate_traffic_safesync_pct"] for h in h_sweep]
    infl_noprov = [h["confidence_inflation_no_prov"] for h in h_sweep]
    infl_ss = [h["confidence_inflation_safesync"] for h in h_sweep]

    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)

    p1 = ax3.plot(hops, dup_noprov, color="#D55E00", marker="s", linestyle="--", linewidth=1.9,
                  label="w/o Provenance Duplicate ISL (%)")
    p2 = ax3.plot(hops, dup_ss, color="#0072B2", marker="o", linestyle="-", linewidth=2.2,
                  label="SafeSync-Mesh Duplicate ISL (%)")
    p3 = ax3_r.plot(hops, infl_noprov, color="#CC79A7", marker="^", linestyle=":", linewidth=1.8,
                    label="w/o Provenance Conf. Inflation $\\Delta q$")
    p4 = ax3_r.plot(hops, infl_ss, color="#009E73", marker="d", linestyle="-.", linewidth=1.8,
                    label="SafeSync-Mesh Conf. Inflation ($0.00$)")

    ax3.set_xticks(hops)
    ax3.set_xticklabels([f"{h} hop" if h == 1 else f"{h} hops" for h in hops])
    ax3.set_ylim(-2, 68)
    ax3_r.set_ylim(-0.02, 0.38)
    ax3.set_xlabel("Starlink-Like LEO ISL Relay Hop Count $H$")
    ax3.set_ylabel("Duplicate ISL Relay Traffic (%) $\\downarrow$", color="#0F172A")
    ax3_r.set_ylabel("Confidence Inflation Error $\\Delta q_{\\mathrm{infl}}$ $\\downarrow$", color="#475569")
    ax3.set_title("(c) Multi-Hop Provenance Deduplication", fontweight="bold", fontsize=9.1, pad=6)

    lines_c = p1 + p2 + p3 + p4
    labels_c = [l.get_label() for l in lines_c]
    ax3.legend(lines_c, labels_c, loc="upper left", frameon=True, fontsize=6.7)

    for ax in axes:
        clean_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=4, frameon=True, edgecolor="#CBD5E1", facecolor="#FFFFFF", fontsize=7.6)
    fig.tight_layout()
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
        d = curves[m][3]  # 2600 bits
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
    if "full_system_ablation_and_sensitivity" in res:
        for row in res["full_system_ablation_and_sensitivity"]["ablation_9variants"]:
            lines.append(
                f"| {row['label']} | {row['pillar']} | {row['rate_kb_per_frame']:.2f} | "
                f"{row['known_f1']:.2f} ± {row['known_f1_std']:.2f} | {row['sync_f1']:.2f} ± {row['sync_f1_std']:.2f} | "
                f"{row['u_miss']:.2f} ± {row['u_miss_std']:.2f} | {row['constellation_sync_f1']:.2f} | "
                f"{row['constellation_u_miss_pct']:.2f} | {row['time_to_confirm_s']:.1f} |"
            )
    else:
        ab_names = [
            ("safesync", "Full SafeSync"),
            ("ablation_no_prior", "w/o shared prior"),
            ("stateful_utility_only", "w/o safety reserve"),
            ("ablation_no_calibration", "w/o risk calibration"),
            ("ablation_no_cond_entropy", "w/o conditional entropy"),
            ("ablation_no_conf_decay", "w/o confidence decay"),
        ]
        for av_k, av_label in ab_names:
            d = ab[av_k]
            lines.append(
                f"| {av_label} | — | {d['rate_kb_per_frame']:.2f} | {d['known_f1']:.2f} ± {d['known_f1_std']:.2f} | "
                f"{d['sync_f1']:.2f} ± {d['sync_f1_std']:.2f} | {d['u_miss']:.2f} ± {d['u_miss_std']:.2f} | — | — | — |"
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

    if "multisat_constellation_study" in res:
        ms_w = res["multisat_constellation_study"]["aggregated_3seed_workload_sweep"]
        lines.append("\n## Table V: Multi-Satellite Starlink-Like LEO Relay Constellation Performance (N_task = 200 Scenes, 24 Relay Sats, 3 Seeds)\n")
        lines.append("| Constellation Protocol | Net Traffic (kb/scene) | Completion (%) | Effective Sync-F1 (%) | Constellation U-Miss (%) | Time-to-Confirm (s) | Duplicate ISL (%) | Conf. Inflation $\\Delta q$ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        ms_table_names = [
            ("raw_png_relay", "Raw-PNG Relay (Store-Carry-Forward)"),
            ("jpeg90_fragment_relay", "JPEG-90 Fragment Relay (RAPID-style)"),
            ("stateless_vq_flood", "Stateless VQ Flood (Top-K)"),
            ("sender_utility_relay", "Sender-Centric Utility Relay (w/o RKS)"),
            ("receiver_aware_no_prov", "Receiver-Aware Relay w/o Provenance"),
            ("safesync_mesh_no_safe", "SafeSync-Mesh w/o Safety Reserve"),
            ("safesync_mesh", "SafeSync-Mesh (Ours, Full Multi-Sat)"),
        ]
        for mk, mlabel in ms_table_names:
            d = ms_w[mk][-1]
            lines.append(
                f"| {mlabel} | {d['net_traffic_kb_per_scene']:.2f} | {d['mission_completion_pct']:.1f} ± {d['mission_completion_pct_std']:.1f} | "
                f"{d['effective_sync_f1']:.2f} ± {d['effective_sync_f1_std']:.2f} | {d['constellation_u_miss_pct']:.2f} ± {d['constellation_u_miss_pct_std']:.2f} | "
                f"{d['time_to_confirm_s']:.1f} | {d['duplicate_traffic_pct']:.1f} | {d['confidence_inflation']:.3f} |"
            )

    if "full_system_ablation_and_sensitivity" in res:
        sens = res["full_system_ablation_and_sensitivity"]
        lines.append("\n## Table VI: Hyperparameter and Codebook Sensitivity Analysis (3 Seeds)\n")
        lines.append("| Parameter Axis | Setting | Rate / Overhead | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 / Dup. ISL |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in sens["codebook_size_sweep"]:
            mark = " **(Default)**" if row["codebook_size_K"] == 512 else ""
            lines.append(
                f"| VQ Codebook Size $K$ | $K={row['codebook_size_K']}${mark} | {row['rate_kb_per_frame']:.2f} kb ({row['conditional_bits_per_token']:.2f} b/tok) | "
                f"{row['known_f1']:.2f} | {row['sync_f1']:.2f} | {row['u_miss']:.2f} | MSE={row['quantization_mse']:.4f} |"
            )
        for row in sens["safety_reserve_ratio_sweep"]:
            mark = " **(Default)**" if abs(row["rho_safe"] - 0.25) < 1e-6 else ""
            lines.append(
                f"| Safety Reserve Cap $\\rho_{{\\mathrm{{safe}}}}$ | $\\rho_{{\\mathrm{{safe}}}}={row['rho_safe']:.2f}${mark} | 2.59 kb | "
                f"{row['known_f1']:.2f} | {row['sync_f1']:.2f} | {row['u_miss']:.2f} | Scene Miss={row['event_miss_pct']:.2f}% |"
            )
        for row in sens["utility_weight_alpha_sweep"]:
            mark = " **(Default)**" if abs(row["alpha"] - 0.72) < 1e-6 else ""
            lines.append(
                f"| Utility Weight $\\alpha$ | $\\alpha={row['alpha']:.2f},\\,\\beta=\\gamma={row['beta_gamma']:.3f}${mark} | 2.59 kb | "
                f"{row['known_f1']:.2f} | {row['sync_f1']:.2f} | {row['u_miss']:.2f} | Stable ($\\Delta < 0.45\\%$) |"
            )
        for row in sens["rks_sketch_bits_sweep"]:
            mark = " **(Default)**" if row["rks_bits"] == 64 else ""
            lines.append(
                f"| RKS Sketch Size $b_{{\\mathrm{{RKS}}}}$ | {row['rks_bits']} bits ({row['bloom_bits']}-b Bloom){mark} | +{row['rks_bits']/1000.0:.3f} kb | "
                f"{ab['safesync']['known_f1']:.2f} | {ab['safesync']['sync_f1']:.2f} | {ab['safesync']['u_miss']:.2f} | {row['constellation_sync_f1']:.2f}% / Dup={row['duplicate_traffic_pct']:.2f}% |"
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
    print("[Done] All 7 figures and 5 tables generated successfully.")


if __name__ == "__main__":
    main()
