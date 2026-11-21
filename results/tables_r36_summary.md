# SafeSync-Mesh Round 36 Exact Measured Tables (3 Seeds: 42, 1337, 2026)

## Table I: Main Rate-Task-Safety Comparison (Operating Point B_t = 2,600 bits / 2.60 kb/frame)

| Method | Rate (kb/frame) | BPP | Known-Task F1 (%) | Sync-F1 (%) | State Dist. $D_{\mathrm{sync}}$ | Unknown Miss (%) |
|---|---:|---:|---:|---:|---:|---:|
| Full feature | 9.34 ± 0.00 | 0.0089 | 75.89 ± 0.73 | 75.22 ± 0.39 | 0.0000 | 0.00 ± 0.00 |
| Conventional coding (Thumb+JPEG50) | 94.80 ± 0.00 | 0.0904 | 28.40 ± 0.00 | 25.81 ± 0.00 | 0.4820 | 74.19 ± 0.00 |
| Current Top-K | 2.61 ± 0.00 | 0.0025 | 46.50 ± 0.50 | 39.83 ± 0.28 | 0.3857 | 71.31 ± 1.02 |
| STARS-style | 2.60 ± 0.00 | 0.0025 | 54.21 ± 0.43 | 47.43 ± 0.22 | 0.3315 | 62.38 ± 0.89 |
| Change-only | 2.61 ± 0.00 | 0.0025 | 55.74 ± 0.38 | 41.95 ± 0.21 | 0.3062 | 97.97 ± 0.00 |
| Stateful utility-only | 2.61 ± 0.00 | 0.0025 | 69.39 ± 0.48 | 61.49 ± 0.60 | 0.2306 | 44.70 ± 2.23 |
| TWIST-inspired | 2.61 ± 0.00 | 0.0025 | 69.25 ± 0.33 | 57.25 ± 0.15 | 0.2441 | 60.30 ± 1.19 |
| SafeSync-Mesh (Ours) | 2.59 ± 0.00 | 0.0025 | 69.55 ± 0.59 | 65.95 ± 0.16 | 0.1951 | 22.93 ± 1.15 |

## Table II: Open-World Safety Under Matched Rate (Operating Point B_t = 2,600 bits)

| Method | Rate (kb/frame) | Known-Task F1 (%) | U-Recall (%) | Unknown Miss (%) | Scene Event Miss (%) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Current-frame Top-K | 2.61 | 46.50 | 28.69 ± 1.02 | 71.31 ± 1.02 | 20.09 | — |
| Stateful utility-only | 2.61 | 69.39 | 55.30 ± 2.23 | 44.70 ± 2.23 | 4.27 | — |
| SafeSync w/o calibration | 2.59 | 65.92 | 69.77 ± 0.08 | 30.23 ± 0.08 | 0.00 | 0.1680 |
| SafeSync (Ours) | 2.59 | 69.55 | 77.07 ± 1.15 | 22.93 ± 1.15 | 0.43 | 0.0143 |

## Table III: End-to-End Full-System 9-Variant Ablation Study (Single-Hop B_t = 2,600 bits & 24-Sat Constellation N_task = 200, 3 Seeds)

| Variant | Technical Pillar | Rate (kb) | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 (%) | Constellation U-Miss (%) | Confirm Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full SafeSync-Mesh (Ours) | Full System | 2.59 | 69.55 ± 0.59 | 65.95 ± 0.16 | 22.93 ± 1.15 | 68.20 | 18.34 | 17.5 |
| w/o Shared Prior (Stateless VQ) | Pillar 1 (Representation) | 2.60 | 51.94 ± 0.51 | 51.76 ± 0.17 | 29.00 ± 0.62 | 38.86 | 69.19 | 56.5 |
| w/o Conditional Entropy Coding | Pillar 1 (Representation) | 2.60 | 68.54 ± 0.41 | 63.94 ± 0.25 | 28.84 ± 0.22 | 64.31 | 25.58 | 21.1 |
| w/o Safety Reserve S_safe | Pillar 2 (Open-World Safety) | 2.61 | 69.39 ± 0.48 | 61.49 ± 0.60 | 44.70 ± 2.23 | 64.82 | 35.32 | 28.7 |
| w/o Risk Calibration (Raw Score) | Pillar 2 (Open-World Safety) | 2.59 | 65.92 ± 1.12 | 63.26 ± 0.52 | 30.23 ± 0.08 | 66.13 | 23.93 | 20.4 |
| w/o Confidence-Decay Fusion | Pillar 3 (Fusion & Relay) | 2.59 | 68.48 ± 0.97 | 65.13 ± 0.58 | 22.93 ± 1.15 | 67.55 | 18.39 | 19.2 |
| w/o 64-bit RKS State Sketch | Pillar 3 (Fusion & Relay) | 2.59 | 69.55 ± 0.59 | 65.95 ± 0.16 | 22.93 ± 1.15 | 45.05 | 50.81 | 54.4 |
| w/o root_id Provenance Dedup | Pillar 3 (Fusion & Relay) | 2.59 | 69.55 ± 0.59 | 65.95 ± 0.16 | 22.93 ± 1.15 | 29.93 | 61.97 | 53.6 |
| w/o Priority S_safe ISL Relay (FIFO) | Pillar 3 (Fusion & Relay) | 2.59 | 69.55 ± 0.59 | 65.95 ± 0.16 | 22.93 ± 1.15 | 67.78 | 20.19 | 39.6 |

## Table IV: Compute and Training Cost

| Method | Params (M) | Selector FLOPs (G) | Onboard Latency RTX3090 / Orin (ms) | Peak Mem. (MB) | Training GPU-h |
|---|---:|---:|---:|---:|---:|
| 5-Skip FP16 Split | 28.28 | 0.0000 | 86.50 / 363.3 | 612.0 | 0.00 |
| Conventional JPEG-90 | 28.28 | 0.0002 | 88.30 / 370.9 | 438.0 | 0.00 |
| FMSAT-style Generative | 142.60 | 18.4000 | 312.40 / 1418.0 | 2840.0 | 2.40 |
| Current-frame Top-K | 28.31 | 0.0004 | 87.34 / 366.8 | 412.4 | 0.18 |
| STARS-style adapted | 28.44 | 0.0012 | 88.05 / 369.8 | 418.8 | 0.26 |
| TWIST-inspired | 28.55 | 0.0021 | 88.65 / 372.3 | 424.6 | 0.34 |
| SafeSync (Ours) | 28.80 | 0.0034 | 89.43 / 375.6 | 427.4 | 0.38 |
