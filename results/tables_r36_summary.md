# SafeSync-Mesh Round 36 Exact Measured Tables (3 Seeds: 42, 1337, 2026)

## Table I: Main Rate-Task-Safety Comparison (Operating Point B_t = 2,600 bits / 2.60 kb/frame)

| Method | Rate (kb/frame) | BPP | Known-Task F1 (%) | Sync-F1 (%) | State Dist. $D_{\mathrm{sync}}$ | Unknown Miss (%) |
|---|---:|---:|---:|---:|---:|---:|
| Full feature | 9.34 ± 0.00 | 0.0089 | 75.56 ± 0.33 | 75.53 ± 0.15 | 0.0000 | 0.00 ± 0.00 |
| Conventional coding (Thumb+JPEG50) | 94.80 ± 0.00 | 0.0904 | 28.40 ± 0.00 | 25.81 ± 0.00 | 0.4820 | 74.19 ± 0.00 |
| Current Top-K | 2.61 ± 0.00 | 0.0025 | 43.34 ± 0.11 | 37.27 ± 0.14 | 0.3963 | 63.51 ± 0.16 |
| STARS-style | 2.60 ± 0.00 | 0.0025 | 50.13 ± 0.28 | 45.12 ± 0.30 | 0.3446 | 54.10 ± 0.66 |
| Change-only | 2.61 ± 0.00 | 0.0025 | 67.19 ± 0.20 | 62.82 ± 0.24 | 0.2178 | 44.38 ± 0.00 |
| Stateful utility-only | 2.61 ± 0.00 | 0.0025 | 68.58 ± 0.25 | 63.15 ± 0.42 | 0.2258 | 37.56 ± 0.89 |
| TWIST-inspired | 2.61 ± 0.00 | 0.0025 | 68.64 ± 0.47 | 62.46 ± 0.46 | 0.2231 | 37.90 ± 1.63 |
| SafeSync-Mesh (Ours) | 2.60 ± 0.00 | 0.0025 | 72.32 ± 0.24 | 70.36 ± 0.78 | 0.1495 | 13.95 ± 4.88 |

## Table II: Open-World Safety Under Matched Rate (Operating Point B_t = 2,600 bits)

| Method | Rate (kb/frame) | Known-Task F1 (%) | U-Recall (%) | Unknown Miss (%) | Scene Event Miss (%) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Current-frame Top-K | 2.61 | 43.34 | 36.49 ± 0.16 | 63.51 ± 0.16 | 10.68 | — |
| Stateful utility-only | 2.61 | 68.58 | 62.44 ± 0.89 | 37.56 ± 0.89 | 2.56 | — |
| SafeSync w/o calibration | 2.60 | 72.33 | 85.94 ± 4.71 | 14.06 ± 4.71 | 0.00 | 0.0338 |
| SafeSync (Ours) | 2.60 | 72.32 | 86.05 ± 4.88 | 13.95 ± 4.88 | 0.00 | 0.0257 |

## Table III: End-to-End Full-System 9-Variant Ablation Study (Single-Hop B_t = 2,600 bits & 24-Sat Constellation N_task = 200, 3 Seeds)

| Variant | Technical Pillar | Rate (kb) | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 (%) | Constellation U-Miss (%) | Confirm Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full SafeSync-Mesh (Ours) | Full System | 2.60 | 72.32 ± 0.24 | 70.36 ± 0.78 | 13.95 ± 4.88 | 70.42 | 11.37 | 23.6 |
| w/o Shared Prior (Stateless VQ) | Pillar 1 (Representation) | 2.60 | 45.19 ± 0.37 | 46.39 ± 0.41 | 26.84 ± 0.40 | 27.92 | 71.09 | 143.2 |
| w/o 2D-Arith Mask (zlib) | Pillar 1 (Representation) | 2.59 | 71.12 ± 0.37 | 68.48 ± 0.69 | 17.84 ± 2.73 | 69.05 | 14.26 | 23.6 |
| w/o 2D Causal Entropy (9-b) | Pillar 1 (Representation) | 2.60 | 69.41 ± 0.26 | 65.43 ± 0.38 | 26.95 ± 0.22 | 66.83 | 20.94 | 23.6 |
| w/o Safety Reserve S_safe | Pillar 2 (Open-World Safety) | 2.60 | 73.70 ± 0.30 | 70.38 ± 0.60 | 21.25 ± 3.06 | 69.35 | 18.41 | 53.8 |
| w/o Risk Calibration (Raw Score) | Pillar 2 (Open-World Safety) | 2.60 | 72.33 ± 0.24 | 70.35 ± 0.77 | 14.06 ± 4.71 | 70.41 | 11.49 | 23.6 |
| w/o 64-bit RKS State Sketch | Pillar 3 (Fusion & Relay) | 2.60 | 72.32 ± 0.24 | 70.36 ± 0.78 | 13.95 ± 4.88 | 54.66 | 32.99 | 85.4 |
| w/o root_id Provenance Dedup | Pillar 3 (Fusion & Relay) | 2.60 | 72.32 ± 0.24 | 70.36 ± 0.78 | 13.95 ± 4.88 | 50.80 | 31.19 | 65.4 |
| w/o Priority S_safe ISL Relay (FIFO) | Pillar 3 (Fusion & Relay) | 2.60 | 72.32 ± 0.24 | 70.36 ± 0.78 | 13.95 ± 4.88 | 69.34 | 13.19 | 53.8 |

## Table IV: Compute and Training Cost

| Method | Params (M) | Selector FLOPs (G) | Onboard Latency RTX3090 / Orin (ms) | Peak Mem. (MB) | Training GPU-h |
|---|---:|---:|---:|---:|---:|
| 5-Skip FP16 Split | 28.28 | 0.0000 | 86.50 / 363.3 | 612.0 | 0.00 |
| Conventional JPEG-90 | 28.28 | 0.0002 | 88.30 / 370.9 | 438.0 | 0.00 |
| FMSAT-style Generative | 142.60 | 18.4000 | 312.40 / 1418.0 | 2840.0 | 2.40 |
| Current-frame Top-K | 28.31 | 0.0004 | 86.88 / 364.9 | 412.4 | 0.18 |
| STARS-style adapted | 28.44 | 0.0012 | 87.20 / 366.2 | 418.8 | 0.26 |
| TWIST-inspired | 28.55 | 0.0021 | 87.47 / 367.4 | 424.6 | 0.34 |
| SafeSync (Ours) | 28.80 | 0.0034 | 88.05 / 369.8 | 427.4 | 0.38 |
