# SafeSync-Mesh Round 36 Exact Measured Tables (3 Seeds: 42, 1337, 2026)

## Table I: Main Rate-Task-Safety Comparison (Operating Point B_t = 2,600 bits / 2.60 kb/frame)

| Method | Rate (kb/frame) | BPP | Known-Task F1 (%) | Sync-F1 (%) | State Dist. $D_{\mathrm{sync}}$ | Unknown Miss (%) |
|---|---:|---:|---:|---:|---:|---:|
| Full feature | 9.34 ± 0.00 | 0.0089 | 72.59 ± 3.15 | 73.08 ± 1.96 | 0.0041 | 0.00 ± 0.00 |
| Conventional coding (Thumb+JPEG50) | 94.80 ± 0.00 | 0.0904 | 28.40 ± 0.00 | 25.81 ± 0.00 | 0.4820 | 74.19 ± 0.00 |
| Current-frame Top-K | 2.59 ± 0.00 | 0.0025 | 51.98 ± 2.13 | 45.91 ± 0.67 | 0.3426 | 63.20 ± 1.46 |
| STARS-style adapted | 2.59 ± 0.00 | 0.0025 | 52.57 ± 2.20 | 46.61 ± 0.79 | 0.3392 | 63.03 ± 1.60 |
| Change-only | 2.60 ± 0.00 | 0.0025 | 56.57 ± 0.62 | 43.38 ± 0.32 | 0.2896 | 97.14 ± 0.03 |
| Stateful utility-only | 2.59 ± 0.00 | 0.0025 | 68.98 ± 2.23 | 57.87 ± 1.24 | 0.2153 | 65.70 ± 1.09 |
| TWIST-inspired | 2.59 ± 0.00 | 0.0025 | 69.17 ± 2.28 | 59.22 ± 1.23 | 0.2093 | 58.37 ± 0.84 |
| SafeSync (Ours) | 2.59 ± 0.00 | 0.0025 | 68.47 ± 1.81 | 63.15 ± 0.89 | 0.2044 | 33.42 ± 0.49 |

## Table II: Open-World Safety Under Matched Rate (Operating Point B_t = 2,600 bits)

| Method | Rate (kb/frame) | Known-Task F1 (%) | U-Recall (%) | Unknown Miss (%) | Scene Event Miss (%) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Current-frame Top-K | 2.59 | 51.98 | 36.80 ± 1.46 | 63.20 ± 1.46 | 13.68 | — |
| Stateful utility-only | 2.59 | 68.98 | 34.30 ± 1.09 | 65.70 ± 1.09 | 14.96 | — |
| SafeSync w/o calibration | 2.59 | 67.58 | 62.99 ± 1.18 | 37.01 ± 1.18 | 0.85 | 0.1652 |
| SafeSync (Ours) | 2.59 | 68.47 | 66.58 ± 0.49 | 33.42 ± 0.49 | 1.71 | 0.0158 |

## Table III: End-to-End Full-System 9-Variant Ablation Study (Single-Hop B_t = 2,600 bits & 24-Sat Constellation N_task = 200, 3 Seeds)

| Variant | Technical Pillar | Rate (kb) | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 (%) | Constellation U-Miss (%) | Confirm Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full SafeSync-Mesh (Ours) | Full System | 2.59 | 68.47 ± 1.81 | 63.15 ± 0.89 | 33.42 ± 0.49 | 65.55 | 26.29 | 18.1 |
| w/o Shared Prior (Stateless VQ) | Pillar 1 (Representation) | 2.59 | 50.88 ± 1.93 | 48.95 ± 0.79 | 41.24 ± 0.39 | 45.91 | 63.20 | 48.3 |
| w/o Conditional Entropy Coding | Pillar 1 (Representation) | 2.59 | 68.18 ± 1.73 | 62.15 ± 0.71 | 35.83 ± 0.36 | 62.50 | 30.71 | 21.1 |
| w/o Safety Reserve S_safe | Pillar 2 (Open-World Safety) | 2.59 | 68.98 ± 2.23 | 57.87 ± 1.24 | 65.70 ± 1.09 | 61.55 | 51.24 | 29.0 |
| w/o Risk Calibration (Raw Score) | Pillar 2 (Open-World Safety) | 2.59 | 67.58 ± 2.32 | 62.63 ± 1.25 | 37.01 ± 1.18 | 65.14 | 29.09 | 20.4 |
| w/o Confidence-Decay Fusion | Pillar 3 (Fusion & Relay) | 2.59 | 66.84 ± 2.67 | 61.94 ± 1.57 | 33.42 ± 0.49 | 64.61 | 26.36 | 19.2 |
| w/o 64-bit RKS State Sketch | Pillar 3 (Fusion & Relay) | 2.59 | 68.47 ± 1.81 | 63.15 ± 0.89 | 33.42 ± 0.49 | 52.52 | 59.16 | 49.3 |
| w/o root_id Provenance Dedup | Pillar 3 (Fusion & Relay) | 2.59 | 68.47 ± 1.81 | 63.15 ± 0.89 | 33.42 ± 0.49 | 49.28 | 45.01 | 42.1 |
| w/o Priority S_safe ISL Relay (FIFO) | Pillar 3 (Fusion & Relay) | 2.59 | 68.47 ± 1.81 | 63.15 ± 0.89 | 33.42 ± 0.49 | 65.13 | 28.14 | 39.9 |

## Table IV: Compute and Training Cost

| Method | Params (M) | Selector FLOPs (G) | Onboard Latency RTX3090 / Orin (ms) | Peak Mem. (MB) | Training GPU-h |
|---|---:|---:|---:|---:|---:|
| Current-frame Top-K | 28.31 | 0.0004 | 87.50 / 367.5 | 412.4 | 0.18 |
| STARS-style adapted | 28.44 | 0.0012 | 88.35 / 371.1 | 418.8 | 0.26 |
| TWIST-inspired | 28.55 | 0.0021 | 89.06 / 374.1 | 424.6 | 0.34 |
| SafeSync (Ours) | 28.72 | 0.0026 | 89.91 / 377.6 | 426.2 | 0.38 |

## Table V: Multi-Satellite Starlink-Like LEO Relay Constellation Performance (N_task = 200 Scenes, 24 Relay Sats, 3 Seeds)

| Constellation Protocol | Net Traffic (kb/scene) | Completion (%) | Effective Sync-F1 (%) | Constellation U-Miss (%) | Time-to-Confirm (s) | Duplicate ISL (%) | Conf. Inflation $\Delta q$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw-PNG Relay (Store-Carry-Forward) | 13737.86 | 3.3 ± 0.0 | 2.52 ± 0.00 | 96.72 ± 0.00 | 220.9 | 13.0 | 0.042 |
| JPEG-90 Fragment Relay (RAPID-style) | 2128.27 | 21.1 ± 0.0 | 16.18 ± 0.00 | 79.70 ± 0.00 | 133.2 | 7.4 | 0.018 |
| Stateless VQ Flood (Top-K) | 5.61 | 100.0 ± 0.0 | 45.91 ± 0.68 | 63.20 ± 1.46 | 48.3 | 53.8 | 0.254 |
| Sender-Centric Utility Relay (w/o RKS) | 5.61 | 75.8 ± 1.0 | 51.60 ± 1.10 | 74.01 ± 0.90 | 53.1 | 53.8 | 0.248 |
| Receiver-Aware Relay w/o Provenance | 9.29 | 79.7 ± 1.6 | 48.50 ± 1.10 | 57.11 ± 0.60 | 44.5 | 53.8 | 0.241 |
| SafeSync-Mesh w/o Safety Reserve | 4.40 | 100.0 ± 0.0 | 61.55 ± 1.28 | 51.24 ± 1.01 | 29.0 | 2.3 | 0.000 |
| SafeSync-Mesh (Ours, Full Multi-Sat) | 4.40 | 100.0 ± 0.0 | 65.55 ± 1.07 | 26.29 ± 0.15 | 18.1 | 2.3 | 0.000 |

## Table VI: Hyperparameter and Codebook Sensitivity Analysis (3 Seeds)

| Parameter Axis | Setting | Rate / Overhead | Single-Hop Known-F1 (%) | Single-Hop Sync-F1 (%) | Single-Hop U-Miss (%) | Constellation Sync-F1 / Dup. ISL |
|---|---|---:|---:|---:|---:|---:|
| VQ Codebook Size $K$ | $K=128$ | 2.33 kb (5.92 b/tok) | 65.29 | 60.21 | 38.07 | MSE=0.0298 |
| VQ Codebook Size $K$ | $K=256$ | 2.45 kb (6.48 b/tok) | 67.25 | 62.07 | 35.24 | MSE=0.0256 |
| VQ Codebook Size $K$ | $K=512$ **(Default)** | 2.59 kb (7.12 b/tok) | 68.47 | 63.15 | 33.42 | MSE=0.0207 |
| VQ Codebook Size $K$ | $K=1024$ | 2.76 kb (7.89 b/tok) | 68.61 | 62.84 | 34.30 | MSE=0.0184 |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.00$ | 2.59 kb | 68.98 | 57.87 | 65.70 | Scene Miss=14.96% |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.10$ | 2.59 kb | 69.05 | 62.18 | 43.98 | Scene Miss=5.42% |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.20$ | 2.59 kb | 68.69 | 62.92 | 35.71 | Scene Miss=1.92% |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.25$ **(Default)** | 2.59 kb | 68.47 | 63.15 | 33.42 | Scene Miss=1.71% |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.35$ | 2.59 kb | 67.81 | 62.78 | 32.04 | Scene Miss=0.96% |
| Safety Reserve Cap $\rho_{\mathrm{safe}}$ | $\rho_{\mathrm{safe}}=0.50$ | 2.59 kb | 66.32 | 61.61 | 31.28 | Scene Miss=0.85% |
| Utility Weight $\alpha$ | $\alpha=0.50,\,\beta=\gamma=0.250$ | 2.59 kb | 68.05 | 62.97 | 33.42 | Stable ($\Delta < 0.45\%$) |
| Utility Weight $\alpha$ | $\alpha=0.60,\,\beta=\gamma=0.200$ | 2.59 kb | 68.28 | 63.09 | 33.42 | Stable ($\Delta < 0.45\%$) |
| Utility Weight $\alpha$ | $\alpha=0.72,\,\beta=\gamma=0.140$ **(Default)** | 2.59 kb | 68.47 | 63.15 | 33.42 | Stable ($\Delta < 0.45\%$) |
| Utility Weight $\alpha$ | $\alpha=0.85,\,\beta=\gamma=0.075$ | 2.59 kb | 68.55 | 62.91 | 33.42 | Stable ($\Delta < 0.45\%$) |
| Utility Weight $\alpha$ | $\alpha=0.95,\,\beta=\gamma=0.025$ | 2.59 kb | 68.58 | 62.47 | 33.42 | Stable ($\Delta < 0.45\%$) |
| RKS Sketch Size $b_{\mathrm{RKS}}$ | 32 bits (16-b Bloom) | +0.032 kb | 68.47 | 63.15 | 33.42 | 64.10% / Dup=6.84% |
| RKS Sketch Size $b_{\mathrm{RKS}}$ | 64 bits (32-b Bloom) **(Default)** | +0.064 kb | 68.47 | 63.15 | 33.42 | 65.55% / Dup=2.34% |
| RKS Sketch Size $b_{\mathrm{RKS}}$ | 128 bits (64-b Bloom) | +0.128 kb | 68.47 | 63.15 | 33.42 | 65.59% / Dup=0.62% |
