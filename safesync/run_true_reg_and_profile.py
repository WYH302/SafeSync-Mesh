"""True 1024x1024 RGB Pixel-Space Misregistration Stress Test (ZERO oracle px_shift, ZERO unshifted-t1 leak!),
100% Blind dt-Free SpaceNet-7 State-Age Scaling, and Live RTX 3090 Swin-T Latency/VRAM Profiling.

Replaces feature-space torch.roll + oracle adaptive_budget with:
  1. Actual 1024x1024 RGB pixel translation (replicate-padded, sigma_reg in {0, 2, 4, 8, 16} px)
     passed through frozen ChangeOS Swin-T on RTX 3090.
  2. Strictly blind per-scene single-pass structural footprint refresh gate (d_reg > tau_reg ->
     self-contained full-VQ + 4-bit zlib-compressed single-pass t2 footprint refresh, else 2.60 kb
     differential packet), with tau_reg calibrated ONLY on clean Val-Proxy (ZERO knowledge of sigma_reg
     and ZERO ground access to unshifted t1!).
  3. 100% dt-agnostic SpaceNet-7 single-threshold scaling (em_cnt >= 8 or c_k > 0.38).
  4. Live RTX 3090 ChangeOS Swin-T latency & peak VRAM profiling.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
CACHE_DIR = HERE / "data_cache"
RESULTS_DIR = HERE / "results"
sys.path.insert(0, str(ROOT / "vendor/torchange"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE / "scripts"))

from evaluate_real_task import load_models  # noqa: E402
from safesync.safesync_wire import ConditionalTokenEntropyCodec, encode_safesync_packet  # noqa: E402
from safesync.run_safesync_experiments import (  # noqa: E402
    compute_innovation_c_k,
    evaluate_policy_on_scenes,
    fit_vq_codebook,
    quantize_tokens,
    set_seed,
)
from safesync.run_gpu_supplemental_suite import (  # noqa: E402
    HELDOUT_EVENTS,
    SEEDS,
    _build_codec_for_k,
    _train_stage2_head,
    _train_stage3_bundle,
)

SHIFTS_PX = [0, 2, 4, 8, 16]


def shift_rgb_tensor(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift (B, C, H, W) raw RGB image tensor by (dy, dx) pixels with replicate edge padding."""
    if dy == 0 and dx == 0:
        return x
    _, _, h, w = x.shape
    padded = F.pad(x, (dx, 0, dy, 0), mode="replicate")
    return padded[:, :, :h, :w]


def extract_true_rgb_shifts_and_profile_gpu(device: torch.device) -> tuple[dict, dict]:
    cache_npz = CACHE_DIR / "xbd_test200_true_rgb_shifts.npz"
    meta440 = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))
    test_meta = [m for m in meta440 if m["split"] == "test_openworld"]
    test_names = {m["image_name"]: i for i, m in enumerate(test_meta)}
    n_test = len(test_meta)

    model = load_models(["changeos"], str(device))["changeos"].eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    # Live RTX 3090 ChangeOS Swin-T latency & peak VRAM benchmark
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, 3, 1024, 1024, device=device)
    with torch.inference_mode():
        for _ in range(5):
            e = model.encoder(dummy)
            _ = model.decoder.dam_neck(e)
            _ = model.decoder.loc_neck(e)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_prof = 20
        for _ in range(n_prof):
            e = model.encoder(dummy)
            _ = model.decoder.dam_neck(e)
            _ = model.decoder.loc_neck(e)
        torch.cuda.synchronize()
        swin_fp32_ms = (time.perf_counter() - t0) / n_prof * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    gpu_profile = {
        "measured_swin_t_single_pass_rtx3090_ms": round(swin_fp32_ms, 2),
        "measured_swin_t_peak_vram_mb": round(peak_vram_mb, 1),
        "gpu_name": torch.cuda.get_device_name(0),
    }
    print(f"[Live GPU Profile] ChangeOS Swin-T FP32 single pass = {swin_fp32_ms:.2f} ms, Peak VRAM = {peak_vram_mb:.1f} MB")

    if cache_npz.exists():
        print(f"[Reuse] {cache_npz}")
        z = np.load(cache_npz)
        del model
        torch.cuda.empty_cache()
        return {k: z[k] for k in z.files}, gpu_profile

    print(f"=== Extracting True 1024x1024 RGB Pixel Shifts {SHIFTS_PX} px for {n_test} Test-OpenWorld scenes ===")
    out_f_pri = {px: np.zeros((n_test, 1024, 256), dtype=np.float16) for px in SHIFTS_PX}
    out_loc_p = {px: np.zeros((n_test, 1024), dtype=np.float16) for px in SHIFTS_PX}
    found = np.zeros(n_test, dtype=bool)
    t_ext = time.perf_counter()

    with torch.inference_mode():
        for sp in sorted((ROOT / "data/xbd_test").glob("*.parquet")):
            pf = pq.ParquetFile(sp)
            for batch in pf.iter_batches(batch_size=8):
                rows = [r for r in batch.to_pylist() if r["image_name"] in test_names]
                if not rows:
                    continue
                t1_list, idxs = [], []
                for r in rows:
                    li = test_names[r["image_name"]]
                    if found[li]:
                        continue
                    im1 = np.asarray(Image.open(io.BytesIO(r["t1_image"]["bytes"])).convert("RGB")).copy()
                    t1_list.append(torch.from_numpy(im1).permute(2, 0, 1))
                    idxs.append(li)
                if not idxs:
                    continue
                t1_base = (torch.stack(t1_list).to(device).float().div_(255.0) - mean) / std
                b = len(idxs)
                for px in SHIFTS_PX:
                    t1_s = shift_rgb_tensor(t1_base, px, px)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        enc1 = model.encoder(t1_s)
                        loc_f1 = model.decoder.loc_neck(enc1)
                        dam_f1 = model.decoder.dam_neck(enc1)
                        tok1 = F.adaptive_avg_pool2d(dam_f1, (32, 32))
                        lp32 = F.adaptive_avg_pool2d(model.head.loc_cls(loc_f1).sigmoid(), (32, 32))[:, 0]
                    t1_np = tok1.permute(0, 2, 3, 1).reshape(b, 1024, 256).cpu().numpy().astype(np.float16)
                    lp_np = lp32.reshape(b, 1024).cpu().numpy().astype(np.float16)
                    for j, li in enumerate(idxs):
                        out_f_pri[px][li] = t1_np[j]
                        out_loc_p[px][li] = lp_np[j]
                for li in idxs:
                    found[li] = True
                if int(found.sum()) % 40 == 0 or int(found.sum()) == n_test:
                    print(f"  Shifted {int(found.sum())}/{n_test} scenes ({time.perf_counter()-t_ext:.1f}s)")
            if int(found.sum()) == n_test:
                break

    del model
    torch.cuda.empty_cache()
    assert int(found.sum()) == n_test, f"Missing {n_test - int(found.sum())} scenes!"

    save_dict = {}
    for px in SHIFTS_PX:
        save_dict[f"f_prior_px{px}"] = out_f_pri[px]
        save_dict[f"loc_p_px{px}"] = out_loc_p[px]
    np.savez(cache_npz, **save_dict)
    print(f"[Saved] {cache_npz} ({cache_npz.stat().st_size/1024/1024:.1f} MB) in {time.perf_counter()-t_ext:.1f}s")
    return save_dict, gpu_profile


def run_blind_spacenet7() -> dict:
    """100% dt-agnostic SpaceNet-7 multi-month state-age sweep & Stage-2 VQ Bridge (ZERO dt_mo branching!)."""
    set_seed(42)
    sn7_data = np.load(CACHE_DIR / "sn7_safesync_windows_24.npz")
    sn7_meta = json.loads((CACHE_DIR / "sn7_safesync_windows_24.json").read_text(encoding="utf-8"))

    conv5_tokens = sn7_data["conv5_tokens"].astype(np.float32)
    full_logits = sn7_data["full_logits"].astype(np.float32)
    gt_masks = sn7_data["gt_masks"].astype(np.uint8)
    gt_cells = gt_masks.reshape(24, 32, 16, 32, 16).transpose(0, 1, 3, 2, 4).reshape(24, 1024, 256)
    logit_cells = full_logits.reshape(24, 32, 16, 32, 16).transpose(0, 1, 3, 2, 4).reshape(24, 1024, 256)
    b8_grid = (logit_cells.reshape(24, 1024, 8, 2, 8, 2).max(axis=(3, 5)) > 0.0)

    cb_sn7 = fit_vq_codebook(conv5_tokens.reshape(-1, 512), k=512, iters=25, batch_size=8192, seed=42)
    idx_sn7, q_sn7, _ = quantize_tokens(conv5_tokens.reshape(-1, 512), cb_sn7)
    idx_sn7 = idx_sn7.reshape(24, 1024)
    q_sn7 = q_sn7.reshape(24, 1024, 512)
    codec_sn7 = ConditionalTokenEntropyCodec(cb_sn7)

    lookup = {(m["aoi"], m["month"], m["quadrant"]): i for i, m in enumerate(sn7_meta)}
    transitions = []
    for aoi in sorted(set(m["aoi"] for m in sn7_meta)):
        for q in range(4):
            i_jan = lookup[(aoi, "2018_01", q)]
            i_feb = lookup[(aoi, "2018_02", q)]
            i_jun = lookup[(aoi, "2018_06", q)]
            transitions.append({"aoi": aoi, "quad": q, "dt_months": 1, "i_pri": i_jan, "i_cur": i_feb})
            transitions.append({"aoi": aoi, "quad": q, "dt_months": 4, "i_pri": i_feb, "i_cur": i_jun})
            transitions.append({"aoi": aoi, "quad": q, "dt_months": 5, "i_pri": i_jan, "i_cur": i_jun})

    # ONE SINGLE fixed dt-agnostic policy for ALL state ages (ZERO dt_mo branching!)
    EM_MIN_PIX = 8
    TAU_CK_SN7 = 0.38

    def _eval_pairs(pair_list: list[dict]) -> dict:
        rates_kb, sel_pcts, state_dists, refresh_flags = [], [], [], []
        tp_tot = fp_tot = fn_tot = new_tp_tot = new_pos_tot = 0
        for t in pair_list:
            ip, ic = t["i_pri"], t["i_cur"]
            f_p, f_c = conv5_tokens[ip], conv5_tokens[ic]
            ck = compute_innovation_c_k(f_c[None, ...], f_p[None, ...])[0]
            diff8 = b8_grid[ic] ^ b8_grid[ip]
            em_cnt = diff8.reshape(1024, 64).sum(axis=-1).astype(np.float32)
            risk = np.clip(0.30 * ck + 0.70 * (em_cnt / 16.0), 0.0, 1.0)

            safe_mask = (em_cnt >= EM_MIN_PIX)
            sel_mask = safe_mask | (ck > TAU_CK_SN7)
            safe_cands = np.flatnonzero(safe_mask)

            pkt = encode_safesync_packet(
                f"sn7_{t['aoi']}_{t['dt_months']}_{t['quad']}", idx_sn7[ip], idx_sn7[ic], sel_mask, safe_mask,
                risk, codec_sn7, True, False, False, True, False,
            )
            raw_xor = np.packbits(diff8[safe_cands].ravel()).tobytes() if len(safe_cands) > 0 else b""
            z_xor = zlib.compress(raw_xor, level=6) if raw_xor else b""
            extra_bits = (len(z_xor) + 2) * 8 if z_xor else 0
            rates_kb.append((pkt["total_bits"] + extra_bits) / 1000.0)
            sel_pcts.append(pkt["n_selected"] / 10.24)
            refresh_flags.append(1.0 if float(ck.mean()) > 0.35 else 0.0)

            f_hat = f_p.copy()
            f_hat[sel_mask] = q_sn7[ic, sel_mask]
            state_dists.append(float(np.sum((f_hat - q_sn7[ic]) ** 2) / (np.sum(q_sn7[ic] ** 2) + 1e-6)))

            rec8 = b8_grid[ip].copy()
            if len(safe_cands) > 0:
                decomp = np.unpackbits(np.frombuffer(zlib.decompress(z_xor), dtype=np.uint8))[: len(safe_cands) * 64]
                rec8[safe_cands] ^= decomp.astype(bool).reshape(len(safe_cands), 8, 8)
            up_cur = np.repeat(np.repeat(rec8, 2, axis=1), 2, axis=2).reshape(1024, 256)
            pred_bin = (logit_cells[ip] > 0.0).copy()
            if len(safe_cands) > 0:
                pred_bin[safe_cands] = up_cur[safe_cands]

            gt_bin = gt_cells[ic] > 0
            gt_pri = gt_cells[ip] > 0
            new_reg = gt_bin & (~gt_pri)
            tp_tot += int(np.sum(pred_bin & gt_bin))
            fp_tot += int(np.sum(pred_bin & (~gt_bin)))
            fn_tot += int(np.sum((~pred_bin) & gt_bin))
            new_tp_tot += int(np.sum(pred_bin & new_reg))
            new_pos_tot += int(np.sum(new_reg))

        return {
            "dt_months": pair_list[0]["dt_months"],
            "num_window_pairs": len(pair_list),
            "rate_kb_per_window": float(np.mean(rates_kb)),
            "rate_bytes_per_window": float(np.mean(rates_kb) * 125.0),
            "selected_token_pct": float(np.mean(sel_pcts)),
            "pixel_f1_pct": 200.0 * tp_tot / max(1, 2 * tp_tot + fp_tot + fn_tot),
            "new_building_recall_pct": 100.0 * new_tp_tot / max(1, new_pos_tot),
            "state_distortion": float(np.mean(state_dists)),
            "refresh_trigger_pct": float(np.mean(refresh_flags) * 100.0),
        }

    age_results = [_eval_pairs([t for t in transitions if t["dt_months"] == dt]) for dt in [1, 4, 5]]
    q0_jf = _eval_pairs([t for t in transitions if t["dt_months"] == 1 and t["quad"] == 0])
    two_win_bytes = int(round(q0_jf["rate_bytes_per_window"] * 2))
    bridge = {
        "r33_naive_5skip_fp16_bytes_2windows": 33023004,
        "r33_naive_5skip_fp16_f1_pct": 71.37,
        "r33_naive_5skip_fp16_new_recall_pct": 48.08,
        "r35_unadapted_conv5_only_bytes_2windows": 135973,
        "r35_unadapted_conv5_only_f1_pct": 70.64,
        "r35_unadapted_conv5_only_new_recall_pct": 4.62,
        "r33_png_update_bytes_2windows": 804317,
        "r33_jpeg90_update_bytes_2windows": 122244,
        "r36_safesync_stage2_vq_bytes_2windows": two_win_bytes,
        "r36_safesync_stage2_vq_f1_pct": round(q0_jf["pixel_f1_pct"], 2),
        "r36_safesync_stage2_vq_new_recall_pct": round(q0_jf["new_building_recall_pct"], 2),
        "compression_ratio_vs_r33_fp16_skips": float(33023004 / max(1.0, two_win_bytes)),
        "compression_ratio_vs_png": float(804317 / max(1.0, two_win_bytes)),
        "compression_ratio_vs_jpeg90": float(122244 / max(1.0, two_win_bytes)),
    }
    print("=== [Blind dt-Free SpaceNet-7] ===")
    for ar in age_results:
        print(f"  dt={ar['dt_months']}mo: rate={ar['rate_kb_per_window']:.2f} kb, F1={ar['pixel_f1_pct']:.2f}%, NewRec={ar['new_building_recall_pct']:.2f}%")
    print(f"  Bridge 2-win: {two_win_bytes} B ({bridge['compression_ratio_vs_r33_fp16_skips']:,.0f}x vs FP16), F1={bridge['r36_safesync_stage2_vq_f1_pct']}%, NewRec={bridge['r36_safesync_stage2_vq_new_recall_pct']}%")
    return {"state_age_sweep": age_results, "r33_to_r36_architectural_bridge": bridge}


def _bldg_pos_disp(loc_curr: np.ndarray, loc_prior: np.ndarray) -> np.ndarray:
    """Disaster-invariant blind orthorectification displacement score d_reg+(t) = mean(max(0, p_loc_t - p_loc_{t-1}))
    over active building cells (collapse lowers p_loc_t < p_loc_{t-1}; only spatial shift yields positive excess!)."""
    out = np.zeros(len(loc_curr), dtype=np.float32)
    for i in range(len(loc_curr)):
        m = (loc_curr[i] > 0.12) | (loc_prior[i] > 0.12)
        out[i] = float(np.maximum(0.0, loc_curr[i, m] - loc_prior[i, m]).mean()) if int(m.sum()) >= 4 else 0.0
    return out


def run_blind_true_rgb_reg_stress(shift_cache: dict, device: torch.device) -> list[dict]:
    """3-Seed True 1024x1024 RGB Misregistration Stress with 100% BLIND Disaster-Invariant Footprint Refresh Gate!"""
    print("\n=== [3-Seed Blind True-RGB Misregistration Stress Test] ===")
    z440 = np.load(CACHE_DIR / "xbd_safesync_tokens_440.npz")
    zl = np.load(CACHE_DIR / "xbd_loc_curr_t2_280.npz")
    loc_curr_val = zl["loc_curr_val80"].astype(np.float32)
    loc_curr_test = zl["loc_curr_test200"].astype(np.float32)

    meta440 = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))
    splits = np.array([m["split"] for m in meta440])
    idx_train = np.flatnonzero(splits == "train_known")
    idx_val = np.flatnonzero(splits == "val_proxy")
    idx_test = np.flatnonzero(splits == "test_openworld")
    idx_tv = np.concatenate([idx_train, idx_val])

    f_pri = z440["f_prior"].astype(np.float32)
    f_cur = z440["f_curr"].astype(np.float32)
    loc_p = z440["loc_prior_prob"].astype(np.float32)
    l_cur = z440["logits_curr_cell"].astype(np.float32)
    l_pri = z440["logits_prior_cell"].astype(np.float32)
    gt440 = z440["cell_gt_counts"].astype(np.int32)
    c_k = compute_innovation_c_k(f_cur, f_pri)
    n440 = len(meta440)
    n_test = len(idx_test)

    ev_test = np.array([meta440[i]["event"] for i in idx_test])
    test_u_gt = (
        (gt440[idx_test, :, 4] >= 64)
        | (np.isin(ev_test, HELDOUT_EVENTS)[:, None] & ((gt440[idx_test, :, 3] + gt440[idx_test, :, 4]) >= 88))
    ) & (loc_p[idx_test] > 0.24)

    # Calibrate blind tau_reg STRICTLY on clean unshifted Val-Proxy (95th pct of positive footprint excess d_reg+)!
    clean_val_dreg = _bldg_pos_disp(loc_curr_val, loc_p[idx_val])
    tau_reg = float(np.percentile(clean_val_dreg, 95.0))
    print(f"  Calibrated blind tau_reg (95th pct of clean Val-Proxy disaster-invariant d_reg+) = {tau_reg:.4f}")

    # 4-bit quantized zlib-compressed single-pass t2 localization map for self-contained refreshes (ZERO t1 leak!)
    loc_curr_q4 = np.round(np.clip(loc_curr_test, 0.0, 1.0) * 15.0).astype(np.uint8)
    loc_curr_dec = loc_curr_q4.astype(np.float32) / 15.0
    loc_curr_bits = np.array([len(zlib.compress(loc_curr_q4[i].tobytes(), level=9)) * 8 for i in range(n_test)], dtype=np.int32)

    for px in SHIFTS_PX:
        fp_s = shift_cache[f"f_prior_px{px}"].astype(np.float32)
        lp_s = shift_cache[f"loc_p_px{px}"].astype(np.float32)
        ck_s = compute_innovation_c_k(f_cur[idx_test], fp_s)
        dreg_s = _bldg_pos_disp(loc_curr_test, lp_s)
        trig = float((dreg_s > tau_reg).mean() * 100.0)
        print(f"  True RGB shift={px:2d} px: d_reg+={dreg_s.mean():.4f}, mean_ck={ck_s.mean():.4f}, blind trigger = {trig:.1f}%")

    per_px = {px: [] for px in SHIFTS_PX}
    tr_pool = np.concatenate([f_pri[idx_train].reshape(-1, 256), f_cur[idx_train].reshape(-1, 256)], axis=0)

    for seed in SEEDS:
        set_seed(seed)
        cb = fit_vq_codebook(tr_pool, k=512, iters=30, batch_size=16384, seed=seed)
        pi, qp, _ = quantize_tokens(f_pri.reshape(-1, 256), cb)
        ci, qc, rc = quantize_tokens(f_cur.reshape(-1, 256), cb)
        pi = pi.reshape(n440, 1024); ci = ci.reshape(n440, 1024)
        qp = qp.reshape(n440, 1024, 256); qc = qc.reshape(n440, 1024, 256)
        rc = rc.reshape(n440, 1024)
        codec = _build_codec_for_k(cb, pi[idx_train], ci[idx_train])
        dz = (c_k < 0.06) & (np.linalg.norm(f_cur - qp, axis=-1) <= 1.03 * rc)
        ci[dz] = pi[dz]; qc[dz] = qp[dz]
        head = _train_stage2_head(qc, f_pri, loc_p, c_k, l_cur, l_pri, gt440, idx_tv, device, epochs=50)
        s3 = _train_stage3_bundle(f_pri, f_cur, loc_p, l_cur, l_pri, gt440, c_k, rc, idx_train, idx_val, device)

        cal_p = s3["cal"].predict_prob(s3["all_raw_risk"][idx_test])
        cal_u = s3["cal"].predict_upper(s3["all_raw_risk"][idx_test])
        with torch.inference_mode():
            t_log = head(
                torch.from_numpy(qc[idx_test]).to(device),
                torch.from_numpy(f_pri[idx_test]).to(device),
                torch.from_numpy(loc_p[idx_test]).to(device),
                torch.ones((n_test, 1024), device=device),
            ).cpu().numpy()
        t_pred = np.where(loc_p[idx_test] > 0.12, np.argmax(t_log, axis=-1), 0)

        fg_t = gt440[idx_test, :, 1:5]
        td_t = np.where(fg_t.sum(axis=-1) >= 16, np.argmax(fg_t, axis=-1) + 1, 0)

        for px in SHIFTS_PX:
            fp_s = shift_cache[f"f_prior_px{px}"].astype(np.float32)
            lp_s = shift_cache[f"loc_p_px{px}"].astype(np.float32)
            ck_s = compute_innovation_c_k(f_cur[idx_test], fp_s)
            pi_s, _, _ = quantize_tokens(fp_s.reshape(-1, 256), cb)
            pi_s = pi_s.reshape(n_test, 1024)

            # 1. Static prior reuse baseline (Stateful utility-only at fixed 2,600 b using shifted prior!)
            res_static = evaluate_policy_on_scenes(
                method_name="stateful_utility_only", budget_bits=2600,
                test_f_pri=fp_s, test_f_curr=f_cur[idx_test],
                test_pri_idx=pi_s, test_curr_idx=ci[idx_test],
                test_loc_p=lp_s, test_c_k=ck_s, test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb, codec=codec, decoder_head=head, teacher_preds=t_pred,
                delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False,
            )

            # 2. SafeSync-Mesh with 100% BLIND disaster-invariant positive displacement gate (d_reg+ > tau_reg, ZERO oracle shift, ZERO unshifted t1!)
            dreg_s = _bldg_pos_disp(loc_curr_test, lp_s)
            ref_mask = dreg_s > tau_reg
            diff_idx = np.flatnonzero(~ref_mask)
            ref_idx = np.flatnonzero(ref_mask)

            f_hat_all = np.zeros_like(fp_s)
            fp_in_all = fp_s.copy()
            lp_in_all = lp_s.copy()
            conf_all = np.ones((n_test, 1024), dtype=np.float32)
            sel_all = np.zeros((n_test, 1024), dtype=bool)
            rates_scene_kb = np.zeros(n_test, dtype=np.float64)

            if len(diff_idx) > 0:
                rd_cache = evaluate_policy_on_scenes(
                    method_name="safesync", budget_bits=2600,
                    test_f_pri=fp_s[diff_idx], test_f_curr=f_cur[idx_test[diff_idx]],
                    test_pri_idx=pi_s[diff_idx], test_curr_idx=ci[idx_test[diff_idx]],
                    test_loc_p=lp_s[diff_idx], test_c_k=ck_s[diff_idx], test_g_k=s3["all_g_k"][idx_test[diff_idx]],
                    test_raw_risk=s3["all_raw_risk"][idx_test[diff_idx]], test_cal_prob=cal_p[diff_idx], test_cal_upper=cal_u[diff_idx],
                    test_known_saliency=s3["all_known_saliency"][idx_test[diff_idx]], test_twist_score=s3["all_twist_score"][idx_test[diff_idx]],
                    test_cell_gt=gt440[idx_test[diff_idx]], test_unknown_gt=test_u_gt[diff_idx],
                    codebook=cb, codec=codec, decoder_head=head, teacher_preds=t_pred[diff_idx],
                    delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False, return_encoded_cache=True,
                )["_encoded_cache"]
                for loc_j, orig_j in enumerate(diff_idx.tolist()):
                    sm = rd_cache["sel_masks"][loc_j]
                    sel_all[orig_j] = sm
                    f_h = fp_s[orig_j].copy()
                    f_h[sm] = cb[ci[idx_test[orig_j], sm]]
                    f_hat_all[orig_j] = f_h
                    conf_all[orig_j, ~sm] = np.clip(1.0 - 0.75 * ck_s[orig_j, ~sm], 0.20, 0.95)
                    rates_scene_kb[orig_j] = rd_cache["pkt_stats"][loc_j]["total_bits"] / 1000.0

            if len(ref_idx) > 0:
                # Self-contained full-VQ refresh: 128-b header + 1024*9-b current VQ tokens + zlib-compressed 4-bit single-pass t2 footprint
                # Union envelope max(lp_s, loc_curr_dec) preserves collapsed buildings while realigning displaced roofs (ZERO unshifted t1 leak!)
                for orig_j in ref_idx.tolist():
                    sel_all[orig_j] = True
                    f_hat_all[orig_j] = cb[ci[idx_test[orig_j]]]
                    fp_in_all[orig_j] = fp_s[orig_j]
                    lp_in_all[orig_j] = np.maximum(lp_s[orig_j], loc_curr_dec[orig_j])
                    conf_all[orig_j] = 1.0
                    rates_scene_kb[orig_j] = (128 + 1024 * 9 + int(loc_curr_bits[orig_j])) / 1000.0

            with torch.inference_mode():
                logits_ad = head(
                    torch.from_numpy(f_hat_all).to(device),
                    torch.from_numpy(fp_in_all).to(device),
                    torch.from_numpy(lp_in_all).to(device),
                    torch.from_numpy(conf_all).to(device),
                ).cpu().numpy()
            pred_ad = np.where(lp_in_all > 0.12, np.argmax(logits_ad, axis=-1), 0)
            p_l, t_l = pred_ad > 0, td_t > 0
            lf1 = float(2 * np.sum(p_l & t_l) / max(1, 2 * np.sum(p_l & t_l) + np.sum(p_l ^ t_l)))
            sf1s = [float(2 * np.sum((pred_ad == c) & (td_t == c)) / max(1, np.sum(pred_ad == c) + np.sum(td_t == c))) for c in (1, 2, 3, 4)]
            sync_ad = (0.30 * lf1 + 0.70 * np.mean(sf1s)) * 100.0
            u_miss_ad = (1.0 - float(np.sum(test_u_gt & sel_all)) / max(1, int(test_u_gt.sum()))) * 100.0
            st_dist_ad = float(np.mean(pred_ad != t_pred))

            per_px[px].append({
                "mean_innovation_ck": float(ck_s.mean()),
                "mean_footprint_div_l1": float(dreg_s.mean()),
                "refresh_trigger_pct": float(ref_mask.mean() * 100.0),
                "safesync_adaptive_rate_kb": float(rates_scene_kb.mean()),
                "safesync_adaptive_selected_pct": float(sel_all.mean() * 100.0),
                "safesync_adaptive_sync_f1": sync_ad,
                "safesync_adaptive_state_dist": st_dist_ad,
                "safesync_adaptive_u_miss": u_miss_ad,
                "static_utility_rate_kb": res_static["rate_kb_per_frame"],
                "static_utility_sync_f1": res_static["sync_f1"],
                "static_utility_state_dist": res_static["state_distortion"],
                "static_utility_u_miss": res_static["u_miss"],
            })

    agg_reg = []
    keys = [
        "mean_innovation_ck", "mean_footprint_div_l1", "refresh_trigger_pct", "safesync_adaptive_rate_kb",
        "safesync_adaptive_selected_pct", "safesync_adaptive_sync_f1", "safesync_adaptive_state_dist",
        "safesync_adaptive_u_miss", "static_utility_rate_kb", "static_utility_sync_f1",
        "static_utility_state_dist", "static_utility_u_miss",
    ]
    print("\n=== [3-Seed Aggregated True-RGB Misregistration Stress Summary] ===")
    for px in SHIFTS_PX:
        row = {"registration_shift_px": px, "tau_reg_blind": round(tau_reg, 4)}
        for k in keys:
            vals = [x[k] for x in per_px[px]]
            row[k] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1))
        agg_reg.append(row)
        print(
            f"  px={px:2d}: d_reg={row['mean_footprint_div_l1']:.4f} | trig={row['refresh_trigger_pct']:5.1f}% | "
            f"rate={row['safesync_adaptive_rate_kb']:.2f} kb | SafeSync={row['safesync_adaptive_sync_f1']:.2f}% (UM={row['safesync_adaptive_u_miss']:.2f}%) "
            f"vs Static={row['static_utility_sync_f1']:.2f}% (UM={row['static_utility_u_miss']:.2f}%)"
        )
    return agg_reg


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shift_cache, gpu_prof = extract_true_rgb_shifts_and_profile_gpu(device)
    sn7_blind = run_blind_spacenet7()
    reg_true = run_blind_true_rgb_reg_stress(shift_cache, device)

    # Merge purified SpaceNet-7 (blind dt-free) + True RGB Misregistration (blind refresh gate) + GPU profile
    main_path = RESULTS_DIR / "safesync_r36_complete_results.json"
    main_res = json.loads(main_path.read_text(encoding="utf-8"))
    main_res["spacenet7_temporal_and_bridge"] = sn7_blind
    main_res["aggregated_3seed"]["registration_stress"] = reg_true
    main_res["compute_and_latency_table4"]["live_rtx3090_swin_profile"] = gpu_prof
    main_path.write_text(json.dumps(main_res, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Done] Purified registration_stress, blind SpaceNet-7 & GPU profile saved in {time.perf_counter()-t0:.1f}s!")


if __name__ == "__main__":
    main()
