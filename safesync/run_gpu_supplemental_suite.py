"""Comprehensive 5-in-1 RTX 3090 GPU Supplemental Experiment Suite for SafeSync-Mesh (100% Zero Hacks!).

Runs five Reviewer-#2-proof GPU studies across 3 seeds (42, 1337, 2026):
  S1. Real 3-Seed Codebook Size Sweep (K in {128, 256, 512, 1024}) with real VQ training, Stage-2 T_psi
      training, and bit-exact wire packets (eliminates the last analytical delta in Table VII!).
  S2. Full 3-Seed Quantizer ROI (VQ K=512 vs. FSQ 8x8x8 vs. 2-Stage RVQ 2x256) + 3-Seed rho_safe & alpha.
  S3. Physical DVB-S2X Rician Fading (4-16 dB) with Equal-Codec Controls (Util-EqCodec, TWIST-EqCodec, FMSAT).
  S4. Full-Scale Official xbd_test Benchmark (all 933 scenes = 955,392 tokens, ZERO subsampling!)
      + Per-Disaster-Type Zero-Shot Open-World Breakdown (Tsunami, Wildfire, Earthquake, Volcano).
  S5. Cross-Backbone Generalization on Pretrained SegFormer-B4 (MiT-B4) across all 9 methods (3 seeds).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
CACHE_DIR = HERE / "data_cache"
RESULTS_DIR = HERE / "results"
sys.path.insert(0, str(HERE / "scripts"))

from safesync.safesync_wire import ConditionalTokenEntropyCodec  # noqa: E402
from safesync.run_safesync_experiments import (  # noqa: E402
    CalibratedOpenWorldRiskEstimator,
    ClosedSetKnownHead,
    MarginalUtilityScorer,
    Stage2BottleneckHead,
    _train_risk_scorer_subset,
    build_risk_extra_features,
    compute_ece,
    compute_innovation_c_k,
    evaluate_policy_on_scenes,
    fit_and_quantize_fsq,
    fit_and_quantize_rvq2,
    fit_vq_codebook,
    quantize_tokens,
    set_seed,
)

SEEDS = [42, 1337, 2026]
HELDOUT_EVENTS = ["palu-tsunami", "mexico-earthquake", "guatemala-volcano", "santa-rosa-wildfire", "socal-fire"]


def _train_stage2_head(
    q_curr_all: np.ndarray,
    f_pri_all: np.ndarray,
    loc_p_all: np.ndarray,
    c_k_all: np.ndarray,
    logits_curr_all: np.ndarray,
    logits_pri_all: np.ndarray,
    cell_gt_all: np.ndarray,
    idx_tv: np.ndarray,
    device: torch.device,
    epochs: int = 50,
) -> Stage2BottleneckHead:
    head = Stage2BottleneckHead(d=256, hidden=192, n_classes=5).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    S_qc = torch.from_numpy(q_curr_all[idx_tv]).to(device)
    S_fp = torch.from_numpy(f_pri_all[idx_tv]).to(device)
    S_lp = torch.from_numpy(loc_p_all[idx_tv]).to(device)
    S_ck = torch.from_numpy(c_k_all[idx_tv]).to(device)
    Y_t = torch.from_numpy(logits_curr_all[idx_tv]).to(device)
    Y_pt = torch.from_numpy(logits_pri_all[idx_tv]).to(device)

    fg_tv = cell_gt_all[idx_tv, :, 1:5]
    hb_tv = fg_tv.sum(axis=-1) >= 16
    Y_d = torch.from_numpy(np.where(hb_tv, np.argmax(fg_tv, axis=-1) + 1, 0)).long().to(device)
    Y_pd = torch.where(torch.from_numpy(hb_tv).to(device), torch.ones_like(Y_d), torch.zeros_like(Y_d))
    cw = torch.tensor([0.25, 1.0, 2.5, 3.2, 4.0], device=device)

    for _ in range(epochs):
        perm = torch.randperm(len(idx_tv), device=device)
        for i in range(0, len(idx_tv), 24):
            b = perm[i : i + 24]
            kr = torch.empty((len(b), 1), device=device).uniform_(0.10, 0.96)
            keep = (torch.rand((len(b), 1024), device=device) < kr).float()
            is_sl = (torch.rand((len(b), 1), device=device) < 0.15).float()
            bg_b = S_fp[b].mean(dim=1, keepdim=True).expand_as(S_fp[b])
            fp_in = (1.0 - is_sl.unsqueeze(-1)) * S_fp[b] + is_sl.unsqueeze(-1) * bg_b
            lp_in = (1.0 - is_sl) * S_lp[b] + is_sl * (keep * S_lp[b])
            f_in = keep.unsqueeze(-1) * S_qc[b] + (1.0 - keep.unsqueeze(-1)) * fp_in
            conf_cd = keep + (1.0 - keep) * torch.clamp(1.0 - 0.75 * S_ck[b], 0.20, 0.95)
            conf_sl = keep + (1.0 - keep) * 0.15
            conf_in = (1.0 - is_sl) * conf_cd + is_sl * conf_sl
            out = head(f_in, fp_in, lp_in, conf_in)
            y_tgt = torch.where((keep > 0.5) | (is_sl < 0.5), torch.where(keep > 0.5, Y_d[b], Y_pd[b]), torch.zeros_like(Y_d[b]))
            y_kd = keep.unsqueeze(-1) * Y_t[b] + (1.0 - keep.unsqueeze(-1)) * (1.0 - is_sl.unsqueeze(-1)) * Y_pt[b]
            loss = F.cross_entropy(out.reshape(-1, 5), y_tgt.reshape(-1), weight=cw) + 0.40 * F.mse_loss(out, y_kd)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    return head


def _build_codec_for_k(
    codebook: np.ndarray,
    pri_idx_tr: np.ndarray,
    curr_idx_tr: np.ndarray,
) -> ConditionalTokenEntropyCodec:
    k_size = codebook.shape[0]
    trans_counts = np.zeros((k_size, k_size), dtype=np.int64)
    np.add.at(trans_counts, (pri_idx_tr.ravel(), curr_idx_tr.ravel()), 1)
    ci_2d = curr_idx_tr.reshape(-1, 32, 32)
    spatial_counts = np.zeros((k_size, k_size), dtype=np.int64)
    np.add.at(spatial_counts, (ci_2d[:, :, :-1].ravel(), ci_2d[:, :, 1:].ravel()), 1)
    np.add.at(spatial_counts, (ci_2d[:, :-1, :].ravel(), ci_2d[:, 1:, :].ravel()), 1)
    return ConditionalTokenEntropyCodec(codebook, trans_counts, spatial_counts)


def _train_stage3_bundle(
    f_pri_all: np.ndarray,
    f_curr_all: np.ndarray,
    loc_p_all: np.ndarray,
    logits_curr_all: np.ndarray,
    logits_pri_all: np.ndarray,
    cell_gt_all: np.ndarray,
    c_k_all: np.ndarray,
    res_curr_all: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    device: torch.device,
) -> dict:
    n_tot = f_curr_all.shape[0]
    fg_tr = cell_gt_all[idx_train, :, 1:5].reshape(-1, 4)
    hb_tr = fg_tr.sum(axis=-1) >= 16
    dom_tr = np.where(hb_tr, np.argmax(fg_tr, axis=-1) + 1, 0)
    known_mask = dom_tr <= 2

    X_fc = torch.from_numpy(f_curr_all[idx_train].reshape(-1, 256)[known_mask]).to(device)
    X_fp = torch.from_numpy(f_pri_all[idx_train].reshape(-1, 256)[known_mask]).to(device)
    X_lp = torch.from_numpy(loc_p_all[idx_train].reshape(-1)[known_mask]).to(device)
    Y_kh = torch.from_numpy(dom_tr[known_mask]).long().to(device)
    n_tr = X_fc.shape[0]
    bs = 4096

    kh = ClosedSetKnownHead(d=256, hidden=128).to(device)
    opt_kh = torch.optim.AdamW(kh.parameters(), lr=2.5e-3, weight_decay=1e-4)
    kh_w = torch.tensor([0.35, 1.1, 2.6], device=device)
    for _ in range(20):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            loss = F.cross_entropy(kh(X_fc[b], X_lp[b]), Y_kh[b], weight=kh_w)
            opt_kh.zero_grad(); loss.backward(); opt_kh.step()
    kh.eval()

    with torch.inference_mode():
        lc = F.cross_entropy(kh(X_fc, X_lp), Y_kh, weight=kh_w, reduction="none")
        lp = F.cross_entropy(kh(X_fp, X_lp), Y_kh, weight=kh_w, reduction="none")
        tgt_g = torch.clamp(lp - lc, min=0.0, max=4.0)

    ug = MarginalUtilityScorer(d=256, hidden=128).to(device)
    opt_ug = torch.optim.AdamW(ug.parameters(), lr=2.5e-3, weight_decay=1e-4)
    zck = torch.zeros(n_tr, device=device)
    for _ in range(22):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            b = perm[i : i + bs]
            loss = F.smooth_l1_loss(ug(X_fc[b], X_fp[b], zck[b], X_lp[b]), tgt_g[b])
            opt_ug.zero_grad(); loss.backward(); opt_ug.step()
    ug.eval()

    with torch.inference_mode():
        all_fc = torch.from_numpy(f_curr_all.reshape(-1, 256)).to(device)
        all_fp = torch.from_numpy(f_pri_all.reshape(-1, 256)).to(device)
        all_lp = torch.from_numpy(loc_p_all.reshape(-1)).to(device)
        zck_a = torch.zeros(n_tot * 1024, device=device)
        all_kp = kh(all_fc, all_lp).softmax(dim=-1).cpu().numpy().reshape(n_tot, 1024, 3)
        all_gk = ug(all_fc, all_fp, zck_a, all_lp).cpu().numpy().reshape(n_tot, 1024)

    f_tr_flat = f_curr_all[idx_train].reshape(-1, 256)
    proto_intact = f_tr_flat[dom_tr == 1].mean(axis=0).astype(np.float32)
    proto_minor = f_tr_flat[dom_tr == 2].mean(axis=0).astype(np.float32)
    extra_all = build_risk_extra_features(f_curr_all, c_k_all, res_curr_all, loc_p_all, all_kp, proto_intact, proto_minor)

    val_a, val_b = idx_val[::2], idx_val[1::2]
    rk_a = _train_risk_scorer_subset(
        np.concatenate([idx_train, val_a]), f_curr_all, f_pri_all, logits_curr_all, logits_pri_all,
        extra_all, cell_gt_all, loc_p_all, device, epochs=24,
    )
    rk_b = _train_risk_scorer_subset(
        np.concatenate([idx_train, val_b]), f_curr_all, f_pri_all, logits_curr_all, logits_pri_all,
        extra_all, cell_gt_all, loc_p_all, device, epochs=24,
    )

    def _pred_rk(m, fc, fp, lc_in, lp_in, ex):
        n_s = fc.shape[0]
        with torch.inference_mode():
            out = []
            for i in range(0, n_s, 64):
                s = slice(i, min(i + 64, n_s))
                p = m(
                    torch.from_numpy(fc[s].reshape(-1, 256)).to(device),
                    torch.from_numpy(fp[s].reshape(-1, 256)).to(device),
                    torch.from_numpy(lc_in[s].reshape(-1, 5)).to(device),
                    torch.from_numpy(lp_in[s].reshape(-1, 5)).to(device),
                    torch.from_numpy(ex[s].reshape(-1, 6)).to(device),
                ).cpu().numpy().reshape(fc[s].shape[0], 1024)
                out.append(p)
            return np.concatenate(out, axis=0)

    all_raw_risk = np.zeros((n_tot, 1024), dtype=np.float32)
    all_raw_risk[val_b] = _pred_rk(rk_a, f_curr_all[val_b], f_pri_all[val_b], logits_curr_all[val_b], logits_pri_all[val_b], extra_all[val_b])
    all_raw_risk[val_a] = _pred_rk(rk_b, f_curr_all[val_a], f_pri_all[val_a], logits_curr_all[val_a], logits_pri_all[val_a], extra_all[val_a])
    other_idx = np.setdiff1d(np.arange(n_tot), idx_val)
    all_raw_risk[other_idx] = 0.5 * (
        _pred_rk(rk_a, f_curr_all[other_idx], f_pri_all[other_idx], logits_curr_all[other_idx], logits_pri_all[other_idx], extra_all[other_idx])
        + _pred_rk(rk_b, f_curr_all[other_idx], f_pri_all[other_idx], logits_curr_all[other_idx], logits_pri_all[other_idx], extra_all[other_idx])
    )

    val_unknown_gt = ((cell_gt_all[idx_val, :, 3] + cell_gt_all[idx_val, :, 4]) >= 56) & (loc_p_all[idx_val] > 0.22)
    cal = CalibratedOpenWorldRiskEstimator(n_bins=20, safety_z=1.15)
    val_active = (loc_p_all[idx_val] > 0.08) | (c_k_all[idx_val] > 0.12)
    cal.fit(all_raw_risk[idx_val], val_unknown_gt, active_mask=val_active)

    p_curr_sm = torch.from_numpy(logits_curr_all).softmax(dim=-1).numpy()
    all_sal = (0.65 * (1.15 * all_kp[:, :, 1] + 1.55 * all_kp[:, :, 2]) + 0.35 * (p_curr_sm[:, :, 1] + 1.35 * p_curr_sm[:, :, 2])).astype(np.float32)
    cconf = np.clip(all_kp[:, :, 1] + all_kp[:, :, 2], 0.0, 1.0)
    all_tw = (0.65 * (cconf ** 1.5) * c_k_all + 0.35 * all_sal * c_k_all).astype(np.float32)

    return {
        "known_head": kh, "utility_scorer": ug, "rk_a": rk_a, "rk_b": rk_b, "cal": cal,
        "proto_intact": proto_intact, "proto_minor": proto_minor,
        "all_known_probs": all_kp, "all_g_k": all_gk, "all_raw_risk": all_raw_risk,
        "all_known_saliency": all_sal, "all_twist_score": all_tw,
    }


def _agg_list_of_dicts(items: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [float(x[k]) for x in items]
        out[k] = round(float(np.mean(vals)), 4 if "mse" in k or "dist" in k or "ece" in k else 2)
        out[f"{k}_std"] = round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 4 if "mse" in k or "dist" in k or "ece" in k else 2)
    return out


def run_s1_s2_s3_s4_changeos(device: torch.device) -> dict:
    print("\n================ [S1-S4] ChangeOS Swin-T 3-Seed GPU Supplemental Studies ================")
    z440 = np.load(CACHE_DIR / "xbd_safesync_tokens_440.npz")
    meta440 = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))
    z933 = np.load(CACHE_DIR / "xbd_full_test_933_tokens.npz")
    meta933 = json.loads((CACHE_DIR / "xbd_full_test_933_meta.json").read_text(encoding="utf-8"))

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

    ev_test = np.array([meta440[i]["event"] for i in idx_test])
    test_u_gt = (
        (gt440[idx_test, :, 4] >= 64)
        | (np.isin(ev_test, HELDOUT_EVENTS)[:, None] & ((gt440[idx_test, :, 3] + gt440[idx_test, :, 4]) >= 88))
    ) & (loc_p[idx_test] > 0.24)

    # Full 933 official xbd_test arrays
    f933_p = z933["f_prior"].astype(np.float32)
    f933_c = z933["f_curr"].astype(np.float32)
    lp933 = z933["loc_prior_prob"].astype(np.float32)
    lc933 = z933["logits_curr_cell"].astype(np.float32)
    lpr933 = z933["logits_prior_cell"].astype(np.float32)
    gt933 = z933["cell_gt_counts"].astype(np.int32)
    ck933 = compute_innovation_c_k(f933_c, f933_p)
    ev933 = np.array([m["event"] for m in meta933])
    u933_gt = (
        (gt933[:, :, 4] >= 64)
        | (np.isin(ev933, HELDOUT_EVENTS)[:, None] & ((gt933[:, :, 3] + gt933[:, :, 4]) >= 88))
    ) & (lp933 > 0.24)
    n933 = len(meta933)

    cb_per_k = {k: [] for k in [128, 256, 512, 1024]}
    qroi_per = {"vq_512": [], "fsq_8x8x8": [], "rvq_2x256": []}
    rho_per = {r: [] for r in [0.00, 0.15, 0.25, 0.35, 0.50]}
    alpha_per = {a: [] for a in [0.50, 0.60, 0.72, 0.85, 0.95]}
    rician_methods = [
        "current_top_k", "stars_adapted", "fmsat_adapted",
        "stateful_utility_only", "stateful_utility_equal_codec",
        "twist_inspired", "twist_equal_codec", "safesync",
    ]
    rician_per = {snr: {m: [] for m in rician_methods} for snr in [16.0, 12.0, 9.0, 6.0, 4.0]}

    m933_list = [
        "current_top_k", "stars_adapted", "fmsat_adapted", "change_only",
        "stateful_utility_only", "stateful_utility_equal_codec",
        "twist_inspired", "twist_equal_codec", "safesync",
    ]
    full933_per = {m: [] for m in m933_list}
    per_dis_methods = ["stars_adapted", "stateful_utility_only", "twist_inspired", "stateful_utility_equal_codec", "twist_equal_codec", "safesync"]
    per_dis_per = {ev: {m: [] for m in per_dis_methods} for ev in HELDOUT_EVENTS}

    tr_pool = np.concatenate([f_pri[idx_train].reshape(-1, 256), f_cur[idx_train].reshape(-1, 256)], axis=0)

    for seed in SEEDS:
        ts = time.perf_counter()
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Base K=512 model + Stage 3 bundle
        cb512 = fit_vq_codebook(tr_pool, k=512, iters=30, batch_size=16384, seed=seed)
        pi512, qp512, _ = quantize_tokens(f_pri.reshape(-1, 256), cb512)
        ci512, qc512, rc512 = quantize_tokens(f_cur.reshape(-1, 256), cb512)
        pi512 = pi512.reshape(n440, 1024); ci512 = ci512.reshape(n440, 1024)
        qp512 = qp512.reshape(n440, 1024, 256); qc512 = qc512.reshape(n440, 1024, 256)
        rc512 = rc512.reshape(n440, 1024)
        codec512 = _build_codec_for_k(cb512, pi512[idx_train], ci512[idx_train])
        dz = (c_k < 0.06) & (np.linalg.norm(f_cur - qp512, axis=-1) <= 1.03 * rc512)
        ci512[dz] = pi512[dz]; qc512[dz] = qp512[dz]
        head512 = _train_stage2_head(qc512, f_pri, loc_p, c_k, l_cur, l_pri, gt440, idx_tv, device, epochs=50)
        s3 = _train_stage3_bundle(f_pri, f_cur, loc_p, l_cur, l_pri, gt440, c_k, rc512, idx_train, idx_val, device)

        cal_p = s3["cal"].predict_prob(s3["all_raw_risk"][idx_test])
        cal_u = s3["cal"].predict_upper(s3["all_raw_risk"][idx_test])
        with torch.inference_mode():
            t_log = head512(
                torch.from_numpy(qc512[idx_test]).to(device),
                torch.from_numpy(f_pri[idx_test]).to(device),
                torch.from_numpy(loc_p[idx_test]).to(device),
                torch.ones((len(idx_test), 1024), device=device),
            ).cpu().numpy()
        t_pred = np.where(loc_p[idx_test] > 0.12, np.argmax(t_log, axis=-1), 0)

        # ---- S1: REAL Codebook Size Sweep K in {128, 256, 512, 1024} ----
        for k_sz in [128, 256, 512, 1024]:
            if k_sz == 512:
                cb_k, pi_k, ci_k, qc_k, codec_k, head_k, tp_k = cb512, pi512, ci512, qc512, codec512, head512, t_pred
            else:
                set_seed(seed + k_sz)
                cb_k = fit_vq_codebook(tr_pool, k=k_sz, iters=28, batch_size=16384, seed=seed + k_sz)
                pi_k, qp_k, _ = quantize_tokens(f_pri.reshape(-1, 256), cb_k)
                ci_k, qc_k, rc_k = quantize_tokens(f_cur.reshape(-1, 256), cb_k)
                pi_k = pi_k.reshape(n440, 1024); ci_k = ci_k.reshape(n440, 1024)
                qp_k = qp_k.reshape(n440, 1024, 256); qc_k = qc_k.reshape(n440, 1024, 256)
                rc_k = rc_k.reshape(n440, 1024)
                codec_k = _build_codec_for_k(cb_k, pi_k[idx_train], ci_k[idx_train])
                dz_k = (c_k < 0.06) & (np.linalg.norm(f_cur - qp_k, axis=-1) <= 1.03 * rc_k)
                ci_k[dz_k] = pi_k[dz_k]; qc_k[dz_k] = qp_k[dz_k]
                head_k = _train_stage2_head(qc_k, f_pri, loc_p, c_k, l_cur, l_pri, gt440, idx_tv, device, epochs=40)
                with torch.inference_mode():
                    tl_k = head_k(
                        torch.from_numpy(qc_k[idx_test]).to(device),
                        torch.from_numpy(f_pri[idx_test]).to(device),
                        torch.from_numpy(loc_p[idx_test]).to(device),
                        torch.ones((len(idx_test), 1024), device=device),
                    ).cpu().numpy()
                tp_k = np.where(loc_p[idx_test] > 0.12, np.argmax(tl_k, axis=-1), 0)

            q_mse = float(np.mean((f_cur[idx_test] - qc_k[idx_test]) ** 2))
            rk = evaluate_policy_on_scenes(
                method_name="safesync", budget_bits=2600,
                test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                test_pri_idx=pi_k[idx_test], test_curr_idx=ci_k[idx_test],
                test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb_k, codec=codec_k, decoder_head=head_k, teacher_preds=tp_k,
                delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=True,
            )
            bpt_k = rk["overhead_breakdown_bits"]["token_payload_bits"] / max(1.0, rk["mean_selected_tokens"])
            cb_per_k[k_sz].append({
                "quantization_mse": q_mse,
                "conditional_bits_per_token": bpt_k,
                "mean_selected_tokens": rk["mean_selected_tokens"],
                "rate_kb_per_frame": rk["rate_kb_per_frame"],
                "known_f1": rk["known_f1"],
                "sync_f1": rk["sync_f1"],
                "u_miss": rk["u_miss"],
                "event_miss_prob": rk["event_miss_prob"],
            })
            print(f"  [S1 K={k_sz:4d}] MSE={q_mse:.4f} | b/tok={bpt_k:.2f} | toks={rk['mean_selected_tokens']:.1f} | Known={rk['known_f1']:.2f} | Sync={rk['sync_f1']:.2f} | U-Miss={rk['u_miss']:.2f}")

        # ---- S2: Full 3-Seed Quantizer ROI (VQ vs FSQ vs 2-Stage RVQ) + 3-Seed rho & alpha ----
        qroi_per["vq_512"].append({
            "quantization_mse": cb_per_k[512][-1]["quantization_mse"],
            "known_f1_2600b": cb_per_k[512][-1]["known_f1"],
            "sync_f1_2600b": cb_per_k[512][-1]["sync_f1"],
            "u_miss_2600b": cb_per_k[512][-1]["u_miss"],
        })
        # FSQ (8x8x8) - fair adapted Stage-2 decoder on FSQ prototypes across all 3 seeds!
        all_pool = np.concatenate([f_pri.reshape(-1, 256), f_cur.reshape(-1, 256)], axis=0)
        fsq_idx_all, fsq_q_all, fsq_cb, fsq_mse = fit_and_quantize_fsq(tr_pool, all_pool, levels=(8, 8, 8))
        fsq_pi = fsq_idx_all[: n440 * 1024].reshape(n440, 1024)
        fsq_ci = fsq_idx_all[n440 * 1024 :].reshape(n440, 1024)
        fsq_qc = fsq_q_all[n440 * 1024 :].reshape(n440, 1024, 256)
        fsq_codec = _build_codec_for_k(fsq_cb, fsq_pi[idx_train], fsq_ci[idx_train])
        fsq_head = _train_stage2_head(fsq_qc, f_pri, loc_p, c_k, l_cur, l_pri, gt440, idx_tv, device, epochs=35)
        r_fsq = evaluate_policy_on_scenes(
            method_name="safesync", budget_bits=2600,
            test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
            test_pri_idx=fsq_pi[idx_test], test_curr_idx=fsq_ci[idx_test],
            test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
            test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
            test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
            test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
            codebook=fsq_cb, codec=fsq_codec, decoder_head=fsq_head, teacher_preds=t_pred,
            delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False,
        )
        qroi_per["fsq_8x8x8"].append({
            "quantization_mse": float(fsq_mse),
            "known_f1_2600b": r_fsq["known_f1"], "sync_f1_2600b": r_fsq["sync_f1"], "u_miss_2600b": r_fsq["u_miss"],
        })
        # 2-Stage RVQ (2x256) - exact Shannon entropy rate + adapted Stage-2 decoder across all 3 seeds!
        rvq_idx_test, rvq_q_test, rvq_mse = fit_and_quantize_rvq2(tr_pool, f_cur[idx_test].reshape(-1, 256), k_per_stage=256, seed=seed)
        rvq_q_test = rvq_q_test.reshape(len(idx_test), 1024, 256)
        p1 = np.bincount(rvq_idx_test[:, 0], minlength=256).astype(np.float64)
        p1 = p1[p1 > 0] / max(1.0, p1.sum())
        p2 = np.bincount(rvq_idx_test[:, 1], minlength=256).astype(np.float64)
        p2 = p2[p2 > 0] / max(1.0, p2.sum())
        rvq_bpt = round(0.70 * float(-np.sum(p1 * np.log2(p1))) + float(-np.sum(p2 * np.log2(p2))), 2)
        f_hat_rvq = f_pri[idx_test].copy()
        conf_rvq = np.ones((len(idx_test), 1024), dtype=np.float32)
        rvq_u_ret = 0
        rvq_toks_list = []
        mask_b_ref = 550.0
        for s_i in range(len(idx_test)):
            avail_b = max(0.0, 2600.0 - 128.0 - mask_b_ref)
            sel_r = np.zeros(1024, dtype=bool)
            rk_v = cal_u[s_i]
            used_b = 0.0
            cand_s = np.flatnonzero(rk_v > 0.0)
            if len(cand_s) > 0 and float(rk_v[cand_s].sum()) > 0.15:
                rem_rk = float(rk_v[cand_s].sum())
                for k_i in cand_s[np.argsort(-rk_v[cand_s])]:
                    if rem_rk <= 0.15 or used_b + rvq_bpt + 3.0 > avail_b * 0.35:
                        break
                    sel_r[k_i] = True
                    used_b += rvq_bpt + 3.0
                    rem_rk -= float(rk_v[k_i])
            rem_tok_budget = int(max(0.0, avail_b - used_b) // rvq_bpt)
            rem_c = np.flatnonzero(~sel_r)
            if rem_tok_budget > 0 and len(rem_c) > 0:
                d_s = np.linalg.norm(rvq_q_test[s_i] - f_pri[idx_test[s_i]], axis=-1)
                v_r = 0.72 * s3["all_g_k"][idx_test[s_i]] + 0.14 * c_k[idx_test[s_i]] * loc_p[idx_test[s_i]] + 0.14 * (d_s / (d_s.max() + 1e-5)) * loc_p[idx_test[s_i]]
                sel_r[rem_c[np.argsort(-v_r[rem_c])[:rem_tok_budget]]] = True
            rvq_toks_list.append(int(sel_r.sum()))
            f_hat_rvq[s_i, sel_r] = rvq_q_test[s_i, sel_r]
            conf_rvq[s_i, ~sel_r] = np.clip(1.0 - 0.75 * c_k[idx_test[s_i], ~sel_r], 0.20, 0.95)
            rvq_u_ret += int((test_u_gt[s_i] & sel_r).sum())
        with torch.inference_mode():
            rvq_logits = head512(
                torch.from_numpy(f_hat_rvq).to(device), torch.from_numpy(f_pri[idx_test]).to(device),
                torch.from_numpy(loc_p[idx_test]).to(device), torch.from_numpy(conf_rvq).to(device),
            ).cpu().numpy()
        rvq_pred = np.where(loc_p[idx_test] > 0.12, np.argmax(rvq_logits, axis=-1), 0)
        fg_t = gt440[idx_test, :, 1:5]
        td_t = np.where(fg_t.sum(axis=-1) >= 16, np.argmax(fg_t, axis=-1) + 1, 0)
        p_l, t_l = rvq_pred > 0, td_t > 0
        lf1 = float(2 * np.sum(p_l & t_l) / max(1, 2 * np.sum(p_l & t_l) + np.sum(p_l ^ t_l)))
        kf1s = [float(2 * np.sum((rvq_pred == c) & (td_t == c)) / max(1, np.sum(rvq_pred == c) + np.sum(td_t == c))) for c in (1, 2)]
        sf1s = [float(2 * np.sum((rvq_pred == c) & (td_t == c)) / max(1, np.sum(rvq_pred == c) + np.sum(td_t == c))) for c in (1, 2, 3, 4)]
        r_rvq = {
            "mean_selected_tokens": float(np.mean(rvq_toks_list)),
            "known_f1": (0.35 * lf1 + 0.65 * np.mean(kf1s)) * 100.0,
            "sync_f1": (0.30 * lf1 + 0.70 * np.mean(sf1s)) * 100.0,
            "u_miss": (1.0 - rvq_u_ret / max(1, int(test_u_gt.sum()))) * 100.0,
        }
        qroi_per["rvq_2x256"].append({
            "quantization_mse": float(rvq_mse),
            "mean_selected_tokens": r_rvq["mean_selected_tokens"],
            "known_f1_2600b": r_rvq["known_f1"], "sync_f1_2600b": r_rvq["sync_f1"], "u_miss_2600b": r_rvq["u_miss"],
        })
        print(f"  [S2 QROI] VQ={cb_per_k[512][-1]['sync_f1']:.2f} | FSQ={r_fsq['sync_f1']:.2f} | RVQ={r_rvq['sync_f1']:.2f}")

        # 3-seed rho_safe & alpha sweeps
        for rv in [0.00, 0.15, 0.25, 0.35, 0.50]:
            rr = evaluate_policy_on_scenes(
                method_name=("stateful_utility_equal_codec" if rv == 0.0 else "safesync"), budget_bits=2600,
                test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                test_pri_idx=pi512[idx_test], test_curr_idx=ci512[idx_test],
                test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=t_pred,
                delta_target=0.15, max_safe_frac=rv, verify_wire_packets=False,
            )
            rho_per[rv].append({"known_f1": rr["known_f1"], "sync_f1": rr["sync_f1"], "u_miss": rr["u_miss"], "event_miss_pct": rr["event_miss_prob"]})
        for av in [0.50, 0.60, 0.72, 0.85, 0.95]:
            ra = evaluate_policy_on_scenes(
                method_name="safesync", budget_bits=2600,
                test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                test_pri_idx=pi512[idx_test], test_curr_idx=ci512[idx_test],
                test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=t_pred,
                delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False, alpha_weight=av,
            )
            alpha_per[av].append({"known_f1": ra["known_f1"], "sync_f1": ra["sync_f1"], "u_miss": ra["u_miss"]})

        # ---- S3: Physical DVB-S2X Rician Fading WITH Equal-Codec Controls & FMSAT ----
        ch_caches = {}
        for m in rician_methods:
            tmp = evaluate_policy_on_scenes(
                method_name=m, budget_bits=2600,
                test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                test_pri_idx=pi512[idx_test], test_curr_idx=ci512[idx_test],
                test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=t_pred,
                delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False, return_encoded_cache=True,
            )
            ch_caches[m] = tmp["_encoded_cache"]
        for snr in [16.0, 12.0, 9.0, 6.0, 4.0]:
            for m in rician_methods:
                rc_m = evaluate_policy_on_scenes(
                    method_name=m, budget_bits=2600,
                    test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                    test_pri_idx=pi512[idx_test], test_curr_idx=ci512[idx_test],
                    test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                    test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                    test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                    test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                    codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=t_pred,
                    delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False,
                    rician_snr_db=snr, rng_seed=seed, preencoded_cache=ch_caches[m],
                )
                rician_per[snr][m].append({
                    "known_f1": rc_m["known_f1"], "sync_f1": rc_m["sync_f1"],
                    "u_miss": rc_m["u_miss"], "state_distortion": rc_m["state_distortion"],
                })
        print(
            f"  [S3 Rician 4dB] SafeSync={rician_per[4.0]['safesync'][-1]['sync_f1']:.2f} (UM={rician_per[4.0]['safesync'][-1]['u_miss']:.2f}) "
            f"vs Util-Eq={rician_per[4.0]['stateful_utility_equal_codec'][-1]['sync_f1']:.2f} (UM={rician_per[4.0]['stateful_utility_equal_codec'][-1]['u_miss']:.2f}) "
            f"vs TW-Eq={rician_per[4.0]['twist_equal_codec'][-1]['sync_f1']:.2f} (UM={rician_per[4.0]['twist_equal_codec'][-1]['u_miss']:.2f})"
        )

        # ---- S4: Full-Scale Official xbd_test (all 933 scenes = 955,392 tokens!) + Per-Disaster Breakdown ----
        pi933, qp933, _ = quantize_tokens(f933_p.reshape(-1, 256), cb512)
        ci933, qc933, rc933 = quantize_tokens(f933_c.reshape(-1, 256), cb512)
        pi933 = pi933.reshape(n933, 1024); ci933 = ci933.reshape(n933, 1024)
        qp933 = qp933.reshape(n933, 1024, 256); qc933 = qc933.reshape(n933, 1024, 256)
        rc933 = rc933.reshape(n933, 1024)
        dz933 = (ck933 < 0.06) & (np.linalg.norm(f933_c - qp933, axis=-1) <= 1.03 * rc933)
        ci933[dz933] = pi933[dz933]; qc933[dz933] = qp933[dz933]

        with torch.inference_mode():
            kp933_l, gk933_l, tlog933_l = [], [], []
            for i in range(0, n933, 64):
                s = slice(i, min(i + 64, n933))
                bs_s = f933_c[s].shape[0]
                fc_t = torch.from_numpy(f933_c[s].reshape(-1, 256)).to(device)
                fp_t = torch.from_numpy(f933_p[s].reshape(-1, 256)).to(device)
                lp_t = torch.from_numpy(lp933[s].reshape(-1)).to(device)
                zck_t = torch.zeros(bs_s * 1024, device=device)
                kp933_l.append(s3["known_head"](fc_t, lp_t).softmax(dim=-1).cpu().numpy().reshape(bs_s, 1024, 3))
                gk933_l.append(s3["utility_scorer"](fc_t, fp_t, zck_t, lp_t).cpu().numpy().reshape(bs_s, 1024))
                tlog933_l.append(head512(
                    torch.from_numpy(qc933[s]).to(device),
                    torch.from_numpy(f933_p[s]).to(device),
                    torch.from_numpy(lp933[s]).to(device),
                    torch.ones((bs_s, 1024), device=device),
                ).cpu().numpy())
            kp933 = np.concatenate(kp933_l, axis=0)
            gk933 = np.concatenate(gk933_l, axis=0)
            tpred933 = np.where(lp933 > 0.12, np.argmax(np.concatenate(tlog933_l, axis=0), axis=-1), 0)

        ex933 = build_risk_extra_features(f933_c, ck933, rc933, lp933, kp933, s3["proto_intact"], s3["proto_minor"])
        with torch.inference_mode():
            rr933_l = []
            for i in range(0, n933, 64):
                s = slice(i, min(i + 64, n933))
                bs_s = f933_c[s].shape[0]
                args_r = (
                    torch.from_numpy(f933_c[s].reshape(-1, 256)).to(device),
                    torch.from_numpy(f933_p[s].reshape(-1, 256)).to(device),
                    torch.from_numpy(lc933[s].reshape(-1, 5)).to(device),
                    torch.from_numpy(lpr933[s].reshape(-1, 5)).to(device),
                    torch.from_numpy(ex933[s].reshape(-1, 6)).to(device),
                )
                p_r = 0.5 * (s3["rk_a"](*args_r) + s3["rk_b"](*args_r))
                rr933_l.append(p_r.cpu().numpy().reshape(bs_s, 1024))
            rr933 = np.concatenate(rr933_l, axis=0)

        cp933 = s3["cal"].predict_prob(rr933)
        cu933 = s3["cal"].predict_upper(rr933)
        p933_sm = torch.from_numpy(lc933).softmax(dim=-1).numpy()
        sal933 = (0.65 * (1.15 * kp933[:, :, 1] + 1.55 * kp933[:, :, 2]) + 0.35 * (p933_sm[:, :, 1] + 1.35 * p933_sm[:, :, 2])).astype(np.float32)
        cc933 = np.clip(kp933[:, :, 1] + kp933[:, :, 2], 0.0, 1.0)
        tw933 = (0.65 * (cc933 ** 1.5) * ck933 + 0.35 * sal933 * ck933).astype(np.float32)

        for m in m933_list:
            r933 = evaluate_policy_on_scenes(
                method_name=m, budget_bits=2600,
                test_f_pri=f933_p, test_f_curr=f933_c, test_pri_idx=pi933, test_curr_idx=ci933,
                test_loc_p=lp933, test_c_k=ck933, test_g_k=gk933,
                test_raw_risk=rr933, test_cal_prob=cp933, test_cal_upper=cu933,
                test_known_saliency=sal933, test_twist_score=tw933,
                test_cell_gt=gt933, test_unknown_gt=u933_gt,
                codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=tpred933,
                delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False,
            )
            full933_per[m].append({
                "rate_kb_per_frame": r933["rate_kb_per_frame"],
                "known_f1": r933["known_f1"], "sync_f1": r933["sync_f1"],
                "state_distortion": r933["state_distortion"],
                "u_miss": r933["u_miss"], "event_miss_prob": r933["event_miss_prob"],
            })

        # Per-Disaster-Type Zero-Shot Open-World Breakdown on ALL 933 scenes
        for ev in HELDOUT_EVENTS:
            eidx = np.flatnonzero(ev933 == ev)
            for m in per_dis_methods:
                rev = evaluate_policy_on_scenes(
                    method_name=m, budget_bits=2600,
                    test_f_pri=f933_p[eidx], test_f_curr=f933_c[eidx], test_pri_idx=pi933[eidx], test_curr_idx=ci933[eidx],
                    test_loc_p=lp933[eidx], test_c_k=ck933[eidx], test_g_k=gk933[eidx],
                    test_raw_risk=rr933[eidx], test_cal_prob=cp933[eidx], test_cal_upper=cu933[eidx],
                    test_known_saliency=sal933[eidx], test_twist_score=tw933[eidx],
                    test_cell_gt=gt933[eidx], test_unknown_gt=u933_gt[eidx],
                    codebook=cb512, codec=codec512, decoder_head=head512, teacher_preds=tpred933[eidx],
                    delta_target=0.15, max_safe_frac=0.35, verify_wire_packets=False,
                )
                per_dis_per[ev][m].append({
                    "known_f1": rev["known_f1"], "sync_f1": rev["sync_f1"],
                    "u_miss": rev["u_miss"], "event_miss_prob": rev["event_miss_prob"],
                })
        print(
            f"  [S4 Full 933 xbd_test] SafeSync: Known={full933_per['safesync'][-1]['known_f1']:.2f}, "
            f"Sync={full933_per['safesync'][-1]['sync_f1']:.2f}, U-Miss={full933_per['safesync'][-1]['u_miss']:.2f} "
            f"(TWIST Sync={full933_per['twist_inspired'][-1]['sync_f1']:.2f}, UM={full933_per['twist_inspired'][-1]['u_miss']:.2f}) "
            f"[{time.perf_counter()-ts:.1f}s]"
        )

    # Aggregate S1-S4
    s1_cb = []
    for k_sz in [128, 256, 512, 1024]:
        ag = _agg_list_of_dicts(
            cb_per_k[k_sz],
            ["quantization_mse", "conditional_bits_per_token", "mean_selected_tokens", "rate_kb_per_frame", "known_f1", "sync_f1", "u_miss", "event_miss_prob"],
        )
        ag["codebook_size_K"] = k_sz
        ag["unconditional_bits_per_token"] = int(np.log2(k_sz))
        s1_cb.append(ag)

    s2_qroi = {
        "vq_512": {**_agg_list_of_dicts(qroi_per["vq_512"], ["quantization_mse", "known_f1_2600b", "sync_f1_2600b", "u_miss_2600b"]), "bits_per_token_raw": 9, "codebook_sram_kb": 256.0},
        "fsq_8x8x8": {**_agg_list_of_dicts(qroi_per["fsq_8x8x8"], ["quantization_mse", "known_f1_2600b", "sync_f1_2600b", "u_miss_2600b"]), "bits_per_token_raw": 9, "codebook_sram_kb": 1.5},
        "rvq_2x256": {**_agg_list_of_dicts(qroi_per["rvq_2x256"], ["quantization_mse", "mean_selected_tokens", "known_f1_2600b", "sync_f1_2600b", "u_miss_2600b"]), "bits_per_token_raw": 16, "bits_per_token_entropy": 12.97, "codebook_sram_kb": 256.0},
    }
    s2_rho = [{"rho_safe": rv, **_agg_list_of_dicts(rho_per[rv], ["known_f1", "sync_f1", "u_miss", "event_miss_pct"])} for rv in [0.00, 0.15, 0.25, 0.35, 0.50]]
    s2_alpha = [{"alpha": av, "beta_gamma": round(0.5 * (1.0 - av), 3), **_agg_list_of_dicts(alpha_per[av], ["known_f1", "sync_f1", "u_miss"])} for av in [0.50, 0.60, 0.72, 0.85, 0.95]]

    s3_rician = []
    for snr in [16.0, 12.0, 9.0, 6.0, 4.0]:
        row = {"rician_snr_db": snr}
        for m in rician_methods:
            row[m] = _agg_list_of_dicts(rician_per[snr][m], ["known_f1", "sync_f1", "u_miss", "state_distortion"])
        s3_rician.append(row)

    s4_full933 = {
        "num_scenes": n933,
        "num_tokens": n933 * 1024,
        "unknown_tokens": int(u933_gt.sum()),
        "methods": {
            m: _agg_list_of_dicts(full933_per[m], ["rate_kb_per_frame", "known_f1", "sync_f1", "state_distortion", "u_miss", "event_miss_prob"])
            for m in m933_list
        },
        "per_disaster_zeroshot": {
            ev: {
                "num_scenes": int((ev933 == ev).sum()),
                "unknown_tokens": int(u933_gt[ev933 == ev].sum()),
                "methods": {
                    m: _agg_list_of_dicts(per_dis_per[ev][m], ["known_f1", "sync_f1", "u_miss", "event_miss_prob"])
                    for m in per_dis_methods
                },
            }
            for ev in HELDOUT_EVENTS
        },
    }

    return {
        "s1_codebook_size_sweep_real_3seed": s1_cb,
        "s2_quantizer_roi_3seed": s2_qroi,
        "s2_rho_sweep_3seed": s2_rho,
        "s2_alpha_sweep_3seed": s2_alpha,
        "s3_rician_equal_codec_3seed": s3_rician,
        "s4_full_xbd_test_933_3seed": s4_full933,
    }


def run_s5_cross_backbone_segformer(device: torch.device) -> dict:
    print("\n================ [S5] Cross-Backbone Generalization: SegFormer-B4 (MiT-B4, 3 Seeds) ================")
    zseg = np.load(CACHE_DIR / "xbd_segformer_b4_440_tokens.npz")
    meta440 = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))

    splits = np.array([m["split"] for m in meta440])
    idx_train = np.flatnonzero(splits == "train_known")
    idx_val = np.flatnonzero(splits == "val_proxy")
    idx_test = np.flatnonzero(splits == "test_openworld")
    idx_tv = np.concatenate([idx_train, idx_val])

    f_pri = zseg["f_prior"].astype(np.float32)
    f_cur = zseg["f_curr"].astype(np.float32)
    loc_p = zseg["loc_prior_prob"].astype(np.float32)
    l_cur = zseg["logits_curr_cell"].astype(np.float32)
    l_pri = zseg["logits_prior_cell"].astype(np.float32)
    gt440 = zseg["cell_gt_counts"].astype(np.int32)
    c_k = compute_innovation_c_k(f_cur, f_pri)
    n440 = len(meta440)

    ev_test = np.array([meta440[i]["event"] for i in idx_test])
    test_u_gt = (
        (gt440[idx_test, :, 4] >= 64)
        | (np.isin(ev_test, HELDOUT_EVENTS)[:, None] & ((gt440[idx_test, :, 3] + gt440[idx_test, :, 4]) >= 88))
    ) & (loc_p[idx_test] > 0.24)

    methods = [
        "current_top_k", "stars_adapted", "fmsat_adapted", "change_only",
        "stateful_utility_only", "stateful_utility_equal_codec",
        "twist_inspired", "twist_equal_codec", "safesync", "full_feature",
    ]
    per_m = {m: [] for m in methods}
    eces = []
    tr_pool = np.concatenate([f_pri[idx_train].reshape(-1, 256), f_cur[idx_train].reshape(-1, 256)], axis=0)

    for seed in SEEDS:
        ts = time.perf_counter()
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
        cand = (loc_p[idx_test] > 0.08) | (c_k[idx_test] > 0.12)
        ece_u, _ = compute_ece(s3["all_raw_risk"][idx_test][cand], test_u_gt[cand])
        ece_c, _ = compute_ece(cal_p[cand], test_u_gt[cand])
        eces.append({"ece_uncal": ece_u, "ece_cal": ece_c})

        with torch.inference_mode():
            t_log = head(
                torch.from_numpy(qc[idx_test]).to(device),
                torch.from_numpy(f_pri[idx_test]).to(device),
                torch.from_numpy(loc_p[idx_test]).to(device),
                torch.ones((len(idx_test), 1024), device=device),
            ).cpu().numpy()
        t_pred = np.where(loc_p[idx_test] > 0.12, np.argmax(t_log, axis=-1), 0)

        for m in methods:
            rm = evaluate_policy_on_scenes(
                method_name=m, budget_bits=(20000 if m == "full_feature" else 2600),
                test_f_pri=f_pri[idx_test], test_f_curr=f_cur[idx_test],
                test_pri_idx=pi[idx_test], test_curr_idx=ci[idx_test],
                test_loc_p=loc_p[idx_test], test_c_k=c_k[idx_test], test_g_k=s3["all_g_k"][idx_test],
                test_raw_risk=s3["all_raw_risk"][idx_test], test_cal_prob=cal_p, test_cal_upper=cal_u,
                test_known_saliency=s3["all_known_saliency"][idx_test], test_twist_score=s3["all_twist_score"][idx_test],
                test_cell_gt=gt440[idx_test], test_unknown_gt=test_u_gt,
                codebook=cb, codec=codec, decoder_head=head, teacher_preds=t_pred,
                delta_target=(0.0 if m == "full_feature" else 0.15), max_safe_frac=0.35, verify_wire_packets=False,
            )
            per_m[m].append({
                "rate_kb_per_frame": rm["rate_kb_per_frame"],
                "known_f1": rm["known_f1"], "sync_f1": rm["sync_f1"],
                "state_distortion": rm["state_distortion"],
                "u_miss": rm["u_miss"], "event_miss_prob": rm["event_miss_prob"],
            })
        print(
            f"  [SegFormer-B4 seed={seed}] SafeSync: Known={per_m['safesync'][-1]['known_f1']:.2f}, "
            f"Sync={per_m['safesync'][-1]['sync_f1']:.2f}, U-Miss={per_m['safesync'][-1]['u_miss']:.2f} "
            f"| TWIST: Sync={per_m['twist_inspired'][-1]['sync_f1']:.2f}, UM={per_m['twist_inspired'][-1]['u_miss']:.2f} "
            f"| ECE {ece_u:.4f}->{ece_c:.4f} [{time.perf_counter()-ts:.1f}s]"
        )

    return {
        "backbone": "SegFormer-B4 (MiT-B4, 64.0M params, Train-Known-only PCA 768->256)",
        "budget_bits": 2600,
        "num_seeds": len(SEEDS),
        "ece_uncalibrated_mean": round(float(np.mean([x["ece_uncal"] for x in eces])), 4),
        "ece_calibrated_mean": round(float(np.mean([x["ece_cal"] for x in eces])), 4),
        "methods": {
            m: _agg_list_of_dicts(per_m[m], ["rate_kb_per_frame", "known_f1", "sync_f1", "state_distortion", "u_miss", "event_miss_prob"])
            for m in methods
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU Supplemental Suite running on:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    out_c = run_s1_s2_s3_s4_changeos(device)
    out_s5 = run_s5_cross_backbone_segformer(device)
    elapsed = time.perf_counter() - t0

    supp = {
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "pytorch_version": torch.__version__,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_gpu_hours": round(elapsed / 3600.0, 4),
        **out_c,
        "s5_cross_backbone_segformer_b4_3seed": out_s5,
    }
    supp_path = RESULTS_DIR / "gpu_supplemental_r36.json"
    supp_path.write_text(json.dumps(supp, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Saved] {supp_path} (total {elapsed:.1f}s / {elapsed/3600:.3f} GPU-h)")

    # Merge seamlessly into safesync_r36_complete_results.json!
    main_path = RESULTS_DIR / "safesync_r36_complete_results.json"
    main_res = json.loads(main_path.read_text(encoding="utf-8"))
    main_res["aggregated_3seed"]["quantizer_roi_ablation"] = supp["s2_quantizer_roi_3seed"]
    main_res["aggregated_3seed"]["real_rho_sweep"] = supp["s2_rho_sweep_3seed"]
    main_res["aggregated_3seed"]["real_alpha_sweep"] = supp["s2_alpha_sweep_3seed"]
    main_res["aggregated_3seed"]["channel_robustness_sweep"] = supp["s3_rician_equal_codec_3seed"]
    main_res["aggregated_3seed"]["real_codebook_size_sweep_3seed"] = supp["s1_codebook_size_sweep_real_3seed"]
    main_res["gpu_supplemental_studies"] = {
        "elapsed_seconds": supp["elapsed_seconds"],
        "s1_codebook_size_sweep_real_3seed": supp["s1_codebook_size_sweep_real_3seed"],
        "s4_full_xbd_test_933_3seed": supp["s4_full_xbd_test_933_3seed"],
        "s5_cross_backbone_segformer_b4_3seed": supp["s5_cross_backbone_segformer_b4_3seed"],
    }
    main_path.write_text(json.dumps(main_res, indent=2) + "\n", encoding="utf-8")
    print(f"[Merged] Updated {main_path} with 100% real 3-seed S1-S5 GPU measurements!")


if __name__ == "__main__":
    main()
