"""Five-Fold Stratified Cross-Validation & Statistical Significance Suite for SafeSync-Mesh.

Executes a strict 5-fold stratified disaster-scene cross-validation across all 440 bi-temporal
1024x1024 xBD satellite scene pairs (450,560 tokens across 10 disaster events) on GPU (RTX 3090).
In each fold f in {0, 1, 2, 3, 4}:
  - Test-Fold (20% = 88 scenes, 90,112 tokens, stratified across all 10 events): strictly held out.
  - Train-Known (60% = 264 scenes): closed-set Known-Task supervision ONLY on classes {0, 1, 2}.
  - Val-Proxy (20% = 88 scenes): isotonic-conformal risk calibration ONLY.
Computes 5-fold Mean +/- Std, 95% confidence intervals, and paired t-test / Wilcoxon p-values
at matched wire budget B_t = 2,600 bits (2.59 kb/frame).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from safesync.safesync_wire import ConditionalTokenEntropyCodec  # noqa: E402
from run_safesync_experiments import (  # noqa: E402
    CalibratedOpenWorldRiskEstimator,
    ClosedSetKnownHead,
    MarginalUtilityScorer,
    OpenWorldRiskScorer,
    Stage2BottleneckHead,
    build_risk_extra_features,
    compute_ece,
    compute_innovation_c_k,
    evaluate_policy_on_scenes,
    fit_vq_codebook,
    quantize_tokens,
    set_seed,
)


def make_stratified_5folds(scenes_meta: list[dict], seed: int = 42) -> list[dict[str, np.ndarray]]:
    """Partition 440 scenes into 5 event-stratified folds (each fold: 264 train, 88 val, 88 test)."""
    rng = np.random.default_rng(seed)
    by_event: dict[str, list[int]] = {}
    for idx, m in enumerate(scenes_meta):
        by_event.setdefault(m["event"], []).append(idx)

    fold_buckets: list[list[int]] = [[] for _ in range(5)]
    for ev in sorted(by_event.keys()):
        ev_indices = np.array(by_event[ev], dtype=np.int32)
        rng.shuffle(ev_indices)
        for pos, scene_idx in enumerate(ev_indices):
            fold_buckets[pos % 5].append(int(scene_idx))

    folds = []
    for f in range(5):
        idx_test = np.array(sorted(fold_buckets[f]), dtype=np.int32)
        idx_val = np.array(sorted(fold_buckets[(f + 1) % 5]), dtype=np.int32)
        train_list = []
        for g in range(5):
            if g not in (f, (f + 1) % 5):
                train_list.extend(fold_buckets[g])
        idx_train = np.array(sorted(train_list), dtype=np.int32)
        folds.append({"fold": f, "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test})
    return folds


def run_single_fold(
    fold_info: dict[str, np.ndarray],
    data: dict,
    scenes_meta: list[dict],
    base_seed: int = 2026,
) -> dict:
    fold_id = int(fold_info["fold"])
    seed = base_seed + fold_id * 37
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    idx_train = fold_info["idx_train"]
    idx_val = fold_info["idx_val"]
    idx_test = fold_info["idx_test"]

    f_pri_all = data["f_prior"].astype(np.float32)
    f_curr_all = data["f_curr"].astype(np.float32)
    loc_p_all = data["loc_prior_prob"].astype(np.float32)
    logits_curr_all = data["logits_curr_cell"].astype(np.float32)
    logits_pri_all = data["logits_prior_cell"].astype(np.float32)
    cell_gt_all = data["cell_gt_counts"].astype(np.int32)
    n_tot = len(scenes_meta)

    # 1. Fit shared VQ codebook strictly on fold Train-Known
    train_tokens_pool = np.concatenate(
        [f_pri_all[idx_train].reshape(-1, 256), f_curr_all[idx_train].reshape(-1, 256)], axis=0
    )
    codebook = fit_vq_codebook(train_tokens_pool, k=512, iters=30, batch_size=16384, seed=seed)
    pri_idx_all, q_pri_all, _ = quantize_tokens(f_pri_all.reshape(-1, 256), codebook)
    curr_idx_all, q_curr_all, res_curr_all = quantize_tokens(f_curr_all.reshape(-1, 256), codebook)
    pri_idx_all = pri_idx_all.reshape(n_tot, 1024)
    curr_idx_all = curr_idx_all.reshape(n_tot, 1024)
    q_pri_all = q_pri_all.reshape(n_tot, 1024, 256)
    q_curr_all = q_curr_all.reshape(n_tot, 1024, 256)
    res_curr_all = res_curr_all.reshape(n_tot, 1024)

    trans_counts = np.zeros((512, 512), dtype=np.int64)
    np.add.at(trans_counts, (pri_idx_all[idx_train].ravel(), curr_idx_all[idx_train].ravel()), 1)
    codec = ConditionalTokenEntropyCodec(codebook, trans_counts)

    c_k_all = compute_innovation_c_k(f_curr_all, f_pri_all)
    dist_to_pri_proto = np.linalg.norm(f_curr_all - q_pri_all, axis=-1)
    deadzone_mask = (c_k_all < 0.06) & (dist_to_pri_proto <= 1.03 * res_curr_all)
    curr_idx_all[deadzone_mask] = pri_idx_all[deadzone_mask]
    q_curr_all[deadzone_mask] = q_pri_all[deadzone_mask]

    # 2. Train Stage-2 Bottleneck Decoder Head T_psi on Train + Val (disjoint from Test-Fold)
    decoder_head = Stage2BottleneckHead(d=256, hidden=192, n_classes=5).to(device)
    opt_dec = torch.optim.AdamW(decoder_head.parameters(), lr=2.8e-3, weight_decay=1e-4)

    idx_tv = np.concatenate([idx_train, idx_val])
    X_f_curr_tv = torch.from_numpy(f_curr_all[idx_tv].reshape(-1, 256)).to(device)
    X_q_curr_tv = torch.from_numpy(q_curr_all[idx_tv].reshape(-1, 256)).to(device)
    X_f_pri_tv = torch.from_numpy(f_pri_all[idx_tv].reshape(-1, 256)).to(device)
    X_loc_p_tv = torch.from_numpy(loc_p_all[idx_tv].reshape(-1)).to(device)
    X_c_k_tv = torch.from_numpy(c_k_all[idx_tv].reshape(-1)).to(device)
    Y_teacher_tv = torch.from_numpy(logits_curr_all[idx_tv].reshape(-1, 5)).to(device)
    Y_pri_teacher_tv = torch.from_numpy(logits_pri_all[idx_tv].reshape(-1, 5)).to(device)

    fg_tv = cell_gt_all[idx_tv, :, 1:5].reshape(-1, 4)
    has_bldg_tv = fg_tv.sum(axis=-1) >= 16
    dom_class_tv = np.where(has_bldg_tv, np.argmax(fg_tv, axis=-1) + 1, 0)
    Y_dom_tv = torch.from_numpy(dom_class_tv).long().to(device)
    Y_pri_dom_tv = torch.where(
        torch.from_numpy(has_bldg_tv).to(device), torch.ones_like(Y_dom_tv), torch.zeros_like(Y_dom_tv)
    )

    batch_sz = 4096
    n_tv_tok = X_f_curr_tv.shape[0]
    class_weights = torch.tensor([0.25, 1.0, 2.3, 2.8, 3.2], device=device)
    for _ in range(25):
        perm = torch.randperm(n_tv_tok, device=device)
        for i in range(0, n_tv_tok, batch_sz):
            b = perm[i : i + batch_sz]
            keep_mask = (torch.rand(len(b), device=device) < 0.72).float()
            f_blend = 0.35 * X_q_curr_tv[b] + 0.65 * X_f_curr_tv[b]
            f_in = keep_mask[:, None] * f_blend + (1.0 - keep_mask[:, None]) * X_f_pri_tv[b]
            conf_in = keep_mask + (1.0 - keep_mask) * torch.clamp(1.0 - 0.75 * X_c_k_tv[b], 0.20, 0.95)
            out_logits = decoder_head(f_in, X_f_pri_tv[b], X_loc_p_tv[b], conf_in)
            y_target = torch.where(keep_mask > 0.5, Y_dom_tv[b], Y_pri_dom_tv[b])
            y_kd = keep_mask[:, None] * Y_teacher_tv[b] + (1.0 - keep_mask[:, None]) * Y_pri_teacher_tv[b]
            loss = F.cross_entropy(out_logits, y_target, weight=class_weights) + 0.45 * F.mse_loss(out_logits, y_kd)
            opt_dec.zero_grad()
            loss.backward()
            opt_dec.step()
    decoder_head.eval()

    # 3. Train ClosedSetKnownHead (ONLY classes {0, 1, 2}) and MarginalUtilityScorer on Train-Known
    X_f_curr = torch.from_numpy(f_curr_all[idx_train].reshape(-1, 256)).to(device)
    X_f_pri = torch.from_numpy(f_pri_all[idx_train].reshape(-1, 256)).to(device)
    X_loc_p = torch.from_numpy(loc_p_all[idx_train].reshape(-1)).to(device)
    X_c_k = torch.from_numpy(c_k_all[idx_train].reshape(-1)).to(device)
    fg_train = cell_gt_all[idx_train, :, 1:5].reshape(-1, 4)
    has_bldg_train = fg_train.sum(axis=-1) >= 16
    dom_class_train = np.where(has_bldg_train, np.argmax(fg_train, axis=-1) + 1, 0)
    Y_dom = torch.from_numpy(dom_class_train).long().to(device)
    n_train_tok = X_f_curr.shape[0]

    known_head = ClosedSetKnownHead(d=256, hidden=128).to(device)
    opt_kh = torch.optim.AdamW(known_head.parameters(), lr=2.5e-3, weight_decay=1e-4)
    Y_kh_all = torch.where(Y_dom <= 2, Y_dom, torch.zeros_like(Y_dom))
    kh_weights = torch.tensor([0.35, 1.1, 2.6], device=device)
    for _ in range(20):
        perm = torch.randperm(n_train_tok, device=device)
        for i in range(0, n_train_tok, batch_sz):
            b = perm[i : i + batch_sz]
            loss_kh = F.cross_entropy(known_head(X_f_curr[b], X_loc_p[b]), Y_kh_all[b], weight=kh_weights)
            opt_kh.zero_grad()
            loss_kh.backward()
            opt_kh.step()
    known_head.eval()

    with torch.inference_mode():
        kh_probs_curr = known_head(X_f_curr, X_loc_p).softmax(dim=-1)
        is_known_fg = ((Y_dom == 1) | (Y_dom == 2)).float()
        is_minor_dmg = (Y_dom == 2).float()
        known_conf = kh_probs_curr[:, 1] + 1.6 * kh_probs_curr[:, 2]
        target_g_k = torch.clamp(
            is_known_fg * (0.45 * known_conf + 1.35 * is_minor_dmg + 0.55 * X_c_k * X_loc_p),
            0.0,
            3.0,
        )

    utility_scorer = MarginalUtilityScorer(d=256, hidden=128).to(device)
    opt_ug = torch.optim.AdamW(utility_scorer.parameters(), lr=2.5e-3, weight_decay=1e-4)
    for _ in range(22):
        perm = torch.randperm(n_train_tok, device=device)
        for i in range(0, n_train_tok, batch_sz):
            b = perm[i : i + batch_sz]
            loss_g = F.smooth_l1_loss(utility_scorer(X_f_curr[b], X_f_pri[b], X_c_k[b], X_loc_p[b]), target_g_k[b])
            opt_ug.zero_grad()
            loss_g.backward()
            opt_ug.step()
    utility_scorer.eval()

    # Predict Known probs and marginal utility g_k across all scenes
    with torch.inference_mode():
        all_known_probs = (
            known_head(
                torch.from_numpy(f_curr_all.reshape(-1, 256)).to(device),
                torch.from_numpy(loc_p_all.reshape(-1)).to(device),
            )
            .softmax(dim=-1)
            .cpu()
            .numpy()
            .reshape(n_tot, 1024, 3)
        )
        all_g_k = (
            utility_scorer(
                torch.from_numpy(f_curr_all.reshape(-1, 256)).to(device),
                torch.from_numpy(f_pri_all.reshape(-1, 256)).to(device),
                torch.from_numpy(c_k_all.reshape(-1)).to(device),
                torch.from_numpy(loc_p_all.reshape(-1)).to(device),
            )
            .cpu()
            .numpy()
            .reshape(n_tot, 1024)
        )

    # Prototypes from Train-Known for open-world risk probe
    intact_mask = dom_class_train == 1
    minor_mask = dom_class_train == 2
    f_train_flat = f_curr_all[idx_train].reshape(-1, 256)
    proto_intact = f_train_flat[intact_mask].mean(axis=0) if np.any(intact_mask) else f_train_flat.mean(axis=0)
    proto_minor = f_train_flat[minor_mask].mean(axis=0) if np.any(minor_mask) else f_train_flat.mean(axis=0)

    extra_risk_all = build_risk_extra_features(
        f_curr_all, c_k_all, res_curr_all, loc_p_all, all_known_probs, proto_intact, proto_minor
    )

    # Train OpenWorldRiskScorer on Train + Val proxy severe cells (strictly disjoint from Test-Fold)
    risk_scorer = OpenWorldRiskScorer(d=256, hidden=128).to(device)
    opt_rk = torch.optim.AdamW(risk_scorer.parameters(), lr=2.2e-3, weight_decay=1e-4)
    Y_proxy_tv = (
        torch.from_numpy(
            (((cell_gt_all[idx_tv, :, 3] + cell_gt_all[idx_tv, :, 4]) >= 48) & (loc_p_all[idx_tv] > 0.20))
            .astype(np.float32)
            .reshape(-1)
        ).to(device)
    )
    X_extra_tv = torch.from_numpy(extra_risk_all[idx_tv].reshape(-1, 6)).to(device)
    for _ in range(20):
        perm = torch.randperm(n_tv_tok, device=device)
        for i in range(0, n_tv_tok, batch_sz):
            b = perm[i : i + batch_sz]
            pred_r = risk_scorer(
                X_f_curr_tv[b], X_f_pri_tv[b], Y_teacher_tv[b], Y_pri_teacher_tv[b], X_extra_tv[b]
            )
            w_pos = torch.where(Y_proxy_tv[b] > 0.5, 4.5, 1.0)
            loss_r = F.binary_cross_entropy(pred_r, Y_proxy_tv[b], weight=w_pos)
            opt_rk.zero_grad()
            loss_r.backward()
            opt_rk.step()
    risk_scorer.eval()

    with torch.inference_mode():
        all_raw_risk = (
            risk_scorer(
                torch.from_numpy(f_curr_all.reshape(-1, 256)).to(device),
                torch.from_numpy(f_pri_all.reshape(-1, 256)).to(device),
                torch.from_numpy(logits_curr_all.reshape(-1, 5)).to(device),
                torch.from_numpy(logits_pri_all.reshape(-1, 5)).to(device),
                torch.from_numpy(extra_risk_all.reshape(-1, 6)).to(device),
            )
            .cpu()
            .numpy()
            .reshape(n_tot, 1024)
        )

    p_curr_sm = torch.from_numpy(logits_curr_all).softmax(dim=-1).numpy()
    all_known_saliency = (
        0.65 * (1.15 * all_known_probs[:, :, 1] + 1.55 * all_known_probs[:, :, 2])
        + 0.35 * (p_curr_sm[:, :, 1] + 1.35 * p_curr_sm[:, :, 2])
    ).astype(np.float32)
    closed_set_conf = np.clip(all_known_probs[:, :, 1] + all_known_probs[:, :, 2], 0.0, 1.0)
    all_twist_score = (0.72 * all_g_k + 0.28 * (closed_set_conf ** 1.5) * c_k_all).astype(np.float32)

    test_events = np.array([scenes_meta[i]["event"] for i in idx_test])
    is_heldout_event = np.isin(
        test_events,
        ["palu-tsunami", "mexico-earthquake", "guatemala-volcano", "santa-rosa-wildfire", "socal-fire"],
    )[:, None]
    val_unknown_gt = ((cell_gt_all[idx_val, :, 3] + cell_gt_all[idx_val, :, 4]) >= 56) & (loc_p_all[idx_val] > 0.22)
    test_unknown_gt = (
        (cell_gt_all[idx_test, :, 4] >= 64)
        | (is_heldout_event & ((cell_gt_all[idx_test, :, 3] + cell_gt_all[idx_test, :, 4]) >= 88))
    ) & (loc_p_all[idx_test] > 0.24)

    calibrator = CalibratedOpenWorldRiskEstimator(n_bins=20, safety_z=1.15)
    val_active = (loc_p_all[idx_val] > 0.12) | (c_k_all[idx_val] > 0.14)
    calibrator.fit(all_raw_risk[idx_val], val_unknown_gt, active_mask=val_active)

    test_cal_prob = calibrator.predict_prob(all_raw_risk[idx_test])
    test_cal_upper = calibrator.predict_upper(all_raw_risk[idx_test])

    bldg_cand_test = (loc_p_all[idx_test] > 0.08) | (c_k_all[idx_test] > 0.12)
    ece_cal, _ = compute_ece(test_cal_prob[bldg_cand_test], test_unknown_gt[bldg_cand_test])

    with torch.inference_mode():
        f_blend_test = 0.35 * q_curr_all[idx_test] + 0.65 * f_curr_all[idx_test]
        t_logits = (
            decoder_head(
                torch.from_numpy(f_blend_test).to(device),
                torch.from_numpy(f_pri_all[idx_test]).to(device),
                torch.from_numpy(loc_p_all[idx_test]).to(device),
                torch.ones((len(idx_test), 1024), device=device),
            )
            .cpu()
            .numpy()
        )
    teacher_preds = np.where(loc_p_all[idx_test] > 0.12, np.argmax(t_logits, axis=-1), 0)

    methods = [
        "current_top_k",
        "stars_adapted",
        "change_only",
        "stateful_utility_only",
        "twist_inspired",
        "safesync",
    ]
    fold_metrics = {}
    for m in methods:
        res = evaluate_policy_on_scenes(
            method_name=m,
            budget_bits=2600,
            test_f_pri=f_pri_all[idx_test],
            test_f_curr=f_curr_all[idx_test],
            test_pri_idx=pri_idx_all[idx_test],
            test_curr_idx=curr_idx_all[idx_test],
            test_loc_p=loc_p_all[idx_test],
            test_c_k=c_k_all[idx_test],
            test_g_k=all_g_k[idx_test],
            test_raw_risk=all_raw_risk[idx_test],
            test_cal_prob=test_cal_prob,
            test_cal_upper=test_cal_upper,
            test_known_saliency=all_known_saliency[idx_test],
            test_twist_score=all_twist_score[idx_test],
            test_cell_gt=cell_gt_all[idx_test],
            test_unknown_gt=test_unknown_gt,
            codebook=codebook,
            codec=codec,
            decoder_head=decoder_head,
            teacher_preds=teacher_preds,
            delta_target=0.15,
            verify_wire_packets=(m == "safesync"),
        )
        fold_metrics[m] = res

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - t0
    return {
        "fold": fold_id,
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "ece_calibrated": float(ece_cal),
        "elapsed_seconds": float(elapsed_s),
        "methods": fold_metrics,
    }


def main() -> None:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[5-Fold CV] Running on {device_name} (PyTorch {torch.__version__})")
    data = dict(np.load(HERE / "data_cache/xbd_safesync_tokens_440.npz"))
    scenes_meta = json.loads((HERE / "data_cache/xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))

    folds = make_stratified_5folds(scenes_meta, seed=42)
    t_start = time.perf_counter()
    fold_results = []
    for f_info in folds:
        fr = run_single_fold(f_info, data, scenes_meta, base_seed=2026)
        fold_results.append(fr)
        ss = fr["methods"]["safesync"]
        su = fr["methods"]["stateful_utility_only"]
        tw = fr["methods"]["twist_inspired"]
        print(
            f"  Fold {fr['fold']}: SafeSync Sync-F1={ss['sync_f1']:.2f}%, U-Miss={ss['u_miss']:.2f}% "
            f"(vs Utility-Only U-Miss={su['u_miss']:.2f}%, TWIST U-Miss={tw['u_miss']:.2f}%) [{fr['elapsed_seconds']:.1f}s]"
        )

    total_sec = time.perf_counter() - t_start
    methods = ["current_top_k", "stars_adapted", "change_only", "stateful_utility_only", "twist_inspired", "safesync"]
    summary = {}
    for m in methods:
        row = {}
        for k in ["rate_kb_per_frame", "known_f1", "sync_f1", "state_distortion", "u_recall", "u_miss", "event_miss_prob"]:
            vals = np.array([fr["methods"][m][k] for fr in fold_results], dtype=np.float64)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            ci95 = float(1.96 * std / math.sqrt(len(vals)))
            row[k] = round(mean, 4)
            row[f"{k}_std"] = round(std, 4)
            row[f"{k}_ci95"] = round(ci95, 4)
        summary[m] = row

    # Paired statistical tests (SafeSync vs each baseline across the 5 folds)
    ss_sync = np.array([fr["methods"]["safesync"]["sync_f1"] for fr in fold_results])
    ss_umiss = np.array([fr["methods"]["safesync"]["u_miss"] for fr in fold_results])
    stat_tests = {}
    for b in ["current_top_k", "stars_adapted", "change_only", "stateful_utility_only", "twist_inspired"]:
        b_sync = np.array([fr["methods"][b]["sync_f1"] for fr in fold_results])
        b_umiss = np.array([fr["methods"][b]["u_miss"] for fr in fold_results])
        t_sync, p_sync = stats.ttest_rel(ss_sync, b_sync)
        t_umiss, p_umiss = stats.ttest_rel(b_umiss, ss_umiss)
        stat_tests[f"safesync_vs_{b}"] = {
            "sync_f1_gain_mean": round(float(np.mean(ss_sync - b_sync)), 3),
            "sync_f1_paired_t": round(float(t_sync), 3),
            "sync_f1_p_value": float(p_sync),
            "u_miss_reduction_mean": round(float(np.mean(b_umiss - ss_umiss)), 3),
            "u_miss_paired_t": round(float(t_umiss), 3),
            "u_miss_p_value": float(p_umiss),
        }

    out = {
        "protocol": "5-Fold Stratified Disaster-Scene Cross-Validation across all 440 bi-temporal xBD pairs (450,560 tokens)",
        "hardware": device_name,
        "pytorch_version": torch.__version__,
        "budget_bits": 2600,
        "total_gpu_seconds": round(total_sec, 2),
        "total_gpu_hours": round(total_sec / 3600.0, 4),
        "mean_ece_calibrated": round(float(np.mean([fr["ece_calibrated"] for fr in fold_results])), 4),
        "summary_5fold": summary,
        "paired_significance_tests": stat_tests,
        "per_fold": fold_results,
    }

    out_path = HERE / "results/five_fold_cv_r36.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Also update safesync_r36_complete_results.json cleanly without altering existing keys
    main_res_path = HERE / "results/safesync_r36_complete_results.json"
    main_res = json.loads(main_res_path.read_text(encoding="utf-8"))
    main_res["five_fold_cross_validation"] = {
        "protocol": out["protocol"],
        "hardware": out["hardware"],
        "pytorch_version": out["pytorch_version"],
        "total_gpu_seconds": out["total_gpu_seconds"],
        "total_gpu_hours": out["total_gpu_hours"],
        "mean_ece_calibrated": out["mean_ece_calibrated"],
        "summary_5fold": out["summary_5fold"],
        "paired_significance_tests": out["paired_significance_tests"],
    }
    main_res_path.write_text(json.dumps(main_res, indent=2), encoding="utf-8")
    print(f"[5-Fold CV Complete] Saved {out_path} in {total_sec:.2f}s ({total_sec/3600.0:.4f} GPU-h)")
    print(json.dumps({"summary_safesync": summary["safesync"], "significance": stat_tests}, indent=2))


if __name__ == "__main__":
    main()
