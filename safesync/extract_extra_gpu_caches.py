"""Extract two high-value real-satellite token caches on RTX 3090:
1. Full official xbd_test (all 933 bi-temporal 1024x1024 scenes = 955,392 tokens) via frozen ChangeOS Swin-T.
2. Cross-backbone SegFormer-B4 (MiT-B4) tokens on the exact 440 xBD cohort scenes (zero Swin-T overlap).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
CACHE_DIR = HERE / "data_cache"
sys.path.insert(0, str(ROOT / "vendor/torchange"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_real_task import load_models  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402


def extract_full_xbd_test_933(device: torch.device) -> dict:
    out_npz = CACHE_DIR / "xbd_full_test_933_tokens.npz"
    out_meta = CACHE_DIR / "xbd_full_test_933_meta.json"
    if out_npz.exists() and out_meta.exists():
        print(f"[Skip] Already exists: {out_npz}")
        return {"n_scenes": len(json.loads(out_meta.read_text(encoding="utf-8"))), "cached": True}

    print("=== [GPU Extract 1/2] Full Official xbd_test (933 scenes) via ChangeOS Swin-T ===")
    model = load_models(["changeos"], str(device))["changeos"].eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    # Collect all 933 scenes in deterministic shard order
    shards = sorted((ROOT / "data/xbd_test").glob("*.parquet"))
    total_n = sum(pq.read_metadata(s).num_rows for s in shards)
    print(f"  Total official xbd_test scenes: {total_n}")

    f_prior = np.zeros((total_n, 1024, 256), dtype=np.float16)
    f_curr = np.zeros((total_n, 1024, 256), dtype=np.float16)
    loc_prior_prob = np.zeros((total_n, 1024), dtype=np.float16)
    logits_curr_cell = np.zeros((total_n, 1024, 5), dtype=np.float16)
    logits_prior_cell = np.zeros((total_n, 1024, 5), dtype=np.float16)
    cell_gt_counts = np.zeros((total_n, 1024, 5), dtype=np.int16)
    cell_pre_bldg_pixels = np.zeros((total_n, 1024), dtype=np.int16)
    meta = []

    idx = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for sp in shards:
            pf = pq.ParquetFile(sp)
            for batch in pf.iter_batches(batch_size=8):
                rows = batch.to_pylist()
                t1_list, t2_list, m1_list, m2_list = [], [], [], []
                for r in rows:
                    im1 = np.asarray(Image.open(io.BytesIO(r["t1_image"]["bytes"])).convert("RGB")).copy()
                    im2 = np.asarray(Image.open(io.BytesIO(r["t2_image"]["bytes"])).convert("RGB")).copy()
                    m1 = np.asarray(Image.open(io.BytesIO(r["t1_mask"]["bytes"])))
                    m2 = np.asarray(Image.open(io.BytesIO(r["t2_mask"]["bytes"])))
                    t1_list.append(torch.from_numpy(im1).permute(2, 0, 1))
                    t2_list.append(torch.from_numpy(im2).permute(2, 0, 1))
                    m1_list.append(m1)
                    m2_list.append(m2)

                t1 = (torch.stack(t1_list).to(device).float().div_(255.0) - mean) / std
                t2 = (torch.stack(t2_list).to(device).float().div_(255.0) - mean) / std

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    enc1 = model.encoder(t1)
                    enc2 = model.encoder(t2)
                    loc_f1 = model.decoder.loc_neck(enc1)
                    dam_f1 = model.decoder.dam_neck(enc1)
                    dam_f2 = model.decoder.dam_neck(enc2)

                    tok1 = F.adaptive_avg_pool2d(dam_f1, (32, 32))
                    tok2 = F.adaptive_avg_pool2d(dam_f2, (32, 32))
                    loc_prob_32 = F.adaptive_avg_pool2d(model.head.loc_cls(loc_f1).sigmoid(), (32, 32))[:, 0]

                    up_tok2 = F.interpolate(tok2, size=(256, 256), mode="bilinear", align_corners=False)
                    up_tok1 = F.interpolate(tok1, size=(256, 256), mode="bilinear", align_corners=False)
                    st_curr = model.decoder.fuse_conv(torch.cat([loc_f1, up_tok2], dim=1))
                    st_prior = model.decoder.fuse_conv(torch.cat([loc_f1, up_tok1], dim=1))
                    dam_logit_curr = F.adaptive_avg_pool2d(model.head.dam_cls(st_curr), (32, 32))
                    dam_logit_prior = F.adaptive_avg_pool2d(model.head.dam_cls(st_prior), (32, 32))

                b = len(rows)
                f_prior[idx : idx + b] = tok1.permute(0, 2, 3, 1).reshape(b, 1024, 256).cpu().numpy().astype(np.float16)
                f_curr[idx : idx + b] = tok2.permute(0, 2, 3, 1).reshape(b, 1024, 256).cpu().numpy().astype(np.float16)
                loc_prior_prob[idx : idx + b] = loc_prob_32.reshape(b, 1024).cpu().numpy().astype(np.float16)
                logits_curr_cell[idx : idx + b] = dam_logit_curr.permute(0, 2, 3, 1).reshape(b, 1024, 5).cpu().numpy().astype(np.float16)
                logits_prior_cell[idx : idx + b] = dam_logit_prior.permute(0, 2, 3, 1).reshape(b, 1024, 5).cpu().numpy().astype(np.float16)

                for j, (r, m1, m2) in enumerate(zip(rows, m1_list, m2_list)):
                    m1_cells = (m1 > 0).reshape(32, 32, 32, 32).transpose(0, 2, 1, 3).reshape(1024, 1024)
                    m2_cells = m2.reshape(32, 32, 32, 32).transpose(0, 2, 1, 3).reshape(1024, 1024)
                    cell_pre_bldg_pixels[idx + j] = m1_cells.sum(axis=1).astype(np.int16)
                    for c_id in range(5):
                        cell_gt_counts[idx + j, :, c_id] = (m2_cells == c_id).sum(axis=1).astype(np.int16)
                    ev = r["image_name"].rsplit("_", 1)[0].split("/")[-1]
                    meta.append({
                        "scene_index": idx + j,
                        "image_name": r["image_name"],
                        "event": ev,
                        "pre_bldg_pixels": int((m1 > 0).sum()),
                        "class_pixels": [int((m2 == c_i).sum()) for c_i in range(5)],
                    })
                idx += b
                if idx % 160 == 0 or idx == total_n:
                    el = time.perf_counter() - t0
                    print(f"  [ChangeOS xbd_test] {idx}/{total_n} ({el:.1f}s, {idx/el:.1f} pairs/s)")

    del model
    torch.cuda.empty_cache()

    np.savez(
        out_npz,
        f_prior=f_prior[:idx],
        f_curr=f_curr[:idx],
        loc_prior_prob=loc_prior_prob[:idx],
        logits_curr_cell=logits_curr_cell[:idx],
        logits_prior_cell=logits_prior_cell[:idx],
        cell_gt_counts=cell_gt_counts[:idx],
        cell_pre_bldg_pixels=cell_pre_bldg_pixels[:idx],
    )
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    el = time.perf_counter() - t0
    print(f"  [Saved] {out_npz} ({out_npz.stat().st_size/1024/1024:.1f} MB) in {el:.1f}s")
    return {"n_scenes": idx, "elapsed_s": round(el, 2)}


def extract_segformer_b4_440(device: torch.device) -> dict:
    out_npz = CACHE_DIR / "xbd_segformer_b4_440_tokens.npz"
    if out_npz.exists():
        print(f"[Skip] Already exists: {out_npz}")
        return {"n_scenes": 440, "cached": True}

    print("\n=== [GPU Extract 2/2] Cross-Backbone SegFormer-B4 (MiT-B4) on Exact 440 xBD Scenes ===")
    scenes_440 = json.loads((CACHE_DIR / "xbd_safesync_scenes_440.json").read_text(encoding="utf-8"))
    target_names = {s["image_name"]: s["scene_index"] for s in scenes_440}
    n_scenes = len(scenes_440)

    seg_dir = ROOT / "models/post_segformer_b4"
    seg = SegformerForSemanticSegmentation.from_pretrained(str(seg_dir), local_files_only=True).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    # Hook decode_head.batch_norm output (768-ch fused multi-scale features right before classifier)
    fused_holder = {}
    def _hook(_mod, _inp, out):
        fused_holder["feat"] = out
    h_handle = seg.decode_head.batch_norm.register_forward_hook(_hook)

    # Extract raw 768-d pooled features first, then fit Train-Known-only PCA(256) (ZERO leakage!)
    f_pri_768 = np.zeros((n_scenes, 1024, 768), dtype=np.float16)
    f_cur_768 = np.zeros((n_scenes, 1024, 768), dtype=np.float16)
    loc_prior_prob = np.zeros((n_scenes, 1024), dtype=np.float16)
    logits_curr_cell = np.zeros((n_scenes, 1024, 5), dtype=np.float16)
    logits_prior_cell = np.zeros((n_scenes, 1024, 5), dtype=np.float16)
    found = np.zeros(n_scenes, dtype=bool)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for split_dir in [ROOT / "data/xbd_hold", ROOT / "data/xbd_test"]:
            for sp in sorted(split_dir.glob("*.parquet")):
                pf = pq.ParquetFile(sp)
                for batch in pf.iter_batches(batch_size=16):
                    rows = [r for r in batch.to_pylist() if r["image_name"] in target_names]
                    if not rows:
                        continue
                    t1_list, t2_list, idxs = [], [], []
                    for r in rows:
                        si = target_names[r["image_name"]]
                        if found[si]:
                            continue
                        im1 = Image.open(io.BytesIO(r["t1_image"]["bytes"])).convert("RGB").resize((512, 512), Image.BILINEAR)
                        im2 = Image.open(io.BytesIO(r["t2_image"]["bytes"])).convert("RGB").resize((512, 512), Image.BILINEAR)
                        t1_list.append(torch.from_numpy(np.asarray(im1).copy()).permute(2, 0, 1))
                        t2_list.append(torch.from_numpy(np.asarray(im2).copy()).permute(2, 0, 1))
                        idxs.append(si)
                    if not idxs:
                        continue
                    t1 = (torch.stack(t1_list).to(device).float().div_(255.0) - mean) / std
                    t2 = (torch.stack(t2_list).to(device).float().div_(255.0) - mean) / std

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out1 = seg(pixel_values=t1)
                        feat1 = F.adaptive_avg_pool2d(fused_holder["feat"], (32, 32))
                        log1 = F.adaptive_avg_pool2d(out1.logits, (32, 32))
                        p_bldg1 = 1.0 - F.softmax(log1.float(), dim=1)[:, 0]  # (B, 32, 32)

                        out2 = seg(pixel_values=t2)
                        feat2 = F.adaptive_avg_pool2d(fused_holder["feat"], (32, 32))
                        log2 = F.adaptive_avg_pool2d(out2.logits, (32, 32))

                    f1_np = feat1.permute(0, 2, 3, 1).reshape(len(idxs), 1024, 768).cpu().numpy().astype(np.float16)
                    f2_np = feat2.permute(0, 2, 3, 1).reshape(len(idxs), 1024, 768).cpu().numpy().astype(np.float16)
                    lp_np = p_bldg1.reshape(len(idxs), 1024).cpu().numpy().astype(np.float16)
                    l1_np = log1.permute(0, 2, 3, 1).reshape(len(idxs), 1024, 5).cpu().numpy().astype(np.float16)
                    l2_np = log2.permute(0, 2, 3, 1).reshape(len(idxs), 1024, 5).cpu().numpy().astype(np.float16)

                    for j, si in enumerate(idxs):
                        f_pri_768[si] = f1_np[j]
                        f_cur_768[si] = f2_np[j]
                        loc_prior_prob[si] = lp_np[j]
                        logits_prior_cell[si] = l1_np[j]
                        logits_curr_cell[si] = l2_np[j]
                        found[si] = True
                if int(found.sum()) == n_scenes:
                    break
            if int(found.sum()) == n_scenes:
                break

    h_handle.remove()
    del seg
    torch.cuda.empty_cache()
    assert int(found.sum()) == n_scenes, f"Missing scenes: {n_scenes - int(found.sum())}"

    # Strict Train-Known-only PCA(768 -> 256) so bottleneck dim matches 256-d VQ cleanly with zero test leakage!
    train_idx = [s["scene_index"] for s in scenes_440 if s["split"] == "train_known"]
    tr_mat = np.concatenate([
        f_pri_768[train_idx].reshape(-1, 768),
        f_cur_768[train_idx].reshape(-1, 768),
    ], axis=0).astype(np.float32)
    rng = np.random.default_rng(42)
    sub = tr_mat[rng.choice(len(tr_mat), size=min(65536, len(tr_mat)), replace=False)]
    mu = sub.mean(axis=0, keepdims=True)
    sub_c = torch.from_numpy(sub - mu).to(device)
    _, _, vh = torch.linalg.svd(sub_c, full_matrices=False)
    proj = vh[:256].T.cpu().numpy().astype(np.float32)  # (768, 256)
    mu_np = mu.astype(np.float32)

    f_prior = np.zeros((n_scenes, 1024, 256), dtype=np.float16)
    f_curr = np.zeros((n_scenes, 1024, 256), dtype=np.float16)
    for i in range(n_scenes):
        f_prior[i] = ((f_pri_768[i].astype(np.float32) - mu_np) @ proj).astype(np.float16)
        f_curr[i] = ((f_cur_768[i].astype(np.float32) - mu_np) @ proj).astype(np.float16)

    # Copy exact pixel GT counts from existing 440 cache so labels are 100% identical
    z440 = np.load(CACHE_DIR / "xbd_safesync_tokens_440.npz")
    np.savez(
        out_npz,
        f_prior=f_prior,
        f_curr=f_curr,
        loc_prior_prob=loc_prior_prob,
        logits_curr_cell=logits_curr_cell,
        logits_prior_cell=logits_prior_cell,
        cell_gt_counts=z440["cell_gt_counts"],
        cell_pre_bldg_pixels=z440["cell_pre_bldg_pixels"],
    )
    el = time.perf_counter() - t0
    print(f"  [Saved] {out_npz} ({out_npz.stat().st_size/1024/1024:.1f} MB) in {el:.1f}s")
    return {"n_scenes": n_scenes, "elapsed_s": round(el, 2)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    r1 = extract_full_xbd_test_933(device)
    r2 = extract_segformer_b4_440(device)
    print(json.dumps({"status": "ok", "xbd_full_test_933": r1, "segformer_b4_440": r2}, indent=2))


if __name__ == "__main__":
    main()
