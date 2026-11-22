"""Unit tests for SafeSync-Mesh wire codec, 2D-Arith mask, range ranks, RKS Bloom filter, and result integrity."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safesync.safesync_wire import (
    ConditionalTokenEntropyCodec,
    decode_safesync_packet,
    encode_safesync_packet,
)
from safesync.run_multisat_constellation_study import BloomFilterRKS
from verify_quickstart import verify_published_tables, verify_wire_protocol_and_idempotence


def test_wire_codec_and_idempotence() -> None:
    verify_wire_protocol_and_idempotence()


def test_2d_arith_mask_and_range_ranks_bit_exact() -> None:
    rng = np.random.default_rng(1337)
    cb = rng.standard_normal((512, 256), dtype=np.float32)
    codec = ConditionalTokenEntropyCodec(cb)
    pri = rng.integers(0, 512, size=1024, dtype=np.int32)
    cur = pri.copy()
    # Spatially clustered 2D disaster patch (realistic mask correlation)
    mask2d = np.zeros((32, 32), dtype=bool)
    mask2d[6:20, 8:24] = True
    sel = mask2d.reshape(1024)
    safe2d = np.zeros((32, 32), dtype=bool)
    safe2d[8:13, 10:17] = True
    safe = safe2d.reshape(1024)
    sel_idx = np.flatnonzero(sel)
    cur[sel_idx] = codec.order[pri[sel_idx], rng.integers(0, 8, size=len(sel_idx))]
    risk = rng.uniform(0.0, 1.0, size=1024).astype(np.float32)

    enc_arith = encode_safesync_packet(
        "cluster_scene", pri, cur, sel, safe, risk, codec,
        use_conditional_entropy=True, use_2d_arith_mask=True, use_range_ranks=False,
    )
    enc_zlib = encode_safesync_packet(
        "cluster_scene", pri, cur, sel, safe, risk, codec,
        use_conditional_entropy=True, use_2d_arith_mask=False, use_range_ranks=False,
    )
    enc_range = encode_safesync_packet(
        "cluster_scene", pri, cur, sel, safe, risk, codec,
        use_conditional_entropy=True, use_2d_arith_mask=True, use_range_ranks=True,
    )

    assert enc_arith["mask_mode"] == 3
    assert enc_arith["mask_bits"] < enc_zlib["mask_bits"], "2D-Arith should beat zlib on clustered 2D masks"
    for enc in (enc_arith, enc_zlib, enc_range):
        dec = decode_safesync_packet(enc["packet"], "cluster_scene", pri, codec)
        assert np.array_equal(dec["selected_mask"], sel)
        assert np.array_equal(dec["safety_mask"], safe)
        assert np.array_equal(dec["reconstructed_indices"][sel], cur[sel])


def test_bloom_filter_rks_no_false_negatives() -> None:
    bf = BloomFilterRKS(n_bits=32)
    ids = [1001, 2048, 99991, 424242]
    for rid in ids:
        bf.add(rid)
    for rid in ids:
        assert bf.maybe_contains(rid), f"Bloom filter false negative on {rid}"


def test_arbitrary_codebook_sizes_bit_exact() -> None:
    rng = np.random.default_rng(2026)
    for k_size in (128, 256, 512, 1024):
        cb = rng.standard_normal((k_size, 64), dtype=np.float32)
        codec = ConditionalTokenEntropyCodec(cb)
        pri = rng.integers(0, k_size, size=1024, dtype=np.int32)
        cur = rng.integers(0, k_size, size=1024, dtype=np.int32)
        sel = rng.random(1024) < 0.30
        safe = sel & (rng.random(1024) < 0.15)
        risk = rng.random(1024).astype(np.float32)
        for uc in (True, False):
            enc = encode_safesync_packet(f"k_{k_size}", pri, cur, sel, safe, risk, codec, use_conditional_entropy=uc)
            dec = decode_safesync_packet(enc["packet"], f"k_{k_size}", pri, codec)
            assert np.array_equal(dec["selected_mask"], sel)
            assert np.array_equal(dec["safety_mask"], safe)
            assert np.array_equal(dec["reconstructed_indices"][sel], cur[sel])


def test_published_result_tables() -> None:
    verify_published_tables()
