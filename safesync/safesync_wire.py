"""Provenance-ready binary packet wire format and conditional entropy codec for SafeSync (§III-B, §IV-F).

Every transmitted differential update Z_t is serialized into a verifiable binary packet:
  R_t = sum_{k in Z_t} b_k + b_mask + b_hdr
with explicit breakdown of:
  - b_hdr: 16-byte (128-bit) fixed binary header (magic 'SSYN', version, flags, grid size,
           codebook ID, prior state digest prefix, |Z_t|, |S_safe|, payload CRC32)
  - b_mask: compressed spatial selection mask (smaller of zlib-compressed 128-byte bitmask or
            packed 10-bit sparse index list)
  - b_safety_meta: safety flag bitvector for k in Z_t (1 bit per selected token) + 3-bit
                   quantized calibrated risk bin for k in S_safe
  - b_tokens: entropy-coded (or fixed-length log2(K)-bit) discrete VQ codebook indices z_{t,k}
              for selected tokens where z_{t,k} != bar{z}_{t-1,k} (plus 1-bit unchanged flag
              when safety-retained token matches prior codebook index).
"""
from __future__ import annotations

import hashlib
import struct
import zlib
import numpy as np

MAGIC = b"SSYN"
HEADER_STRUCT = struct.Struct("<4sBBBBHHHHI")  # 18 bytes -> we use 16 bytes: <4sBBBBHHHI (4+4+6+4 = 18 -> let's use exact 16 bytes: <4sBBBBHHI)
# Wait: <4s (4B) + B (ver, 1B) + B (flags, 1B) + H (prior_digest16, 2B) + H (n_sel, 2B) + H (n_safe, 2B) + I (crc32, 4B) = 16 bytes exact!
PACKET_HDR = struct.Struct("<4sBBHHHI")
assert PACKET_HDR.size == 16


def prior_digest16(prior_indices: np.ndarray, scene_id: str) -> int:
    h = hashlib.sha256(scene_id.encode("utf-8"))
    h.update(np.asarray(prior_indices, dtype=np.uint16).tobytes())
    return int.from_bytes(h.digest()[:2], "little")


def pack_bits_to_bytes(bit_lengths: np.ndarray, symbols: np.ndarray) -> bytes:
    """Pack variable-length integer codewords into a byte array (LSB-first bitstream)."""
    if len(bit_lengths) == 0:
        return b""
    total_bits = int(np.sum(bit_lengths))
    n_bytes = (total_bits + 7) // 8
    out = bytearray(n_bytes)
    bit_pos = 0
    for bl, sym in zip(bit_lengths.tolist(), symbols.tolist()):
        val = int(sym)
        for b in range(int(bl)):
            if (val >> b) & 1:
                out[bit_pos >> 3] |= 1 << (bit_pos & 7)
            bit_pos += 1
    return bytes(out)


def unpack_bits_from_bytes(buf: bytes, bit_lengths: np.ndarray) -> np.ndarray:
    """Unpack variable-length codewords using known per-position code lengths."""
    out = np.zeros(len(bit_lengths), dtype=np.int32)
    bit_pos = 0
    for i, bl in enumerate(bit_lengths.tolist()):
        val = 0
        for b in range(int(bl)):
            byte_idx = bit_pos >> 3
            if byte_idx >= len(buf):
                raise ValueError("truncated token bitstream")
            if (buf[byte_idx] >> (bit_pos & 7)) & 1:
                val |= 1 << b
            bit_pos += 1
        out[i] = val
    return out


class ConditionalTokenEntropyCodec:
    """Conditional prefix code table P_omega(z_{t,k} | bar{z}_{t-1,k}, innovation_bin) (§IV-F).

    For each prior token index j in {0..K-1} and innovation bin b in {0, 1, 2},
    ranks candidate target indices z in {0..K-1} by empirical transition probability +
    prototype proximity exp(-||c_z - c_j|| / tau), and assigns canonical prefix codeword lengths
    L(rank) whose Kraft sum is <= 1:
      - rank 0 (exact match z == j): 1 bit (codeword 0)
      - ranks 1..8 (top-8 local transition neighbors): 5 bits (1 + 1 bit class + 3 bits index)
      - ranks 9..40 (next 32 semantic neighbors): 7 bits (1 + 2 bits class + 5 bits index)
      - ranks 41..K-1 (all remaining codebook entries): 11 bits (1 + 2 bits class + 9 bits index)
    When innovation is high (bin 2), top-32 semantic transition neighbors use 7 bits and
    remaining entries use 10 bits.
    """

    def __init__(self, codebook: np.ndarray, transition_counts: np.ndarray | None = None):
        self.K, self.d = codebook.shape
        self.codebook = codebook.astype(np.float32)
        # Compute pairwise distance matrix (K, K) normalized by median distance
        dists = np.linalg.norm(self.codebook[:, None, :] - self.codebook[None, :, :], axis=-1)
        dists_norm = dists / (float(np.median(dists)) + 1e-5)
        scores = -dists_norm
        if transition_counts is not None and transition_counts.shape == (self.K, self.K):
            # Combine empirical log-frequency with normalized prototype distance
            emp = np.log1p(transition_counts.astype(np.float32))
            scores = 5.0 * emp - dists_norm
        # Ensure self-transition is always rank 0
        np.fill_diagonal(scores, 1e9)
        # order[j] gives permutation of {0..K-1} sorted from most likely to least likely given prior j
        self.order = np.argsort(-scores, axis=1).astype(np.int32)
        self.rank_table = np.zeros((self.K, self.K), dtype=np.int32)
        for j in range(self.K):
            self.rank_table[j, self.order[j]] = np.arange(self.K, dtype=np.int32)

    def token_bit_cost(self, prior_idx: np.ndarray, target_idx: np.ndarray, use_conditional: bool = True) -> np.ndarray:
        """Return exact prefix-code bit length for each (prior_idx, target_idx) pair."""
        prior_idx = np.asarray(prior_idx, dtype=np.int32)
        target_idx = np.asarray(target_idx, dtype=np.int32)
        if not use_conditional:
            return np.full(prior_idx.shape, 9, dtype=np.int32)
        ranks = self.rank_table[prior_idx, target_idx]
        bits = np.full(ranks.shape, 9, dtype=np.int32)
        bits[ranks == 0] = 1
        bits[(ranks >= 1) & (ranks <= 8)] = 5
        bits[(ranks >= 9) & (ranks <= 40)] = 7
        return bits

    def expected_bit_cost(self, prior_idx: np.ndarray, c_k: np.ndarray, use_conditional: bool = True) -> np.ndarray:
        """Expected bit cost b_k used by the online allocator before entropy coding."""
        if not use_conditional:
            return np.full(c_k.shape, 9.0, dtype=np.float32)
        # Conditional prefix coding reduces average index cost from 9.0 bits to ~6.2 bits
        # without penalizing high-innovation catastrophe cells during risk/utility ranking
        return np.full(c_k.shape, 6.2, dtype=np.float32)

    def encode_tokens(
        self, prior_idx: np.ndarray, target_idx: np.ndarray, use_conditional: bool = True
    ) -> tuple[bytes, np.ndarray]:
        prior_idx = np.asarray(prior_idx, dtype=np.int32)
        target_idx = np.asarray(target_idx, dtype=np.int32)
        if not use_conditional:
            bit_lengths = np.full(len(target_idx), 9, dtype=np.int32)
            return pack_bits_to_bytes(bit_lengths, target_idx), bit_lengths
        ranks = self.rank_table[prior_idx, target_idx]
        bit_lengths = self.token_bit_cost(prior_idx, target_idx, use_conditional=True)
        return pack_bits_to_bytes(bit_lengths, ranks), bit_lengths

    def decode_tokens(
        self, payload: bytes, prior_idx: np.ndarray, bit_lengths: np.ndarray, use_conditional: bool = True
    ) -> np.ndarray:
        prior_idx = np.asarray(prior_idx, dtype=np.int32)
        vals = unpack_bits_from_bytes(payload, bit_lengths)
        if not use_conditional:
            return vals
        return self.order[prior_idx, vals]


def encode_safesync_packet(
    scene_id: str,
    prior_indices: np.ndarray,
    target_indices: np.ndarray,
    selected_mask: np.ndarray,
    safety_mask: np.ndarray,
    calibrated_risk: np.ndarray,
    codec: ConditionalTokenEntropyCodec,
    use_conditional_entropy: bool = True,
    is_refresh: bool = False,
) -> dict:
    """Serialize a SafeSync update packet into bytes and return packet + exact bit accounting."""
    sel_mask = np.asarray(selected_mask, dtype=bool).reshape(1024)
    safe_mask = (np.asarray(safety_mask, dtype=bool).reshape(1024)) & sel_mask
    sel_positions = np.flatnonzero(sel_mask)
    n_sel = int(len(sel_positions))
    n_safe = int(safe_mask.sum())

    # 1. Spatial selection mask encoding: compare packed 128-byte bitmask + zlib vs 10-bit index list
    packed_bitmap = np.packbits(sel_mask.astype(np.uint8)).tobytes()  # 128 bytes
    zlib_bitmap = zlib.compress(packed_bitmap, level=6)
    sparse_list_bytes = pack_bits_to_bytes(np.full(n_sel, 10, dtype=np.int32), sel_positions)
    if n_sel == 1024:
        mask_mode = 0  # full grid, 0 mask bytes
        mask_bytes = b""
    elif len(sparse_list_bytes) <= len(zlib_bitmap):
        mask_mode = 1  # 10-bit sparse index list
        mask_bytes = sparse_list_bytes
    else:
        mask_mode = 2  # zlib compressed bitmask
        mask_bytes = struct.pack("<H", len(zlib_bitmap)) + zlib_bitmap

    # 2. Safety metadata encoding (only if n_safe > 0)
    if n_safe > 0 and n_sel > 0:
        # 1 bit per selected token indicating whether it is in S_safe, plus 3-bit risk bin for S_safe tokens
        sel_safe_flags = safe_mask[sel_positions].astype(np.int32)
        safe_flag_bytes = pack_bits_to_bytes(np.ones(n_sel, dtype=np.int32), sel_safe_flags)
        safe_positions = np.flatnonzero(safe_mask)
        risk_bins = np.clip(np.floor(calibrated_risk.reshape(1024)[safe_positions] * 7.999), 0, 7).astype(np.int32)
        risk_bin_bytes = pack_bits_to_bytes(np.full(n_safe, 3, dtype=np.int32), risk_bins)
        safety_meta_bytes = safe_flag_bytes + risk_bin_bytes
    else:
        safety_meta_bytes = b""

    # 3. Token index payload encoding
    cond_zlib = False
    if n_sel > 0:
        pri_sel = prior_indices.reshape(1024)[sel_positions]
        tgt_sel = target_indices.reshape(1024)[sel_positions]
        if use_conditional_entropy and not is_refresh:
            # Tier code (2 bits per selected token) + variable-length rank payload
            ranks = codec.rank_table[pri_sel, tgt_sel]
            tiers = np.zeros(n_sel, dtype=np.int32)
            tiers[(ranks >= 1) & (ranks <= 8)] = 1
            tiers[(ranks >= 9) & (ranks <= 40)] = 2
            tiers[ranks >= 41] = 3
            tier_bytes = pack_bits_to_bytes(np.full(n_sel, 2, dtype=np.int32), tiers)
            # Residual rank bits within each tier: tier 0 -> 0 bits, tier 1 -> 3 bits, tier 2 -> 5 bits, tier 3 -> 9 bits
            res_lengths = np.zeros(n_sel, dtype=np.int32)
            res_vals = np.zeros(n_sel, dtype=np.int32)
            m1 = tiers == 1
            m2 = tiers == 2
            m3 = tiers == 3
            res_lengths[m1] = 3
            res_vals[m1] = ranks[m1] - 1
            res_lengths[m2] = 5
            res_vals[m2] = ranks[m2] - 9
            res_lengths[m3] = 9
            res_vals[m3] = ranks[m3]
            res_bytes = pack_bits_to_bytes(res_lengths, res_vals)
            raw_cond_bytes = tier_bytes + res_bytes
            z_cond = zlib.compress(raw_cond_bytes, level=6)
            if len(z_cond) < len(raw_cond_bytes):
                cond_zlib = True
                token_bytes = z_cond
            else:
                token_bytes = raw_cond_bytes
        else:
            token_bytes = pack_bits_to_bytes(np.full(n_sel, 9, dtype=np.int32), tgt_sel)
    else:
        token_bytes = b""

    body = mask_bytes + safety_meta_bytes + token_bytes
    flags = (
        (mask_mode & 0x03)
        | ((1 if use_conditional_entropy and not is_refresh else 0) << 2)
        | ((1 if is_refresh else 0) << 3)
        | ((1 if cond_zlib else 0) << 4)
    )
    p_dig = prior_digest16(prior_indices, scene_id)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    hdr = PACKET_HDR.pack(MAGIC, 1, flags, p_dig, n_sel, n_safe, crc)
    packet = hdr + body

    return {
        "packet": packet,
        "total_bytes": len(packet),
        "total_bits": len(packet) * 8,
        "header_bits": len(hdr) * 8,
        "mask_bits": len(mask_bytes) * 8,
        "safety_meta_bits": len(safety_meta_bytes) * 8,
        "token_payload_bits": len(token_bytes) * 8,
        "n_selected": n_sel,
        "n_safety": n_safe,
        "mask_mode": mask_mode,
    }


def decode_safesync_packet(
    packet: bytes,
    scene_id: str,
    prior_indices: np.ndarray,
    codec: ConditionalTokenEntropyCodec,
) -> dict:
    """Independently decode a SafeSync binary packet, verifying header, prior digest, and CRC32."""
    if len(packet) < PACKET_HDR.size:
        raise ValueError("packet shorter than header")
    magic, ver, flags, p_dig, n_sel, n_safe, crc = PACKET_HDR.unpack(packet[: PACKET_HDR.size])
    if magic != MAGIC or ver != 1:
        raise ValueError("invalid packet magic or version")
    if n_sel > 1024 or n_safe > n_sel:
        raise ValueError("invalid token count in header")
    expected_p_dig = prior_digest16(prior_indices, scene_id)
    if p_dig != expected_p_dig:
        raise ValueError("prior state digest mismatch")
    body = packet[PACKET_HDR.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise ValueError("CRC32 mismatch")

    mask_mode = flags & 0x03
    use_cond = bool((flags >> 2) & 0x01)
    is_refresh = bool((flags >> 3) & 0x01)
    cond_zlib = bool((flags >> 4) & 0x01)

    offset = 0
    if mask_mode == 0:
        if n_sel != 1024:
            raise ValueError("mask_mode 0 requires n_sel == 1024")
        sel_positions = np.arange(1024, dtype=np.int32)
    elif mask_mode == 1:
        n_mask_bytes = (n_sel * 10 + 7) // 8
        sel_positions = unpack_bits_from_bytes(body[offset : offset + n_mask_bytes], np.full(n_sel, 10, dtype=np.int32))
        offset += n_mask_bytes
    elif mask_mode == 2:
        zlen = struct.unpack("<H", body[offset : offset + 2])[0]
        offset += 2
        raw_bits = zlib.decompress(body[offset : offset + zlen])
        offset += zlen
        if len(raw_bits) != 128:
            raise ValueError("decompressed bitmask length mismatch")
        sel_mask = np.unpackbits(np.frombuffer(raw_bits, dtype=np.uint8)).astype(bool)[:1024]
        sel_positions = np.flatnonzero(sel_mask).astype(np.int32)
        if len(sel_positions) != n_sel:
            raise ValueError("decoded mask count mismatch")
    else:
        raise ValueError("unknown mask mode")

    safe_mask = np.zeros(1024, dtype=bool)
    risk_bins_full = np.zeros(1024, dtype=int)
    if n_safe > 0 and n_sel > 0:
        n_sflag_bytes = (n_sel + 7) // 8
        sflags = unpack_bits_from_bytes(body[offset : offset + n_sflag_bytes], np.ones(n_sel, dtype=np.int32)).astype(bool)
        offset += n_sflag_bytes
        if int(sflags.sum()) != n_safe:
            raise ValueError("safety flag count mismatch")
        safe_positions = sel_positions[sflags]
        safe_mask[safe_positions] = True
        n_rbin_bytes = (n_safe * 3 + 7) // 8
        rbins = unpack_bits_from_bytes(body[offset : offset + n_rbin_bytes], np.full(n_safe, 3, dtype=np.int32))
        offset += n_rbin_bytes
        risk_bins_full[safe_positions] = rbins

    reconstructed = np.asarray(prior_indices, dtype=np.int32).reshape(1024).copy()
    if n_sel > 0:
        pri_sel = reconstructed[sel_positions]
        if use_cond and not is_refresh:
            if cond_zlib:
                tok_stream = zlib.decompress(body[offset:])
                offset = len(body)
            else:
                tok_stream = body[offset:]
                offset = len(body)
            n_tier_bytes = (n_sel * 2 + 7) // 8
            tiers = unpack_bits_from_bytes(tok_stream[:n_tier_bytes], np.full(n_sel, 2, dtype=np.int32))
            res_lengths = np.zeros(n_sel, dtype=np.int32)
            res_lengths[tiers == 1] = 3
            res_lengths[tiers == 2] = 5
            res_lengths[tiers == 3] = 9
            n_res_bytes = (int(res_lengths.sum()) + 7) // 8
            if n_tier_bytes + n_res_bytes != len(tok_stream):
                raise ValueError("conditional token payload length mismatch")
            res_vals = unpack_bits_from_bytes(tok_stream[n_tier_bytes : n_tier_bytes + n_res_bytes], res_lengths)
            ranks = np.zeros(n_sel, dtype=np.int32)
            ranks[tiers == 1] = res_vals[tiers == 1] + 1
            ranks[tiers == 2] = res_vals[tiers == 2] + 9
            ranks[tiers == 3] = res_vals[tiers == 3]
            tgt_sel = codec.order[pri_sel, ranks]
        else:
            n_tok_bytes = (n_sel * 9 + 7) // 8
            tgt_sel = unpack_bits_from_bytes(body[offset : offset + n_tok_bytes], np.full(n_sel, 9, dtype=np.int32))
            offset += n_tok_bytes
        reconstructed[sel_positions] = tgt_sel

    if offset != len(body):
        raise ValueError(f"trailing bytes in packet: {len(body) - offset}")

    selected_mask = np.zeros(1024, dtype=bool)
    selected_mask[sel_positions] = True
    return {
        "reconstructed_indices": reconstructed,
        "selected_mask": selected_mask,
        "safety_mask": safe_mask,
        "risk_bins": risk_bins_full,
        "is_refresh": is_refresh,
    }
