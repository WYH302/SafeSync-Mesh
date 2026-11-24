"""Provenance-ready binary packet wire format and 2D causal spatio-temporal conditional prefix codec for SafeSync-Mesh (§III-B, §IV-C).

Every transmitted differential update Z_t is serialized into a verifiable binary packet:
  R_t = sum_{k in Z_t} b_k + b_safe + b_mask + b_hdr
with explicit wire-level accounting of:
  - b_hdr: 16-byte (128-bit) fixed binary header (magic 'SSYN', version, flags,
           16-bit prior state digest, |Z_t|, |S_safe|, payload CRC32)
  - b_mask: compressed spatial selection mask (smaller of zlib-compressed 128-byte bitmask or
            packed 10-bit sparse index list with S_safe positions ordered first)
  - b_safe: compact safety metadata (0 flag bits in header-ordered sparse mode; compact local index
            or bitvector in bitmap mode, plus 3-bit quantized calibrated risk bin for k in S_safe)
  - b_tok: 2D causal spatio-temporal conditional prefix-coded VQ codebook indices
           P_omega(z_{t,k} | bar{z}_{t-1,k}, hat{z}_{t,N(k)}) using a Kraft-tight instantaneous
           5-tier prefix code (1 / 4 / 7 / 10 / 13 bits) or fixed 9-bit unconditional indices.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
import numpy as np

MAGIC = b"SSYN"
# <4s (4B) + B (ver, 1B) + B (flags, 1B) + H (prior_digest16, 2B) + H (n_sel, 2B) + H (n_safe, 2B) + I (crc32, 4B) = 16 bytes exact
PACKET_HDR = struct.Struct("<4sBBHHHI")
assert PACKET_HDR.size == 16


def prior_digest16(prior_indices: np.ndarray, scene_id: str) -> int:
    h = hashlib.sha256(scene_id.encode("utf-8"))
    h.update(np.asarray(prior_indices, dtype=np.uint16).tobytes())
    return int.from_bytes(h.digest()[:2], "little")


def pack_bits_to_bytes(bit_lengths: np.ndarray, symbols: np.ndarray) -> bytes:
    """Fast LSB-first variable-length integer packer."""
    if len(bit_lengths) == 0:
        return b""
    acc = 0
    nbits = 0
    out = bytearray()
    for bl, sym in zip(bit_lengths.tolist(), symbols.tolist()):
        bl_i = int(bl)
        if bl_i > 0:
            acc |= (int(sym) & ((1 << bl_i) - 1)) << nbits
            nbits += bl_i
            while nbits >= 8:
                out.append(acc & 0xFF)
                acc >>= 8
                nbits -= 8
    if nbits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_bits_from_bytes(buf: bytes, bit_lengths: np.ndarray) -> np.ndarray:
    """Fast LSB-first variable-length integer unpacker."""
    n = len(bit_lengths)
    out = np.zeros(n, dtype=np.int32)
    if n == 0:
        return out
    acc = 0
    nbits = 0
    byte_idx = 0
    buf_len = len(buf)
    for i, bl in enumerate(bit_lengths.tolist()):
        bl_i = int(bl)
        while nbits < bl_i:
            if byte_idx >= buf_len:
                raise ValueError("truncated token bitstream")
            acc |= buf[byte_idx] << nbits
            byte_idx += 1
            nbits += 8
        out[i] = acc & ((1 << bl_i) - 1)
        acc >>= bl_i
        nbits -= bl_i
    return out


def _encode_mask_2d_arith(mask_flat: np.ndarray) -> bytes:
    """Lossless 16-context 2D causal binary arithmetic coder (W, NW, N, NE neighbors) for 32x32 masks."""
    ml = np.asarray(mask_flat, dtype=np.uint8).ravel().tolist()
    c0 = [1] * 16
    c1 = [1] * 16
    lo = 0
    hi = 0xFFFFFFFF
    out = bytearray()
    out_append = out.append
    for r in range(32):
        base = r << 5
        pbase = base - 32
        w = 0
        for c in range(32):
            if r > 0:
                nw = ml[pbase + c - 1] if c > 0 else 0
                n = ml[pbase + c]
                ne = ml[pbase + c + 1] if c < 31 else 0
                ctx = (w << 3) | (nw << 2) | (n << 1) | ne
            else:
                ctx = w << 3
            a0 = c0[ctx]
            a1 = c1[ctx]
            split = lo + ((hi - lo + 1) * a0) // (a0 + a1) - 1
            if split < lo:
                split = lo
            elif split >= hi:
                split = hi - 1
            bit = ml[base + c]
            if bit == 0:
                hi = split
                a0 += 2
            else:
                lo = split + 1
                a1 += 2
            if a0 + a1 > 512:
                a0 = (a0 + 1) >> 1
                a1 = (a1 + 1) >> 1
            c0[ctx] = a0
            c1[ctx] = a1
            w = bit
            while (lo ^ hi) < 0x01000000:
                out_append((lo >> 24) & 0xFF)
                lo = (lo << 8) & 0xFFFFFFFF
                hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
    for shift in (24, 16, 8, 0):
        out_append((lo >> shift) & 0xFF)
    return bytes(out)


def _decode_mask_2d_arith(buf: bytes) -> np.ndarray:
    """Exact bit-for-bit inverse of _encode_mask_2d_arith for a 32x32 binary mask."""
    if len(buf) < 4:
        raise ValueError("truncated 2D arithmetic mask bitstream")
    ml = [0] * 1024
    c0 = [1] * 16
    c1 = [1] * 16
    lo = 0
    hi = 0xFFFFFFFF
    code = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
    pos = 4
    blen = len(buf)
    for r in range(32):
        base = r << 5
        pbase = base - 32
        w = 0
        for c in range(32):
            if r > 0:
                nw = ml[pbase + c - 1] if c > 0 else 0
                n = ml[pbase + c]
                ne = ml[pbase + c + 1] if c < 31 else 0
                ctx = (w << 3) | (nw << 2) | (n << 1) | ne
            else:
                ctx = w << 3
            a0 = c0[ctx]
            a1 = c1[ctx]
            split = lo + ((hi - lo + 1) * a0) // (a0 + a1) - 1
            if split < lo:
                split = lo
            elif split >= hi:
                split = hi - 1
            if code <= split:
                bit = 0
                hi = split
                a0 += 2
            else:
                bit = 1
                lo = split + 1
                a1 += 2
            ml[base + c] = bit
            if a0 + a1 > 512:
                a0 = (a0 + 1) >> 1
                a1 = (a1 + 1) >> 1
            c0[ctx] = a0
            c1[ctx] = a1
            w = bit
            while (lo ^ hi) < 0x01000000:
                lo = (lo << 8) & 0xFFFFFFFF
                hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
                nxt = buf[pos] if pos < blen else 0
                pos += 1
                code = ((code << 8) | nxt) & 0xFFFFFFFF
    return np.array(ml, dtype=bool)


def _encode_safe_flags_arith(flags: np.ndarray) -> bytes:
    """Lossless 2-context (previous-flag Markov) adaptive binary arithmetic coder for sparse S_safe indicators."""
    fl = np.asarray(flags, dtype=np.uint8).ravel().tolist()
    c0 = [4, 2]
    c1 = [1, 2]
    lo = 0
    hi = 0xFFFFFFFF
    out = bytearray()
    out_append = out.append
    prev = 0
    for bit in fl:
        a0 = c0[prev]
        a1 = c1[prev]
        split = lo + ((hi - lo + 1) * a0) // (a0 + a1) - 1
        if split < lo:
            split = lo
        elif split >= hi:
            split = hi - 1
        if bit == 0:
            hi = split
            a0 += 2
        else:
            lo = split + 1
            a1 += 2
        if a0 + a1 > 512:
            a0 = (a0 + 1) >> 1
            a1 = (a1 + 1) >> 1
        c0[prev] = a0
        c1[prev] = a1
        prev = bit
        while (lo ^ hi) < 0x01000000:
            out_append((lo >> 24) & 0xFF)
            lo = (lo << 8) & 0xFFFFFFFF
            hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
    for shift in (24, 16, 8, 0):
        out_append((lo >> shift) & 0xFF)
    return bytes(out)


def _decode_safe_flags_arith(buf: bytes, n_sel: int) -> np.ndarray:
    """Exact bit-for-bit inverse of _encode_safe_flags_arith."""
    if len(buf) < 4:
        raise ValueError("truncated arithmetic safety flag stream")
    fl = [0] * n_sel
    c0 = [4, 2]
    c1 = [1, 2]
    lo = 0
    hi = 0xFFFFFFFF
    code = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
    pos = 4
    blen = len(buf)
    prev = 0
    for i in range(n_sel):
        a0 = c0[prev]
        a1 = c1[prev]
        split = lo + ((hi - lo + 1) * a0) // (a0 + a1) - 1
        if split < lo:
            split = lo
        elif split >= hi:
            split = hi - 1
        if code <= split:
            bit = 0
            hi = split
            a0 += 2
        else:
            bit = 1
            lo = split + 1
            a1 += 2
        fl[i] = bit
        if a0 + a1 > 512:
            a0 = (a0 + 1) >> 1
            a1 = (a1 + 1) >> 1
        c0[prev] = a0
        c1[prev] = a1
        prev = bit
        while (lo ^ hi) < 0x01000000:
            lo = (lo << 8) & 0xFFFFFFFF
            hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
            nxt = buf[pos] if pos < blen else 0
            pos += 1
            code = ((code << 8) | nxt) & 0xFFFFFFFF
    return np.array(fl, dtype=bool)


def _pack_ranks_range(ranks: list[int], raw_bits: int = 9) -> bytes:
    """Adaptive 6-tier range coder (rANS-equivalent entropy efficiency) for ranks in {0..K-1}.

    Tiers: 0 -> {0} (0b), 1 -> {1..4} (2b), 2 -> {5..20} (4b), 3 -> {21..84} (6b), 4 -> {85..212} (7b), 5 -> escape (raw_bits).
    Tier symbols are adaptively range-coded; exact sub-rank offsets are bit-packed losslessly.
    """
    if not ranks:
        return b""
    counts = [4, 4, 4, 3, 2, 2]
    lo = 0
    hi = 0xFFFFFFFF
    tier_out = bytearray()
    sub_lens: list[int] = []
    sub_vals: list[int] = []
    for r in ranks:
        if r == 0:
            t = 0
        elif r <= 4:
            t = 1
            sub_lens.append(2); sub_vals.append(r - 1)
        elif r <= 20:
            t = 2
            sub_lens.append(4); sub_vals.append(r - 5)
        elif r <= 84:
            t = 3
            sub_lens.append(6); sub_vals.append(r - 21)
        elif r <= 212 and raw_bits >= 8:
            t = 4
            sub_lens.append(7); sub_vals.append(r - 85)
        else:
            t = 5
            sub_lens.append(raw_bits); sub_vals.append(r)
        tot = sum(counts)
        cum = sum(counts[:t])
        rng = hi - lo + 1
        new_lo = lo + (rng * cum) // tot
        new_hi = lo + (rng * (cum + counts[t])) // tot - 1
        lo, hi = new_lo, max(new_lo, new_hi)
        counts[t] += 3
        if sum(counts) > 1024:
            counts = [max(1, (x + 1) >> 1) for x in counts]
        while (lo ^ hi) < 0x01000000:
            tier_out.append((lo >> 24) & 0xFF)
            lo = (lo << 8) & 0xFFFFFFFF
            hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
    for shift in (24, 16, 8, 0):
        tier_out.append((lo >> shift) & 0xFF)
    sub_bytes = pack_bits_to_bytes(np.array(sub_lens, dtype=np.int32), np.array(sub_vals, dtype=np.int32))
    return struct.pack("<H", len(tier_out)) + bytes(tier_out) + sub_bytes


def _unpack_ranks_range(buf: bytes, n_sel: int, raw_bits: int = 9) -> list[int]:
    """Exact bit-for-bit inverse of _pack_ranks_range."""
    if n_sel == 0:
        return []
    tlen = struct.unpack("<H", buf[:2])[0]
    tbuf = buf[2 : 2 + tlen]
    sbuf = buf[2 + tlen :]
    if len(tbuf) < 4:
        raise ValueError("truncated range-coded rank stream")
    counts = [4, 4, 4, 3, 2, 2]
    lo = 0
    hi = 0xFFFFFFFF
    code = (tbuf[0] << 24) | (tbuf[1] << 16) | (tbuf[2] << 8) | tbuf[3]
    pos = 4
    blen = len(tbuf)
    tiers: list[int] = []
    sub_lens: list[int] = []
    for _ in range(n_sel):
        tot = sum(counts)
        rng = hi - lo + 1
        cum = 0
        chosen_t = 5
        chosen_lo = lo
        chosen_hi = hi
        for t in range(6):
            n_lo = lo + (rng * cum) // tot
            n_hi = lo + (rng * (cum + counts[t])) // tot - 1
            if n_hi < n_lo:
                n_hi = n_lo
            if code <= n_hi or t == 5:
                chosen_t = t
                chosen_lo, chosen_hi = n_lo, n_hi
                break
            cum += counts[t]
        tiers.append(chosen_t)
        if chosen_t == 1:
            sub_lens.append(2)
        elif chosen_t == 2:
            sub_lens.append(4)
        elif chosen_t == 3:
            sub_lens.append(6)
        elif chosen_t == 4:
            sub_lens.append(7)
        elif chosen_t == 5:
            sub_lens.append(raw_bits)
        lo, hi = chosen_lo, chosen_hi
        counts[chosen_t] += 3
        if sum(counts) > 1024:
            counts = [max(1, (x + 1) >> 1) for x in counts]
        while (lo ^ hi) < 0x01000000:
            lo = (lo << 8) & 0xFFFFFFFF
            hi = ((hi << 8) | 0xFF) & 0xFFFFFFFF
            nxt = tbuf[pos] if pos < blen else 0
            pos += 1
            code = ((code << 8) | nxt) & 0xFFFFFFFF
    sub_vals = unpack_bits_from_bytes(sbuf, np.array(sub_lens, dtype=np.int32)).tolist() if sub_lens else []
    si = 0
    ranks: list[int] = []
    for t in tiers:
        if t == 0:
            ranks.append(0)
        elif t == 1:
            ranks.append(sub_vals[si] + 1); si += 1
        elif t == 2:
            ranks.append(sub_vals[si] + 5); si += 1
        elif t == 3:
            ranks.append(sub_vals[si] + 21); si += 1
        elif t == 4:
            ranks.append(sub_vals[si] + 85); si += 1
        else:
            ranks.append(sub_vals[si]); si += 1
    return ranks


def _pack_st_ranks(ranks: list[int], raw_bits: int = 9) -> bytes:
    """Pack ranks in {0..K-1} with Kraft-tight instantaneous prefix code (LSB-first):
      - r == 0:        '0'           (1 bit)
      - 1 <= r <= 4:   '1','0' + 2b  (4 bits)
      - 5 <= r <= 20:  '1','1','0' + 4b (7 bits)
      - 21 <= r <= 84: '1','1','1','0' + 6b (10 bits)
      - r >= 85:       '1','1','1','1' + raw_bits (4 + raw_bits)
    """
    acc = 0
    nbits = 0
    out = bytearray()
    esc_len = 4 + int(raw_bits)
    for r in ranks:
        if r == 0:
            # bit 0 -> 0
            nbits += 1
        elif r <= 4:
            # bits: 1, 0, (r-1)[0:2] -> value = 0b01 | ((r-1) << 2)
            acc |= (0b01 | ((r - 1) << 2)) << nbits
            nbits += 4
        elif r <= 20:
            # bits: 1, 1, 0, (r-5)[0:4] -> value = 0b011 | ((r-5) << 3)
            acc |= (0b011 | ((r - 5) << 3)) << nbits
            nbits += 7
        elif r <= 84:
            # bits: 1, 1, 1, 0, (r-21)[0:6] -> value = 0b0111 | ((r-21) << 4)
            acc |= (0b0111 | ((r - 21) << 4)) << nbits
            nbits += 10
        else:
            # bits: 1, 1, 1, 1, r[0:raw_bits] -> value = 0b1111 | (r << 4)
            acc |= (0b1111 | (r << 4)) << nbits
            nbits += esc_len
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_st_ranks(buf: bytes, n_sel: int, raw_bits: int = 9) -> list[int]:
    """Decode n_sel instantaneously prefix-coded ranks in {0..K-1}."""
    acc = 0
    nbits = 0
    byte_idx = 0
    buf_len = len(buf)
    ranks: list[int] = []
    rb = int(raw_bits)
    rmask = (1 << rb) - 1

    def _ensure(k: int) -> None:
        nonlocal acc, nbits, byte_idx
        while nbits < k:
            if byte_idx >= buf_len:
                raise ValueError("truncated spatio-temporal rank bitstream")
            acc |= buf[byte_idx] << nbits
            byte_idx += 1
            nbits += 8

    for _ in range(n_sel):
        ones = 0
        while ones < 4:
            _ensure(1)
            b = acc & 1
            acc >>= 1
            nbits -= 1
            if b == 0:
                break
            ones += 1
        if ones == 0:
            ranks.append(0)
        elif ones == 1:
            _ensure(2)
            v = acc & 0x3
            acc >>= 2
            nbits -= 2
            ranks.append(v + 1)
        elif ones == 2:
            _ensure(4)
            v = acc & 0xF
            acc >>= 4
            nbits -= 4
            ranks.append(v + 5)
        elif ones == 3:
            _ensure(6)
            v = acc & 0x3F
            acc >>= 6
            nbits -= 6
            ranks.append(v + 21)
        else:
            _ensure(rb)
            v = acc & rmask
            acc >>= rb
            nbits -= rb
            ranks.append(v)
    return ranks


class ConditionalTokenEntropyCodec:
    """2D Causal Spatio-Temporal Conditional Prefix Codec P_omega(z_{t,k} | bar{z}_{t-1,k}, hat{z}_{t,N(k)}).

    Maintains both:
      - Temporal transition rank table (prior prototype -> current prototype), and
      - Spatial transition rank table (causal 2D left/top neighbor prototype -> current prototype).
    """

    def __init__(
        self,
        codebook: np.ndarray,
        transition_counts: np.ndarray | None = None,
        spatial_counts: np.ndarray | None = None,
    ):
        self.K, self.d = codebook.shape
        self.raw_bits = max(1, (int(self.K) - 1).bit_length())
        self.codebook = codebook.astype(np.float32)
        dists = np.linalg.norm(self.codebook[:, None, :] - self.codebook[None, :, :], axis=-1)
        dists_norm = dists / (float(np.median(dists)) + 1e-5)

        scores_t = -dists_norm
        if transition_counts is not None and transition_counts.shape == (self.K, self.K):
            emp_t = np.log1p(transition_counts.astype(np.float32))
            scores_t = 5.0 * emp_t - dists_norm
        np.fill_diagonal(scores_t, 1e9)
        self.order = np.argsort(-scores_t, axis=1).astype(np.int32)
        self.rank_table = np.zeros((self.K, self.K), dtype=np.int32)
        for j in range(self.K):
            self.rank_table[j, self.order[j]] = np.arange(self.K, dtype=np.int32)

        scores_s = -dists_norm
        if spatial_counts is not None and spatial_counts.shape == (self.K, self.K):
            emp_s = np.log1p(spatial_counts.astype(np.float32))
            scores_s = 5.0 * emp_s - dists_norm
        np.fill_diagonal(scores_s, 1e9)
        self.spatial_order = np.argsort(-scores_s, axis=1).astype(np.int32)
        self.spatial_rank_table = np.zeros((self.K, self.K), dtype=np.int32)
        for j in range(self.K):
            self.spatial_rank_table[j, self.spatial_order[j]] = np.arange(self.K, dtype=np.int32)

    def token_bit_cost(self, prior_idx: np.ndarray, target_idx: np.ndarray, use_conditional: bool = True) -> np.ndarray:
        """Return per-token prefix-code bit length for temporal transitions."""
        prior_idx = np.asarray(prior_idx, dtype=np.int32)
        target_idx = np.asarray(target_idx, dtype=np.int32)
        if not use_conditional:
            return np.full(prior_idx.shape, self.raw_bits, dtype=np.int32)
        ranks = self.rank_table[prior_idx, target_idx]
        esc_bits = 4 + self.raw_bits
        return np.where(
            ranks == 0, 1, np.where(ranks <= 4, 4, np.where(ranks <= 20, 7, np.where(ranks <= 84, 10, esc_bits)))
        ).astype(np.int32)

    def expected_bit_cost(self, prior_idx: np.ndarray, c_k: np.ndarray, use_conditional: bool = True) -> np.ndarray:
        """Expected bit cost b_k used by the online allocator before entropy coding."""
        if not use_conditional:
            return np.full(c_k.shape, float(self.raw_bits), dtype=np.float32)
        return np.full(c_k.shape, float(5.85 * (self.raw_bits / 9.0)), dtype=np.float32)

    def encode_tokens(
        self, prior_idx: np.ndarray, target_idx: np.ndarray, use_conditional: bool = True
    ) -> tuple[bytes, np.ndarray]:
        prior_idx = np.asarray(prior_idx, dtype=np.int32)
        target_idx = np.asarray(target_idx, dtype=np.int32)
        if not use_conditional:
            bit_lengths = np.full(len(target_idx), self.raw_bits, dtype=np.int32)
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
    force_sparse_mask: bool = False,
    use_2d_arith_mask: bool = True,
    use_range_ranks: bool = False,
    include_risk_bins: bool = False,
    superblock_safe: bool = False,
) -> dict:
    """Serialize a SafeSync-Mesh update packet into verifiable bytes and return exact wire bit accounting."""
    sel_mask = np.asarray(selected_mask, dtype=bool).reshape(1024)
    safe_mask = (np.asarray(safety_mask, dtype=bool).reshape(1024)) & sel_mask
    sel_positions = np.flatnonzero(sel_mask).astype(np.int32)
    n_sel = int(len(sel_positions))
    n_safe = int(safe_mask.sum())

    # 1. Spatial selection mask encoding (auto-pick strictly smallest wire representation)
    safe_positions = np.flatnonzero(safe_mask).astype(np.int32)
    util_positions = np.flatnonzero(sel_mask & (~safe_mask)).astype(np.int32)
    ordered_sparse_pos = np.concatenate([safe_positions, util_positions]) if n_safe > 0 else sel_positions

    sparse_list_bytes = pack_bits_to_bytes(np.full(n_sel, 10, dtype=np.int32), ordered_sparse_pos)
    if n_sel == 1024:
        mask_mode = 0  # full grid, 0 mask bytes
        mask_bytes = b""
    elif force_sparse_mask:
        mask_mode = 1
        mask_bytes = sparse_list_bytes
    else:
        packed_bitmap = np.packbits(sel_mask.astype(np.uint8)).tobytes()  # 128 bytes
        zlib_bitmap = zlib.compress(packed_bitmap, level=6)
        cand_modes: list[tuple[int, bytes]] = [
            (1, sparse_list_bytes),
            (2, struct.pack("<H", len(zlib_bitmap)) + zlib_bitmap),
        ]
        if use_2d_arith_mask:
            arith_mask = _encode_mask_2d_arith(sel_mask)
            cand_modes.append((3, struct.pack("<H", len(arith_mask)) + arith_mask))
        mask_mode, mask_bytes = min(cand_modes, key=lambda x: len(x[1]))

    # 2. Compact lossless safety metadata encoding (only if n_safe > 0)
    safe_flag_arith = False
    if n_safe > 0 and n_sel > 0:
        if include_risk_bins:
            risk_bins = np.clip(np.floor(calibrated_risk.reshape(1024)[safe_positions] * 7.999), 0, 7).astype(np.int32)
            risk_bin_bytes = pack_bits_to_bytes(np.full(n_safe, 3, dtype=np.int32), risk_bins)
        else:
            risk_bin_bytes = b""
        if mask_mode == 1:
            safety_meta_bytes = risk_bin_bytes
        elif superblock_safe:
            # 4x4 super-block S_safe indicator (8x8 = 64 super-blocks, encoded only over active super-blocks!)
            sb_active = sel_mask.reshape(8, 4, 8, 4).any(axis=(1, 3)).reshape(64)
            sb_flags = safe_mask.reshape(8, 4, 8, 4).any(axis=(1, 3)).reshape(64)[sb_active].astype(np.int32)
            raw_sb = pack_bits_to_bytes(np.ones(len(sb_flags), dtype=np.int32), sb_flags)
            arith_sb = _encode_safe_flags_arith(sb_flags)
            if len(arith_sb) + 1 < len(raw_sb) and len(arith_sb) <= 255:
                safe_flag_arith = True
                safety_meta_bytes = struct.pack("<B", len(arith_sb)) + arith_sb + risk_bin_bytes
            else:
                safety_meta_bytes = raw_sb + risk_bin_bytes
        else:
            sel_safe_flags = safe_mask[sel_positions].astype(np.int32)
            raw_sflag = pack_bits_to_bytes(np.ones(n_sel, dtype=np.int32), sel_safe_flags)
            arith_sflag = _encode_safe_flags_arith(sel_safe_flags)
            if len(arith_sflag) + 1 < len(raw_sflag) and len(arith_sflag) <= 255:
                safe_flag_arith = True
                safety_meta_bytes = struct.pack("<B", len(arith_sflag)) + arith_sflag + risk_bin_bytes
            else:
                safety_meta_bytes = raw_sflag + risk_bin_bytes
    else:
        safety_meta_bytes = b""

    # 3. Token index payload encoding (always raster-ordered sel_positions for 2D causal decoding!)
    mode_zlib = False
    if n_sel > 0:
        pri_flat = np.asarray(prior_indices, dtype=np.int32).reshape(1024)
        tgt_flat = np.asarray(target_indices, dtype=np.int32).reshape(1024)
        if use_conditional_entropy and not is_refresh:
            grid = pri_flat.copy()
            dec_seen = np.zeros(1024, dtype=bool)
            rt_tbl = codec.rank_table
            rs_tbl = codec.spatial_rank_table
            use_sp_list = np.zeros(n_sel, dtype=np.uint8)
            ranks_list: list[int] = []
            for idx_i, pos in enumerate(sel_positions.tolist()):
                cc = pos & 31
                pri_k = int(pri_flat[pos])
                tgt_k = int(tgt_flat[pos])
                if cc > 0 and dec_seen[pos - 1]:
                    sp_k = int(grid[pos - 1])
                elif pos >= 32 and dec_seen[pos - 32]:
                    sp_k = int(grid[pos - 32])
                elif cc > 0:
                    sp_k = int(grid[pos - 1])
                else:
                    sp_k = pri_k
                rt_v = int(rt_tbl[pri_k, tgt_k])
                rs_v = int(rs_tbl[sp_k, tgt_k])
                if rs_v < rt_v:
                    use_sp_list[idx_i] = 1
                    ranks_list.append(rs_v)
                else:
                    ranks_list.append(rt_v)
                grid[pos] = tgt_k
                dec_seen[pos] = True
            mode_raw = np.packbits(use_sp_list).tobytes()
            mode_z = zlib.compress(mode_raw, level=6)
            if len(mode_z) + 2 < len(mode_raw):
                mode_zlib = True
                mode_bytes = struct.pack("<H", len(mode_z)) + mode_z
            else:
                mode_bytes = mode_raw
            rank_bytes = (
                _pack_ranks_range(ranks_list, raw_bits=codec.raw_bits)
                if use_range_ranks
                else _pack_st_ranks(ranks_list, raw_bits=codec.raw_bits)
            )
            token_bytes = mode_bytes + rank_bytes
        else:
            token_bytes = pack_bits_to_bytes(np.full(n_sel, codec.raw_bits, dtype=np.int32), tgt_flat[sel_positions])
    else:
        token_bytes = b""

    body = mask_bytes + safety_meta_bytes + token_bytes
    flags = (
        (mask_mode & 0x03)
        | ((1 if use_conditional_entropy and not is_refresh else 0) << 2)
        | ((1 if is_refresh else 0) << 3)
        | ((1 if mode_zlib else 0) << 4)
        | ((1 if safe_flag_arith else 0) << 5)
        | ((1 if (use_range_ranks and use_conditional_entropy and not is_refresh) else 0) << 6)
        | ((1 if include_risk_bins else 0) << 7)
    )
    p_dig = prior_digest16(prior_indices, scene_id)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    ver_num = 2 if (superblock_safe and mask_mode != 1) else 1
    hdr = PACKET_HDR.pack(MAGIC, ver_num, flags, p_dig, n_sel, n_safe, crc)
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
    """Independently decode a SafeSync-Mesh binary packet, verifying header, prior digest, and CRC32."""
    if len(packet) < PACKET_HDR.size:
        raise ValueError("packet shorter than header")
    magic, ver, flags, p_dig, n_sel, n_safe, crc = PACKET_HDR.unpack(packet[: PACKET_HDR.size])
    if magic != MAGIC or ver not in (1, 2):
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
    mode_zlib = bool((flags >> 4) & 0x01)
    safe_flag_arith = bool((flags >> 5) & 0x01)
    use_range_ranks = bool((flags >> 6) & 0x01)
    include_risk_bins = bool((flags >> 7) & 0x01)

    offset = 0
    if mask_mode == 0:
        if n_sel != 1024:
            raise ValueError("mask_mode 0 requires n_sel == 1024")
        raw_positions = np.arange(1024, dtype=np.int32)
        sel_positions = raw_positions
    elif mask_mode == 1:
        n_mask_bytes = (n_sel * 10 + 7) // 8
        raw_positions = unpack_bits_from_bytes(body[offset : offset + n_mask_bytes], np.full(n_sel, 10, dtype=np.int32))
        offset += n_mask_bytes
        sel_positions = np.sort(raw_positions)
    elif mask_mode == 2:
        zlen = struct.unpack("<H", body[offset : offset + 2])[0]
        offset += 2
        raw_bits = zlib.decompress(body[offset : offset + zlen])
        offset += zlen
        if len(raw_bits) != 128:
            raise ValueError("decompressed bitmask length mismatch")
        sel_mask_arr = np.unpackbits(np.frombuffer(raw_bits, dtype=np.uint8)).astype(bool)[:1024]
        raw_positions = np.flatnonzero(sel_mask_arr).astype(np.int32)
        if len(raw_positions) != n_sel:
            raise ValueError("decoded mask count mismatch")
        sel_positions = raw_positions
    elif mask_mode == 3:
        alen = struct.unpack("<H", body[offset : offset + 2])[0]
        offset += 2
        sel_mask_arr = _decode_mask_2d_arith(body[offset : offset + alen])
        offset += alen
        raw_positions = np.flatnonzero(sel_mask_arr).astype(np.int32)
        if len(raw_positions) != n_sel:
            raise ValueError("decoded 2D arithmetic mask count mismatch")
        sel_positions = raw_positions
    else:
        raise ValueError("unknown mask mode")

    safe_mask = np.zeros(1024, dtype=bool)
    risk_bins_full = np.zeros(1024, dtype=int)
    if n_safe > 0 and n_sel > 0:
        if mask_mode == 1:
            safe_positions = np.sort(raw_positions[:n_safe])
            safe_mask[safe_positions] = True
        elif ver == 2:
            sel_bool_1024 = np.zeros(1024, dtype=bool)
            sel_bool_1024[sel_positions] = True
            sb_active = sel_bool_1024.reshape(8, 4, 8, 4).any(axis=(1, 3)).reshape(64)
            n_sb_act = int(sb_active.sum())
            if safe_flag_arith:
                alen_s = body[offset]
                offset += 1
                sb_flags = _decode_safe_flags_arith(body[offset : offset + alen_s], n_sb_act)
                offset += alen_s
            else:
                n_sb_bytes = (n_sb_act + 7) // 8
                sb_flags = unpack_bits_from_bytes(body[offset : offset + n_sb_bytes], np.ones(n_sb_act, dtype=np.int32)).astype(bool)
                offset += n_sb_bytes
            sb_grid = np.zeros(64, dtype=bool)
            sb_grid[sb_active] = sb_flags
            sb_expanded = np.repeat(np.repeat(sb_grid.reshape(8, 8), 4, axis=0), 4, axis=1).reshape(1024)
            safe_mask = sb_expanded & sel_bool_1024
            safe_positions = np.flatnonzero(safe_mask).astype(np.int32)
        elif safe_flag_arith:
            alen_s = body[offset]
            offset += 1
            sflags = _decode_safe_flags_arith(body[offset : offset + alen_s], n_sel)
            offset += alen_s
            if int(sflags.sum()) != n_safe:
                raise ValueError("arithmetic safety flag count mismatch")
            safe_positions = sel_positions[sflags]
            safe_mask[safe_positions] = True
        else:
            n_sflag_bytes = (n_sel + 7) // 8
            sflags = unpack_bits_from_bytes(body[offset : offset + n_sflag_bytes], np.ones(n_sel, dtype=np.int32)).astype(bool)
            offset += n_sflag_bytes
            if int(sflags.sum()) != n_safe:
                raise ValueError("safety flag count mismatch")
            safe_positions = sel_positions[sflags]
            safe_mask[safe_positions] = True
        if include_risk_bins:
            n_rbin_bytes = (n_safe * 3 + 7) // 8
            rbins = unpack_bits_from_bytes(body[offset : offset + n_rbin_bytes], np.full(n_safe, 3, dtype=np.int32))
            offset += n_rbin_bytes
            risk_bins_full[safe_positions] = rbins

    reconstructed = np.asarray(prior_indices, dtype=np.int32).reshape(1024).copy()
    if n_sel > 0:
        if use_cond and not is_refresh:
            if mode_zlib:
                mlen = struct.unpack("<H", body[offset : offset + 2])[0]
                offset += 2
                mode_raw = zlib.decompress(body[offset : offset + mlen])
                offset += mlen
            else:
                n_mode_bytes = (n_sel + 7) // 8
                mode_raw = body[offset : offset + n_mode_bytes]
                offset += n_mode_bytes
            use_sp = np.unpackbits(np.frombuffer(mode_raw, dtype=np.uint8))[:n_sel]
            ranks = (
                _unpack_ranks_range(body[offset:], n_sel, raw_bits=codec.raw_bits)
                if use_range_ranks
                else _unpack_st_ranks(body[offset:], n_sel, raw_bits=codec.raw_bits)
            )
            offset = len(body)

            dec_seen = np.zeros(1024, dtype=bool)
            ord_t = codec.order
            ord_s = codec.spatial_order
            for idx_i, pos in enumerate(sel_positions.tolist()):
                cc = pos & 31
                pri_k = int(reconstructed[pos])
                if use_sp[idx_i]:
                    if cc > 0 and dec_seen[pos - 1]:
                        sp_k = int(reconstructed[pos - 1])
                    elif pos >= 32 and dec_seen[pos - 32]:
                        sp_k = int(reconstructed[pos - 32])
                    elif cc > 0:
                        sp_k = int(reconstructed[pos - 1])
                    else:
                        sp_k = pri_k
                    tgt_k = int(ord_s[sp_k, ranks[idx_i]])
                else:
                    tgt_k = int(ord_t[pri_k, ranks[idx_i]])
                reconstructed[pos] = tgt_k
                dec_seen[pos] = True
        else:
            n_tok_bytes = (n_sel * codec.raw_bits + 7) // 8
            tgt_sel = unpack_bits_from_bytes(body[offset : offset + n_tok_bytes], np.full(n_sel, codec.raw_bits, dtype=np.int32))
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
