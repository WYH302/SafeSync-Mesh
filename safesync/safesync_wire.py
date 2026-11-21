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


def _pack_st_ranks(ranks: list[int]) -> bytes:
    """Pack ranks in {0..511} with Kraft-tight instantaneous prefix code (LSB-first):
      - r == 0:        '0'           (1 bit)
      - 1 <= r <= 4:   '1','0' + 2b  (4 bits)
      - 5 <= r <= 20:  '1','1','0' + 4b (7 bits)
      - 21 <= r <= 84: '1','1','1','0' + 6b (10 bits)
      - 85 <= r <= 511:'1','1','1','1' + 9b (13 bits)
    """
    acc = 0
    nbits = 0
    out = bytearray()
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
            # bits: 1, 1, 1, 1, r[0:9] -> value = 0b1111 | (r << 4)
            acc |= (0b1111 | (r << 4)) << nbits
            nbits += 13
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_st_ranks(buf: bytes, n_sel: int) -> list[int]:
    """Decode n_sel instantaneously prefix-coded ranks in {0..511}."""
    acc = 0
    nbits = 0
    byte_idx = 0
    buf_len = len(buf)
    ranks: list[int] = []

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
            _ensure(9)
            v = acc & 0x1FF
            acc >>= 9
            nbits -= 9
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
            return np.full(prior_idx.shape, 9, dtype=np.int32)
        ranks = self.rank_table[prior_idx, target_idx]
        return np.where(
            ranks == 0, 1, np.where(ranks <= 4, 4, np.where(ranks <= 20, 7, np.where(ranks <= 84, 10, 13)))
        ).astype(np.int32)

    def expected_bit_cost(self, prior_idx: np.ndarray, c_k: np.ndarray, use_conditional: bool = True) -> np.ndarray:
        """Expected bit cost b_k used by the online allocator before entropy coding."""
        if not use_conditional:
            return np.full(c_k.shape, 9.0, dtype=np.float32)
        return np.full(c_k.shape, 5.85, dtype=np.float32)

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
    force_sparse_mask: bool = False,
) -> dict:
    """Serialize a SafeSync-Mesh update packet into verifiable bytes and return exact wire bit accounting."""
    sel_mask = np.asarray(selected_mask, dtype=bool).reshape(1024)
    safe_mask = (np.asarray(safety_mask, dtype=bool).reshape(1024)) & sel_mask
    sel_positions = np.flatnonzero(sel_mask).astype(np.int32)
    n_sel = int(len(sel_positions))
    n_safe = int(safe_mask.sum())

    # 1. Spatial selection mask encoding
    safe_positions = np.flatnonzero(safe_mask).astype(np.int32)
    util_positions = np.flatnonzero(sel_mask & (~safe_mask)).astype(np.int32)
    ordered_sparse_pos = np.concatenate([safe_positions, util_positions]) if n_safe > 0 else sel_positions

    packed_bitmap = np.packbits(sel_mask.astype(np.uint8)).tobytes()  # 128 bytes
    zlib_bitmap = zlib.compress(packed_bitmap, level=6)
    sparse_list_bytes = pack_bits_to_bytes(np.full(n_sel, 10, dtype=np.int32), ordered_sparse_pos)
    if n_sel == 1024:
        mask_mode = 0  # full grid, 0 mask bytes
        mask_bytes = b""
    elif force_sparse_mask or len(sparse_list_bytes) <= len(zlib_bitmap) + 2:
        mask_mode = 1  # 10-bit sparse index list (S_safe ordered first -> 0 safety flag bytes!)
        mask_bytes = sparse_list_bytes
    else:
        mask_mode = 2  # zlib compressed 32x32 bitmap
        mask_bytes = struct.pack("<H", len(zlib_bitmap)) + zlib_bitmap

    # 2. Compact safety metadata encoding (only if n_safe > 0)
    compact_safe_idx = False
    if n_safe > 0 and n_sel > 0:
        risk_bins = np.clip(np.floor(calibrated_risk.reshape(1024)[safe_positions] * 7.999), 0, 7).astype(np.int32)
        risk_bin_bytes = pack_bits_to_bytes(np.full(n_safe, 3, dtype=np.int32), risk_bins)
        if mask_mode == 1:
            safety_meta_bytes = risk_bin_bytes
        else:
            sel_safe_flags = safe_mask[sel_positions].astype(np.int32)
            raw_sflag = pack_bits_to_bytes(np.ones(n_sel, dtype=np.int32), sel_safe_flags)
            idx_bits = max(1, int(np.ceil(np.log2(max(2, n_sel)))))
            local_safe_idx = np.flatnonzero(sel_safe_flags).astype(np.int32)
            sparse_sflag = pack_bits_to_bytes(np.full(n_safe, idx_bits, dtype=np.int32), local_safe_idx)
            if len(sparse_sflag) < len(raw_sflag):
                compact_safe_idx = True
                safety_meta_bytes = sparse_sflag + risk_bin_bytes
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
            rank_bytes = _pack_st_ranks(ranks_list)
            token_bytes = mode_bytes + rank_bytes
        else:
            token_bytes = pack_bits_to_bytes(np.full(n_sel, 9, dtype=np.int32), tgt_flat[sel_positions])
    else:
        token_bytes = b""

    body = mask_bytes + safety_meta_bytes + token_bytes
    flags = (
        (mask_mode & 0x03)
        | ((1 if use_conditional_entropy and not is_refresh else 0) << 2)
        | ((1 if is_refresh else 0) << 3)
        | ((1 if mode_zlib else 0) << 4)
        | ((1 if compact_safe_idx else 0) << 5)
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
    """Independently decode a SafeSync-Mesh binary packet, verifying header, prior digest, and CRC32."""
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
    mode_zlib = bool((flags >> 4) & 0x01)
    compact_safe_idx = bool((flags >> 5) & 0x01)

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
    else:
        raise ValueError("unknown mask mode")

    safe_mask = np.zeros(1024, dtype=bool)
    risk_bins_full = np.zeros(1024, dtype=int)
    if n_safe > 0 and n_sel > 0:
        if mask_mode == 1:
            safe_positions = np.sort(raw_positions[:n_safe])
            safe_mask[safe_positions] = True
        elif compact_safe_idx:
            idx_bits = max(1, int(np.ceil(np.log2(max(2, n_sel)))))
            n_sidx_bytes = (n_safe * idx_bits + 7) // 8
            local_sidx = unpack_bits_from_bytes(body[offset : offset + n_sidx_bytes], np.full(n_safe, idx_bits, dtype=np.int32))
            offset += n_sidx_bytes
            safe_positions = sel_positions[local_sidx]
            safe_mask[safe_positions] = True
        else:
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
            ranks = _unpack_st_ranks(body[offset:], n_sel)
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
