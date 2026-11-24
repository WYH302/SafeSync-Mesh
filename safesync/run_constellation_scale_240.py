#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7: Large-Scale 24 -> 60 -> 120 -> 240-Satellite Starlink Shell-1 (550 km, 53 deg)
Walker-Delta Discrete-Event ISL Scalability Study & Proposition 3 Empirical Verification.

Evaluates true 3D Keplerian orbital propagation + 4-laser optical ISL routing (2 intra-plane +
2 dynamic cross-plane ISLs with Earth-limb clearance) across 4 constellation tiers:
  - 24  sats (3x8,  6 EO sensors,  N_task =  200, ISL slice =  48 kb/ep)
  - 60  sats (6x10, 15 EO sensors, N_task =  500, ISL slice = 120 kb/ep)
  - 120 sats (12x10, 30 EO sensors, N_task = 1000, ISL slice = 240 kb/ep)
  - 240 sats (12x20, 60 EO sensors, N_task = 2000, ISL slice = 480 kb/ep)

Also empirically verifies Proposition 3:
  1. Closed-form Bloom filter false-positive bound p_fp(m, k, n) <= (1 - exp(-k*n/m))^k
     against live simulator Bloom hash collisions across m in {32, 64, 128, 256, 512} bits
     at 240-sat peak surge (N_task = 2,000, 4 regional hubs).
  2. Closed-form sustainable surge capacity N_max = floor(C_ISL / (r_eff * eta_dup(H)))
     explaining why non-provenance protocols degrade with constellation diameter H ~ O(sqrt(N_sat))
     while SafeSync-Mesh maintains near-constant O(1) duplicate factor eta_dup ~ 1.03.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MAIN_RESULTS_JSON = RESULTS_DIR / "safesync_r36_complete_results.json"
OUT_JSON = RESULTS_DIR / "scheme1_constellation_scale240.json"

R_EARTH_KM = 6371.0
MU_EARTH = 398600.4418
OMEGA_EARTH = 7.2921159e-5
SEEDS = [42, 1337, 2026]

CONSUMERS_LL = [
    (37.7, -122.4),  # D0: West US / California
    (19.4, -99.1),   # D1: Central Mexico / Americas
    (-0.9, 119.9),   # D2: Sulawesi / Asia-Pacific
    (38.7, 23.7),    # D3: Mediterranean / EMEA
]


def compute_root_id(sat_id: int, scene_id: str, pass_idx: int = 1) -> int:
    h = hashlib.sha256(f"S{sat_id}:{scene_id}:pass{pass_idx}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


class BloomFilterRKS:
    """Bit-level k-hash (default k=2) Bloom filter carried inside the RKS sketch."""

    def __init__(self, n_bits: int = 32, k_hashes: int = 2):
        self.n_bits = int(n_bits)
        self.k_hashes = int(k_hashes)
        self.mask = 0
        self.unique_ids: set[int] = set()

    @property
    def n_unique(self) -> int:
        return len(self.unique_ids)

    def _hashes(self, root_id: int) -> Tuple[int, ...]:
        h = hashlib.blake2b(int(root_id).to_bytes(4, "little"), digest_size=4 * self.k_hashes).digest()
        return tuple(
            int.from_bytes(h[4 * i : 4 * (i + 1)], "little") % self.n_bits
            for i in range(self.k_hashes)
        )

    def add(self, root_id: int) -> None:
        for b in self._hashes(root_id):
            self.mask |= (1 << b)
        self.unique_ids.add(int(root_id))

    def maybe_contains(self, root_id: int) -> bool:
        return all(bool(self.mask & (1 << b)) for b in self._hashes(root_id))

    def clear(self) -> None:
        self.mask = 0
        self.unique_ids.clear()


def _eci_position(r_km: float, inc_rad: float, raan_rad: float, arg_lat_rad: float) -> np.ndarray:
    cos_u, sin_u = math.cos(arg_lat_rad), math.sin(arg_lat_rad)
    cos_o, sin_o = math.cos(raan_rad), math.sin(raan_rad)
    cos_i, sin_i = math.cos(inc_rad), math.sin(inc_rad)
    return np.array([
        r_km * (cos_o * cos_u - sin_o * sin_u * cos_i),
        r_km * (sin_o * cos_u + cos_o * sin_u * cos_i),
        r_km * (sin_u * sin_i),
    ], dtype=np.float64)


def _los_clears_earth(p1: np.ndarray, p2: np.ndarray, min_alt_km: float = 50.0) -> bool:
    d = p2 - p1
    denom = float(np.dot(d, d))
    if denom < 1e-6:
        return True
    t = float(np.clip(-np.dot(p1, d) / denom, 0.0, 1.0))
    closest = p1 + t * d
    return float(np.linalg.norm(closest)) >= (R_EARTH_KM + min_alt_km)


def build_scaled_walker_delta_plan(
    n_planes: int,
    sats_per_plane: int,
    n_sensors: int,
    n_epochs: int = 20,
    epoch_s: float = 15.0,
) -> Tuple[List[Dict], Dict]:
    """Propagate a T = P * S Walker-Delta (550 km, 53 deg, F=1) relay shell with realistic
    4-laser optical ISL terminals (2 permanent intra-plane + 2 dynamic nearest cross-plane ISLs
    on adjacent orbital planes satisfying <= 4,500 km range and >= 50 km Earth-limb clearance)."""
    n_relays = n_planes * sats_per_plane
    r_relay = R_EARTH_KM + 550.0
    n_relay = math.sqrt(MU_EARTH / (r_relay ** 3))
    inc_relay = math.radians(53.0)

    r_sens = R_EARTH_KM + 520.0
    n_sens = math.sqrt(MU_EARTH / (r_sens ** 3))
    inc_sens = math.radians(97.5)

    epochs_plan: List[Dict] = []
    intra_counts, inter_counts = [], []

    for e in range(n_epochs):
        t = e * epoch_s
        relay_pos = np.zeros((n_relays, 3), dtype=np.float64)
        for p in range(n_planes):
            raan = 2.0 * math.pi * p / n_planes
            for s in range(sats_per_plane):
                idx = p * sats_per_plane + s
                u = 2.0 * math.pi * s / sats_per_plane + 2.0 * math.pi * p / n_relays + n_relay * t
                relay_pos[idx] = _eci_position(r_relay, inc_relay, raan, u)

        # Distribute EO sensing satellites evenly across sun-synchronous orbital planes
        n_sens_planes = max(2, n_sensors // 3)
        sens_pos = np.zeros((n_sensors, 3), dtype=np.float64)
        for s_i in range(n_sensors):
            p = s_i % n_sens_planes
            k = s_i // n_sens_planes
            s_in_p = max(1, (n_sensors + n_sens_planes - 1) // n_sens_planes)
            raan = 2.0 * math.pi * p / n_sens_planes + 0.35
            u = 2.0 * math.pi * k / s_in_p + 0.5 * p + n_sens * t
            sens_pos[s_i] = _eci_position(r_sens, inc_sens, raan, u)

        cons_pos = np.zeros((len(CONSUMERS_LL), 3), dtype=np.float64)
        for c_i, (lat_d, lon_d) in enumerate(CONSUMERS_LL):
            lat = math.radians(lat_d)
            lon = math.radians(lon_d) + OMEGA_EARTH * t
            cons_pos[c_i] = np.array([
                R_EARTH_KM * math.cos(lat) * math.cos(lon),
                R_EARTH_KM * math.cos(lat) * math.sin(lon),
                R_EARTH_KM * math.sin(lat),
            ])

        adj: Dict[int, set] = {i: set() for i in range(n_relays)}
        n_intra = 0
        for p in range(n_planes):
            for s in range(sats_per_plane):
                u = p * sats_per_plane + s
                v = p * sats_per_plane + ((s + 1) % sats_per_plane)
                adj[u].add(v)
                adj[v].add(u)
                n_intra += 1

        # Cross-plane laser terminals: each satellite pairs with nearest valid neighbor in plane (p+1)%P
        inter_edges = set()
        for p in range(n_planes):
            p_next = (p + 1) % n_planes
            next_nodes = [p_next * sats_per_plane + s2 for s2 in range(sats_per_plane)]
            next_pts = relay_pos[next_nodes]
            for s in range(sats_per_plane):
                u = p * sats_per_plane + s
                dists = np.linalg.norm(next_pts - relay_pos[u : u + 1], axis=1)
                for order_idx in np.argsort(dists)[:2]:
                    dist = float(dists[order_idx])
                    v = next_nodes[int(order_idx)]
                    if dist <= 4500.0 and _los_clears_earth(relay_pos[u], relay_pos[v], min_alt_km=50.0):
                        adj[u].add(v)
                        adj[v].add(u)
                        inter_edges.add((min(u, v), max(u, v)))
                        break

        adj_list = {i: sorted(adj[i]) for i in range(n_relays)}
        intra_counts.append(n_intra)
        inter_counts.append(len(inter_edges))

        sens_entry = []
        for s_i in range(n_sensors):
            d = np.linalg.norm(relay_pos - sens_pos[s_i : s_i + 1], axis=1)
            sens_entry.append([int(x) for x in np.argsort(d)[:2]])
        cons_exit = []
        for c_i in range(len(CONSUMERS_LL)):
            d = np.linalg.norm(relay_pos - cons_pos[c_i : c_i + 1], axis=1)
            cons_exit.append([int(x) for x in np.argsort(d)[:2]])

        # Precompute route lookup cache for this epoch
        epochs_plan.append({
            "epoch": e,
            "t_s": t,
            "adj": adj_list,
            "sens_entry": sens_entry,
            "cons_exit": cons_exit,
            "route_cache": {},
        })

    # Measure mean primary hop count across all (epoch, sensor, consumer) pairs
    all_hops = []
    for ep in epochs_plan:
        for s_i in range(n_sensors):
            for c_i in range(len(CONSUMERS_LL)):
                p1, _ = _get_cached_routes(ep, s_i, c_i)
                all_hops.append(max(1, len(p1) - 1))

    topo_stats = {
        "n_relays": n_relays,
        "walker_notation": f"{n_relays}/{n_planes}/1",
        "n_sensors": n_sensors,
        "n_consumers": len(CONSUMERS_LL),
        "mean_intra_isls": round(float(np.mean(intra_counts)), 1),
        "mean_inter_isls": round(float(np.mean(inter_counts)), 1),
        "mean_isl_hops": round(float(np.mean(all_hops)), 2),
        "max_isl_hops": int(np.max(all_hops)),
    }
    return epochs_plan, topo_stats


def _bfs_shortest_path(adj: Dict[int, List[int]], src: int, dst: int, blocked_edges: set | None = None) -> List[int]:
    if src == dst:
        return [src]
    blocked = blocked_edges or set()
    q = deque([(src, [src])])
    vis = {src}
    while q:
        u, path = q.popleft()
        for v in adj[u]:
            edge = (min(u, v), max(u, v))
            if v in vis or edge in blocked:
                continue
            if v == dst:
                return path + [v]
            vis.add(v)
            q.append((v, path + [v]))
    return []


def _get_cached_routes(ep_plan: Dict, sensor_id: int, consumer_id: int) -> Tuple[List[int], List[int]]:
    key = (sensor_id, consumer_id)
    if key in ep_plan["route_cache"]:
        return ep_plan["route_cache"][key]
    adj = ep_plan["adj"]
    s0, s1 = ep_plan["sens_entry"][sensor_id]
    d0, d1 = ep_plan["cons_exit"][consumer_id]
    p1 = _bfs_shortest_path(adj, s0, d0)
    if not p1:
        p1 = [s0, d0]
    used_edges = {(min(p1[i], p1[i + 1]), max(p1[i], p1[i + 1])) for i in range(len(p1) - 1)}
    p2 = _bfs_shortest_path(adj, s1, d1, blocked_edges=used_edges)
    if not p2:
        p2 = _bfs_shortest_path(adj, s0, d1, blocked_edges=used_edges)
    if not p2:
        p2 = p1[:]
    ep_plan["route_cache"][key] = (p1, p2)
    return p1, p2


def simulate_scaled_constellation(
    protocol_cfg: Dict,
    n_tasks: int,
    n_sensors: int,
    contact_plan: List[Dict],
    rng: np.random.Generator,
    m_full: Dict,
    isl_slice_kb_per_epoch: float,
    bloom_bits: int = 32,
    bloom_window_epochs: int = 8,
    bloom_k_hashes: int = 2,
) -> Dict:
    """Discrete-event packet-level multi-hop LEO ISL simulator with real Bloom RKS and exact
    collision accounting for Proposition 3 verification."""
    n_epochs = len(contact_plan)
    epoch_s = 15.0
    n_cons = len(CONSUMERS_LL)
    is_stateless = bool(protocol_cfg.get("is_stateless", False))
    has_rks = bool(protocol_cfg.get("has_rks", False))
    has_prov = bool(protocol_cfg.get("has_provenance", False))
    has_prio = bool(protocol_cfg.get("has_priority", False))
    has_safe = bool(protocol_cfg.get("has_safety_reserve", False))

    rate_base_kb = float(protocol_cfg["rate_kb"])
    rate_refresh_kb = float(m_full["rate_kb_per_frame"])
    rks_kb = (bloom_bits + 32) / 1000.0 if has_rks else 0.0
    single_known = float(protocol_cfg["known_f1"])
    single_sync = float(protocol_cfg["sync_f1"])
    single_urec = float(np.clip((100.0 - float(protocol_cfg["u_miss"])) / 100.0, 0.0, 1.0))
    safe_bit_frac = float(protocol_cfg.get("safe_bit_frac", 0.28)) if has_safe else 0.0
    extra_decode_s = float(protocol_cfg.get("extra_decode_s", 0.0))

    tasks = []
    for tid in range(n_tasks):
        gen_ep = int(rng.integers(0, min(15, n_epochs - 2)))
        sens = int(tid % n_sensors)
        cons = int((tid // n_sensors) % n_cons)
        aligned = bool(rng.random() < 0.76)
        root_id = compute_root_id(sens + 1, f"scale_task_{tid:05d}", pass_idx=1)
        p1, p2 = _get_cached_routes(contact_plan[gen_ep], sens, cons)
        hops_p1 = max(1, len(p1) - 1)
        hops_p2 = max(1, len(p2) - 1)
        # Multi-path cross-plane relay branching grows with ISL hop depth H
        n_bkp_raw = max(1, int(round(0.42 * hops_p2 + (0.60 if rng.random() < 0.65 else 0.0))))
        jitter_s = float(rng.uniform(0.4, 2.0))
        re_rand = float(rng.random())
        # Per-hop optical ISL transient micro-fade (2.5% per hop, capped at 25%)
        p1_fade = bool(rng.random() < min(0.25, 0.025 * hops_p1))
        tasks.append({
            "tid": tid, "root_id": root_id, "gen_ep": gen_ep, "sens": sens, "cons": cons,
            "aligned": aligned, "hops_p1": hops_p1, "hops_p2": hops_p2, "n_bkp_raw": n_bkp_raw,
            "jitter_s": jitter_s, "re_rand": re_rand, "p1_fade": p1_fade,
            "delivered_main": False, "delivered_safe": False,
            "confirm_time_s": float(n_epochs * epoch_s), "copies_received": 0,
            "used_refresh": False, "prior_mismatch_fail": False,
        })

    cons_blooms = [BloomFilterRKS(n_bits=bloom_bits, k_hashes=bloom_k_hashes) for _ in range(n_cons)]
    relay_bloom_snap = [BloomFilterRKS(n_bits=bloom_bits, k_hashes=bloom_k_hashes) for _ in range(n_cons)]
    cons_seen_roots = [set() for _ in range(n_cons)]
    pending_pkts: List[Dict] = []

    total_tx_kb = 0.0
    duplicate_tx_kb = 0.0
    bloom_fp_drops = 0
    bloom_neg_queries = 0
    bloom_true_suppressions = 0
    snap_occupancies = []

    tasks_by_ep: Dict[int, List[Dict]] = {e: [] for e in range(n_epochs)}
    for tk in tasks:
        tasks_by_ep[tk["gen_ep"]].append(tk)

    for ep in range(n_epochs):
        if ep > 0 and ep % bloom_window_epochs == 0:
            for bf in cons_blooms:
                bf.clear()

        for tk in tasks_by_ep[ep]:
            use_ref = (not is_stateless) and has_rks and (not tk["aligned"])
            tk["used_refresh"] = use_ref
            if (not is_stateless) and (not has_rks) and (not tk["aligned"]):
                tk["prior_mismatch_fail"] = True
            pkt_kb = (rate_refresh_kb if use_ref else rate_base_kb) + rks_kb

            if has_prio and has_safe and not use_ref:
                safe_kb = max(0.32, pkt_kb * safe_bit_frac)
                util_kb = max(0.10, pkt_kb - safe_kb)
                pending_pkts.append({
                    "tid": tk["tid"], "root_id": tk["root_id"], "cons": tk["cons"],
                    "kind": "safe", "prio": 0, "kb_rem": safe_kb, "hops_rem": tk["hops_p1"],
                    "is_dup": False, "gen_ep": ep, "enq_ep": ep, "p1_fade": False,
                })
                pending_pkts.append({
                    "tid": tk["tid"], "root_id": tk["root_id"], "cons": tk["cons"],
                    "kind": "util", "prio": 1, "kb_rem": util_kb, "hops_rem": tk["hops_p1"],
                    "is_dup": False, "gen_ep": ep, "enq_ep": ep, "p1_fade": tk["p1_fade"],
                })
            else:
                pending_pkts.append({
                    "tid": tk["tid"], "root_id": tk["root_id"], "cons": tk["cons"],
                    "kind": "full", "prio": 1, "kb_rem": pkt_kb, "hops_rem": tk["hops_p1"],
                    "is_dup": False, "gen_ep": ep, "enq_ep": ep, "p1_fade": tk["p1_fade"],
                })

            for b_i in range(tk["n_bkp_raw"]):
                b_prio = 2 if has_prov else 1
                b_enq = ep if (not has_prov) else (ep if (b_i == 0 and tk["re_rand"] < 0.04) else ep + 1)
                pending_pkts.append({
                    "tid": tk["tid"], "root_id": tk["root_id"], "cons": tk["cons"],
                    "kind": "backup", "prio": b_prio, "kb_rem": pkt_kb, "hops_rem": tk["hops_p2"],
                    "is_dup": True, "gen_ep": ep, "enq_ep": b_enq, "p1_fade": False,
                })

        ready = [p for p in pending_pkts if p["enq_ep"] <= ep]
        future = [p for p in pending_pkts if p["enq_ep"] > ep]
        if has_prio:
            ready.sort(key=lambda p: (p["prio"], p["enq_ep"], p["tid"]))
        else:
            ready.sort(key=lambda p: (p["enq_ep"], p["tid"], p["prio"]))

        cap_rem = isl_slice_kb_per_epoch
        next_pending = list(future)
        for p in ready:
            tk = tasks[p["tid"]]
            c_id = p["cons"]

            if p["is_dup"] and has_prov and has_rks:
                is_unseen = p["root_id"] not in cons_seen_roots[c_id]
                in_bloom = relay_bloom_snap[c_id].maybe_contains(p["root_id"])
                if is_unseen:
                    bloom_neg_queries += 1
                    snap_occupancies.append(relay_bloom_snap[c_id].n_unique)
                if in_bloom:
                    if is_unseen:
                        bloom_fp_drops += 1
                    else:
                        bloom_true_suppressions += 1
                    continue
                if tk["delivered_main"] and p["enq_ep"] > p["gen_ep"]:
                    bloom_true_suppressions += 1
                    continue

            if cap_rem <= 1e-9 or p["kb_rem"] > cap_rem + 1e-9:
                p["enq_ep"] = ep + 1
                next_pending.append(p)
                continue

            cap_rem -= p["kb_rem"]
            total_tx_kb += p["kb_rem"]
            if p["is_dup"]:
                duplicate_tx_kb += p["kb_rem"]

            if p["p1_fade"]:
                continue

            arr_time_s = (ep - p["gen_ep"]) * epoch_s + 2.6 * p["hops_rem"] + extra_decode_s + tk["jitter_s"]
            if p["kind"] == "safe":
                tk["delivered_safe"] = True
                tk["confirm_time_s"] = min(tk["confirm_time_s"], arr_time_s)
                cons_blooms[c_id].add(p["root_id"])
                cons_seen_roots[c_id].add(p["root_id"])
            elif p["kind"] in ("full", "util"):
                tk["delivered_main"] = True
                if p["kind"] == "full":
                    tk["delivered_safe"] = True
                tk["confirm_time_s"] = min(tk["confirm_time_s"], arr_time_s)
                tk["copies_received"] += 1
                cons_blooms[c_id].add(p["root_id"])
                cons_seen_roots[c_id].add(p["root_id"])
            elif p["kind"] == "backup":
                if has_prov:
                    if p["root_id"] not in cons_seen_roots[c_id]:
                        tk["delivered_main"] = True
                        tk["delivered_safe"] = True
                        tk["confirm_time_s"] = min(tk["confirm_time_s"], arr_time_s)
                        tk["copies_received"] += 1
                        cons_seen_roots[c_id].add(p["root_id"])
                        cons_blooms[c_id].add(p["root_id"])
                else:
                    tk["delivered_main"] = True
                    tk["delivered_safe"] = True
                    tk["confirm_time_s"] = min(tk["confirm_time_s"], arr_time_s)
                    tk["copies_received"] += 1

        pending_pkts = next_pending
        for c_i in range(n_cons):
            relay_bloom_snap[c_i].mask = cons_blooms[c_i].mask
            relay_bloom_snap[c_i].unique_ids = set(cons_blooms[c_i].unique_ids)

    n_completed = 0
    sum_known = sum_sync = sum_urec = 0.0
    ttc_list, infl_list = [], []
    for tk in tasks:
        if tk["delivered_main"] or tk["delivered_safe"]:
            if tk["prior_mismatch_fail"]:
                eff_k, eff_s, eff_u = 36.0, 30.0, single_urec * 0.25
                ttc_list.append(float(n_epochs * epoch_s))
            else:
                n_completed += 1
                if tk["used_refresh"]:
                    eff_k = float(m_full["known_f1"])
                    eff_s = float(m_full["sync_f1"])
                    eff_u = (100.0 - float(m_full["u_miss"])) / 100.0
                elif tk["delivered_main"]:
                    eff_k, eff_s, eff_u = single_known, single_sync, single_urec
                else:
                    eff_k, eff_s, eff_u = single_known * 0.72, single_sync * 0.78, single_urec * 0.92
                ttc_list.append(tk["confirm_time_s"])
            sum_known += eff_k
            sum_sync += eff_s
            sum_urec += eff_u
        else:
            ttc_list.append(float(n_epochs * epoch_s))

        c_rx = tk["copies_received"]
        if (not has_prov) and c_rx >= 2:
            p0 = 0.78
            odds_c = (p0 / (1.0 - p0)) ** c_rx
            infl_list.append((odds_c / (1.0 + odds_c)) - p0)
        else:
            infl_list.append(0.0)

    mean_infl = float(np.mean(infl_list)) if infl_list else 0.0
    emp_fp_rate = float(bloom_fp_drops / max(1, bloom_neg_queries)) if bloom_neg_queries > 0 else 0.0
    # Proposition 3 theoretical prediction: E[(1 - exp(-k*n/m))^k] over the exact queried filter occupancies
    if snap_occupancies:
        occ_arr = np.array(snap_occupancies, dtype=np.float64)
        prop3_fp_bound = float(np.mean((1.0 - np.exp(-bloom_k_hashes * occ_arr / bloom_bits)) ** bloom_k_hashes))
        mean_occ = float(np.mean(occ_arr))
    else:
        prop3_fp_bound = 0.0
        mean_occ = 0.0

    # Proposition 3 sustainable capacity check: N_max = floor(C_ISL / (r_eff * eta_dup))
    total_cap_kb = isl_slice_kb_per_epoch * n_epochs
    r_eff = (0.76 * rate_base_kb + 0.24 * (rate_refresh_kb if (not is_stateless and has_rks) else rate_base_kb)) + rks_kb
    dup_frac = duplicate_tx_kb / max(1e-6, total_tx_kb)
    eta_dup_emp = 1.0 / max(1e-4, 1.0 - dup_frac)
    n_max_prop3 = int(math.floor(total_cap_kb / max(1e-6, r_eff * eta_dup_emp)))

    return {
        "num_tasks": n_tasks,
        "bloom_bits": bloom_bits,
        "mission_completion_pct": round(100.0 * n_completed / max(1, n_tasks), 2),
        "effective_known_f1": round(sum_known / max(1, n_tasks), 2),
        "effective_sync_f1": round(max(0.0, sum_sync / max(1, n_tasks) - (4.5 * mean_infl if not has_prov else 0.0)), 2),
        "constellation_u_miss_pct": round((1.0 - sum_urec / max(1, n_tasks)) * 100.0, 2),
        "time_to_confirm_s": round(float(np.mean(ttc_list)), 2),
        "duplicate_traffic_pct": round(100.0 * dup_frac, 2),
        "net_traffic_kb_per_scene": round(total_tx_kb / max(1, n_tasks), 2),
        "confidence_inflation": round(mean_infl, 4),
        "bloom_neg_queries": bloom_neg_queries,
        "bloom_fp_collisions": bloom_fp_drops,
        "bloom_fp_rate_empirical": round(emp_fp_rate, 4),
        "bloom_fp_bound_prop3": round(prop3_fp_bound, 4),
        "mean_bloom_occupancy_n": round(mean_occ, 2),
        "eta_dup_empirical": round(eta_dup_emp, 3),
        "n_max_prop3": n_max_prop3,
    }


def _agg(rows: List[Dict], keys: List[str]) -> Dict:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        nd = 4 if ("rate" in k or "bound" in k or "infl" in k or "eta" in k) else 2
        out[k] = round(float(np.mean(vals)), nd)
        out[f"{k}_std"] = round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), nd)
    return out


def main() -> None:
    t0 = time.perf_counter()
    main_res = json.loads(MAIN_RESULTS_JSON.read_text(encoding="utf-8"))
    per_seed_map = {int(s["seed"]): s for s in main_res["per_seed_runs"]}

    # Four Walker-Delta Starlink Shell-1 tiers (constant 2.0 kb/relay/epoch semantic slice)
    tiers = [
        {"n_relays": 24,  "P": 3,  "S": 8,  "n_sensors": 6,  "n_tasks": 200,  "cap_ep": 48.0,  "opt_bloom": 32},
        {"n_relays": 60,  "P": 6,  "S": 10, "n_sensors": 15, "n_tasks": 500,  "cap_ep": 120.0, "opt_bloom": 64},
        {"n_relays": 120, "P": 12, "S": 10, "n_sensors": 30, "n_tasks": 1000, "cap_ep": 240.0, "opt_bloom": 128},
        {"n_relays": 240, "P": 12, "S": 20, "n_sensors": 60, "n_tasks": 2000, "cap_ep": 480.0, "opt_bloom": 256},
    ]

    metrics_keys = [
        "mission_completion_pct", "effective_known_f1", "effective_sync_f1",
        "constellation_u_miss_pct", "time_to_confirm_s", "duplicate_traffic_pct",
        "net_traffic_kb_per_scene", "confidence_inflation",
        "bloom_fp_rate_empirical", "bloom_fp_bound_prop3", "eta_dup_empirical", "n_max_prop3",
    ]

    tier_results = []
    bloom_prop3_validation = []
    plan_240 = None

    print("=== [S7] 24 -> 60 -> 120 -> 240-Satellite Starlink Shell-1 Scalability Study (3 Seeds) ===")
    for cfg_t in tiers:
        plan, topo = build_scaled_walker_delta_plan(cfg_t["P"], cfg_t["S"], cfg_t["n_sensors"])
        if cfg_t["n_relays"] == 240:
            plan_240 = plan
        print(
            f"\n[Tier {cfg_t['n_relays']} sats ({topo['walker_notation']}), N_task={cfg_t['n_tasks']}, "
            f"mean_hops={topo['mean_isl_hops']}, max_hops={topo['max_isl_hops']}]"
        )

        proto_per_seed: Dict[str, List[Dict]] = {
            "safesync_mesh_scaled_rks": [],
            "safesync_mesh_fixed32": [],
            "receiver_aware_no_prov": [],
            "sender_utility_relay": [],
            "stateless_vq_flood": [],
        }

        for seed in SEEDS:
            sm = per_seed_map[seed]
            bs = sm["budget_sweep"]
            m_ss = bs["safesync"][3]
            m_util = bs["stateful_utility_only"][3]
            m_tw = bs["twist_inspired"][3]
            m_st = bs["stars_adapted"][3]
            m_full = sm["full_feature_reference"]

            specs = {
                "safesync_mesh_scaled_rks": {
                    "has_rks": True, "has_provenance": True, "has_priority": True, "has_safety_reserve": True,
                    "is_stateless": False, "rate_kb": m_ss["rate_kb_per_frame"], "known_f1": m_ss["known_f1"],
                    "sync_f1": m_ss["sync_f1"], "u_miss": m_ss["u_miss"], "extra_decode_s": 0.6,
                    "bloom_bits": cfg_t["opt_bloom"],
                },
                "safesync_mesh_fixed32": {
                    "has_rks": True, "has_provenance": True, "has_priority": True, "has_safety_reserve": True,
                    "is_stateless": False, "rate_kb": m_ss["rate_kb_per_frame"], "known_f1": m_ss["known_f1"],
                    "sync_f1": m_ss["sync_f1"], "u_miss": m_ss["u_miss"], "extra_decode_s": 0.6,
                    "bloom_bits": 32,
                },
                "receiver_aware_no_prov": {
                    "has_rks": True, "has_provenance": False, "has_priority": False, "has_safety_reserve": False,
                    "is_stateless": False, "rate_kb": m_tw["rate_kb_per_frame"], "known_f1": m_tw["known_f1"],
                    "sync_f1": m_tw["sync_f1"], "u_miss": m_tw["u_miss"], "extra_decode_s": 0.8,
                    "bloom_bits": cfg_t["opt_bloom"],
                },
                "sender_utility_relay": {
                    "has_rks": False, "has_provenance": False, "has_priority": False, "has_safety_reserve": False,
                    "is_stateless": False, "rate_kb": m_util["rate_kb_per_frame"], "known_f1": m_util["known_f1"],
                    "sync_f1": m_util["sync_f1"], "u_miss": m_util["u_miss"], "extra_decode_s": 0.8,
                    "bloom_bits": 32,
                },
                "stateless_vq_flood": {
                    "has_rks": False, "has_provenance": False, "has_priority": False, "has_safety_reserve": False,
                    "is_stateless": True, "rate_kb": m_st["rate_kb_per_frame"], "known_f1": m_st["known_f1"],
                    "sync_f1": m_st["sync_f1"], "u_miss": m_st["u_miss"], "extra_decode_s": 1.2,
                    "bloom_bits": 32,
                },
            }

            for pname, pcfg in specs.items():
                rng_p = np.random.default_rng(seed * 10000 + cfg_t["n_relays"] * 31 + cfg_t["n_tasks"])
                r = simulate_scaled_constellation(
                    pcfg, cfg_t["n_tasks"], cfg_t["n_sensors"], plan, rng_p, m_full,
                    isl_slice_kb_per_epoch=cfg_t["cap_ep"],
                    bloom_bits=pcfg["bloom_bits"],
                    bloom_window_epochs=4,  # 60s sliding RKS epoch window for mega-constellation surges
                )
                proto_per_seed[pname].append(r)

        tier_agg = {
            "topology": topo,
            "n_tasks": cfg_t["n_tasks"],
            "isl_slice_kb_per_epoch": cfg_t["cap_ep"],
            "opt_bloom_bits": cfg_t["opt_bloom"],
            "protocols": {p: _agg(proto_per_seed[p], metrics_keys) for p in proto_per_seed},
        }
        tier_results.append(tier_agg)
        ss = tier_agg["protocols"]["safesync_mesh_scaled_rks"]
        ss32 = tier_agg["protocols"]["safesync_mesh_fixed32"]
        npv = tier_agg["protocols"]["receiver_aware_no_prov"]
        su = tier_agg["protocols"]["sender_utility_relay"]
        st = tier_agg["protocols"]["stateless_vq_flood"]
        print(
            f"  SafeSync(m={cfg_t['opt_bloom']:3d}b): Comp={ss['mission_completion_pct']:5.2f}% SyncF1={ss['effective_sync_f1']:5.2f}% "
            f"UM={ss['constellation_u_miss_pct']:5.2f}% Dup={ss['duplicate_traffic_pct']:4.2f}% TTC={ss['time_to_confirm_s']:5.1f}s "
            f"(FP emp={ss['bloom_fp_rate_empirical']:.3f} vs Prop3={ss['bloom_fp_bound_prop3']:.3f})"
        )
        print(
            f"  SafeSync(m= 32b): Comp={ss32['mission_completion_pct']:5.2f}% SyncF1={ss32['effective_sync_f1']:5.2f}% "
            f"FP emp={ss32['bloom_fp_rate_empirical']:.3f} vs Prop3={ss32['bloom_fp_bound_prop3']:.3f}"
        )
        print(
            f"  TWIST(No-Prov)  : Comp={npv['mission_completion_pct']:5.2f}% SyncF1={npv['effective_sync_f1']:5.2f}% "
            f"Dup={npv['duplicate_traffic_pct']:5.2f}% Infl={npv['confidence_inflation']:.4f} Nmax_Prop3={npv['n_max_prop3']:.0f}"
        )
        print(
            f"  SkillComm(Send) : Comp={su['mission_completion_pct']:5.2f}% SyncF1={su['effective_sync_f1']:5.2f}% "
            f"Dup={su['duplicate_traffic_pct']:5.2f}% | STARS: Comp={st['mission_completion_pct']:5.2f}% SyncF1={st['effective_sync_f1']:5.2f}%"
        )

    # Proposition 3 Dedicated Bloom Width Verification at 240-Sat Peak Surge (N_task=2,000, W=4 epochs)
    print("\n=== [Proposition 3 Verification] Bloom Filter Width m in {32, 64, 128, 256, 512} bits at 240 Sats (N=2,000) ===")
    assert plan_240 is not None
    for m_bits in [32, 64, 128, 256, 512]:
        rows_m = []
        for seed in SEEDS:
            sm = per_seed_map[seed]
            m_ss = sm["budget_sweep"]["safesync"][3]
            m_full = sm["full_feature_reference"]
            pcfg = {
                "has_rks": True, "has_provenance": True, "has_priority": True, "has_safety_reserve": True,
                "is_stateless": False, "rate_kb": m_ss["rate_kb_per_frame"], "known_f1": m_ss["known_f1"],
                "sync_f1": m_ss["sync_f1"], "u_miss": m_ss["u_miss"], "extra_decode_s": 0.6,
            }
            rng_m = np.random.default_rng(seed * 7777 + m_bits * 13)
            rows_m.append(simulate_scaled_constellation(
                pcfg, 2000, 60, plan_240, rng_m, m_full,
                isl_slice_kb_per_epoch=480.0, bloom_bits=m_bits, bloom_window_epochs=4,
            ))
        agg_m = _agg(rows_m, metrics_keys + ["mean_bloom_occupancy_n"])
        agg_m["bloom_bits"] = m_bits
        agg_m["rks_total_bits"] = m_bits + 32
        agg_m["abs_fp_gap"] = round(abs(agg_m["bloom_fp_rate_empirical"] - agg_m["bloom_fp_bound_prop3"]), 4)
        bloom_prop3_validation.append(agg_m)
        print(
            f"  m={m_bits:3d}b (RKS={m_bits+32:3d}b): n_occ={agg_m['mean_bloom_occupancy_n']:5.1f} | "
            f"FP_emp={agg_m['bloom_fp_rate_empirical']:.4f} vs Prop3={agg_m['bloom_fp_bound_prop3']:.4f} (gap={agg_m['abs_fp_gap']:.4f}) | "
            f"Comp={agg_m['mission_completion_pct']:5.2f}% SyncF1={agg_m['effective_sync_f1']:5.2f}% UM={agg_m['constellation_u_miss_pct']:5.2f}%"
        )

    out_payload = {
        "description": "S7: 24->60->120->240-Satellite Starlink Shell-1 Scalability Study & Proposition 3 Verification (3 Seeds)",
        "bloom_window_epochs": 4,
        "bloom_k_hashes": 2,
        "constellation_tiers": tier_results,
        "proposition3_bloom_verification_240sat": bloom_prop3_validation,
    }
    OUT_JSON.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    # Merge non-destructively into main JSON under scheme1_upgrades
    main_res.setdefault("scheme1_upgrades", {})
    main_res["scheme1_upgrades"]["constellation_scale_240"] = out_payload
    MAIN_RESULTS_JSON.write_text(json.dumps(main_res, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Saved] {OUT_JSON} & merged into safesync_r36_complete_results.json in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
