"""Advanced cross-source track retrieval and ranking for Re-ID.

Drop-in replacement for the original ``retrieve_and_rank`` (same signature, same
return shape: ``(list_of_pairs, backend_string)``) with these accuracy upgrades:

1. Multi-shot tracks: a track may carry many frame embeddings ("embeddings", shape
   (N, D)) plus optional per-frame "quality" weights. Frames are quality-weighted,
   outlier frames are trimmed, and a track descriptor is built from the rest.
2. Domain-shift removal: joint mean-centering of both sources before cosine.
3. Chamfer (set-to-set) similarity between frame galleries, fused with the
   aggregated-descriptor cosine. More robust to pose/occlusion than one mean vector.
4. CSLS hubness correction: penalises "hub" tracks that look similar to everything,
   a common cause of false cross-camera matches.
5. Symmetric candidate generation (A->B and B->A) so mutual matches are never missed.
6. Mutual-nearest-neighbour flag, row/column margin, and a dual-softmax confidence.
7. Optional spatio-temporal gating (camera_id / start_time / end_time).
8. One-to-one assignment (Hungarian if SciPy is present, greedy otherwise).
9. FAISS used automatically for large sets; exact NumPy for small ones.

Note: "confidence" is a relative, uncalibrated score. Calibrate on labelled pairs
(e.g. Platt/isotonic on `score`) before treating it as a probability.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

log = logging.getLogger(__name__)
_EPS = 1e-12


@dataclass(frozen=True)
class RankConfig:
    pool: int = 20                      # candidates kept per track, per direction
    csls_k: int = 10                    # neighbourhood size for hubness correction
    use_csls: bool = True
    center: Union[bool, str] = "auto"   # True / False / "auto" (needs enough tracks)
    center_min_tracks: int = 8
    multishot_weight: float = 0.5       # weight of chamfer vs. aggregated cosine
    max_frames: int = 32                # cap frames per track (keeps highest quality)
    trim_outliers: bool = True
    temperature: float = 0.05           # dual-softmax temperature
    min_similarity: Optional[float] = None  # drop pairs whose fused cosine is below this
    require_cross_camera: bool = False
    max_time_gap: Optional[float] = None
    one_to_one: bool = True
    faiss_threshold: int = 2_000_000    # na * nb above which FAISS is preferred
    hungarian_max_cells: int = 4_000_000


# ----------------------------------------------------------------------------
# Embedding preparation
# ----------------------------------------------------------------------------
def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), _EPS)


def _gallery(track: dict, cfg: RankConfig):
    """Return (frames (n, D) L2-normalised, weights (n,)) or None if unusable."""
    raw = track.get("embeddings", track.get("embedding"))
    if raw is None:
        return None
    g = np.atleast_2d(np.asarray(raw, dtype=np.float32))
    if g.ndim != 2:
        return None

    q = track.get("quality")
    w = np.ones(len(g), dtype=np.float32)
    if q is not None:
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if len(q) == len(g):
            w = np.clip(np.nan_to_num(q), 0.0, None)

    ok = np.isfinite(g).all(axis=1) & (np.linalg.norm(g, axis=1) > _EPS)
    g, w = g[ok], w[ok]
    if len(g) == 0:
        return None
    if w.sum() <= _EPS:
        w = np.ones(len(g), dtype=np.float32)

    if len(g) > cfg.max_frames:
        keep = np.sort(np.argsort(-w, kind="stable")[: cfg.max_frames])
        g, w = g[keep], w[keep]

    g = _l2(g)
    if cfg.trim_outliers and len(g) >= 4:
        centre = _l2((g * w[:, None]).sum(0))
        s = g @ centre
        keep = s >= s.mean() - 2.0 * s.std()
        if keep.sum() >= 2:
            g, w = g[keep], w[keep]
    return g, w


def _prepare(tracks: list[dict], cfg: RankConfig):
    idx, gal, wts = [], [], []
    for i, t in enumerate(tracks):
        res = _gallery(t, cfg)
        if res is None:
            log.warning("Skipping track %r: missing/invalid embedding", t.get("track_id", i))
            continue
        idx.append(i)
        gal.append(res[0])
        wts.append(res[1])
    return idx, gal, wts


def _aggregate(gal, wts) -> np.ndarray:
    return _l2(np.stack([(g * w[:, None]).sum(0) for g, w in zip(gal, wts)]))


# ----------------------------------------------------------------------------
# Retrieval backends
# ----------------------------------------------------------------------------
def _topk_dense(q: np.ndarray, g: np.ndarray, k: int):
    sims = q @ g.T
    n_g = g.shape[0]
    k = min(k, n_g)
    if k < n_g:
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    else:
        part = np.broadcast_to(np.arange(n_g), (q.shape[0], n_g))
    vals = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _topk_faiss(faiss, q: np.ndarray, g: np.ndarray, k: int):
    index = faiss.IndexFlatIP(g.shape[1])
    index.add(np.ascontiguousarray(g))
    vals, idx = index.search(np.ascontiguousarray(q), min(k, g.shape[0]))
    return idx, vals


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------
def _chamfer(ga, wa, gb, wb) -> float:
    """Symmetric weighted best-match similarity between two frame galleries."""
    s = ga @ gb.T
    a_to_b = float((s.max(axis=1) * wa).sum() / wa.sum())
    b_to_a = float((s.max(axis=0) * wb).sum() / wb.sum())
    return 0.5 * (a_to_b + b_to_a)


def _compatible(a: dict, b: dict, cfg: RankConfig) -> bool:
    if cfg.require_cross_camera:
        ca, cb = a.get("camera_id"), b.get("camera_id")
        if ca is not None and ca == cb:
            return False
    if cfg.max_time_gap is not None:
        try:
            # Unrelated clip-relative zeroes are never comparable. Use the
            # smallest possible gap after accounting for clock uncertainty.
            gap = max(a["utc_start_bounds"][0] - b["utc_end_bounds"][1],
                      b["utc_start_bounds"][0] - a["utc_end_bounds"][1], 0.0)
        except (KeyError, TypeError):
            return True  # no timing info -> do not gate
        if gap > cfg.max_time_gap:
            return False
    return True


def _assign(rows: list[dict], na: int, nb: int, cfg: RankConfig) -> tuple[set, str]:
    """One-to-one assignment over candidate pairs. Returns (set of (i, j), method)."""
    try:
        from scipy.optimize import linear_sum_assignment
        if na * nb <= cfg.hungarian_max_cells:
            big = 1e6
            cost = np.full((na, nb), big, dtype=np.float64)
            for r in rows:
                cost[r["_i"], r["_j"]] = -r["score"]
            ri, ci = linear_sum_assignment(cost)
            return {(int(i), int(j)) for i, j in zip(ri, ci) if cost[i, j] < big}, "Hungarian"
    except ImportError:
        pass
    used_a, used_b, chosen = set(), set(), set()
    for r in sorted(rows, key=lambda r: r["score"], reverse=True):
        if r["_i"] not in used_a and r["_j"] not in used_b:
            used_a.add(r["_i"]); used_b.add(r["_j"]); chosen.add((r["_i"], r["_j"]))
    return chosen, "greedy"


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def retrieve_and_rank(
    tracks_a: list[dict],
    tracks_b: list[dict],
    top_k: int = 5,
    config: Optional[RankConfig] = None,
):
    """Retrieve and rank cross-source track pairs.

    Each track dict needs ``track_id`` and either ``embedding`` (D,) / (N, D) or
    ``embeddings`` (N, D). Optional keys: ``quality`` (N,), ``camera_id``,
    ``start_time``, ``end_time``.

    Returns ``(pairs, backend)``. Every pair dict has:
        track_a, track_b, pair_key
        similarity        raw cosine of aggregated descriptors (original meaning)
        fused_similarity  cosine fused with multi-shot chamfer similarity
        score             CSLS-corrected ranking score (use this to sort/threshold)
        confidence        dual-softmax relative confidence in [0, 1] (uncalibrated)
        margin            score gap to the best competing candidate (row/col min)
        mutual_nn         True if each track is the other's top choice
        one_to_one        True if chosen by the global one-to-one assignment
    """
    cfg = config or RankConfig()
    if top_k < 1 or min(cfg.pool, cfg.csls_k, cfg.max_frames) < 1:
        raise ValueError("Retrieval limits must be positive.")
    if not np.isfinite(cfg.temperature) or cfg.temperature <= 0 or not 0 <= cfg.multishot_weight <= 1:
        raise ValueError("Invalid ranking temperature or fusion weight.")
    if not tracks_a or not tracks_b:
        return [], "no candidates"

    ia, ga, wa = _prepare(tracks_a, cfg)
    ib, gb, wb = _prepare(tracks_b, cfg)
    if not ia or not ib:
        return [], "no valid embeddings"

    dims = {g.shape[1] for g in ga} | {g.shape[1] for g in gb}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {sorted(dims)}")

    na, nb = len(ia), len(ib)

    # Joint mean-centering removes a shared per-source bias (domain shift).
    agg_a, agg_b = _aggregate(ga, wa), _aggregate(gb, wb)
    raw_a, raw_b = agg_a.copy(), agg_b.copy()
    do_center = cfg.center is True or (cfg.center == "auto" and na + nb >= cfg.center_min_tracks)
    if do_center:
        mu = np.concatenate([agg_a, agg_b]).mean(axis=0, keepdims=True)
        ga = [_l2(g - mu) for g in ga]
        gb = [_l2(g - mu) for g in gb]
        agg_a, agg_b = _aggregate(ga, wa), _aggregate(gb, wb)

    # Symmetric top-k retrieval.
    pool = max(cfg.pool, top_k)
    k_ret = max(pool, cfg.csls_k)
    backend = "NumPy exact cosine"
    topk = _topk_dense
    if na * nb > cfg.faiss_threshold:
        try:
            import faiss
            topk = lambda q, g, k: _topk_faiss(faiss, q, g, k)
            backend = "FAISS IndexFlatIP"
        except ImportError:
            pass
    ab_idx, ab_sim = topk(agg_a, agg_b, k_ret)
    ba_idx, ba_sim = topk(agg_b, agg_a, k_ret)

    # Hubness statistics for CSLS.
    r_a = ab_sim[:, : min(cfg.csls_k, ab_sim.shape[1])].mean(axis=1)
    r_b = ba_sim[:, : min(cfg.csls_k, ba_sim.shape[1])].mean(axis=1)

    # Union of candidates from both directions, remembering retrieval ranks.
    cand: dict[tuple[int, int], float] = {}
    rank_ab: dict[tuple[int, int], int] = {}
    rank_ba: dict[tuple[int, int], int] = {}
    for i in range(na):
        for rank in range(min(pool, ab_idx.shape[1])):
            j = int(ab_idx[i, rank])
            cand[(i, j)] = float(ab_sim[i, rank])
            rank_ab[(i, j)] = rank
    for j in range(nb):
        for rank in range(min(pool, ba_idx.shape[1])):
            i = int(ba_idx[j, rank])
            cand[(i, j)] = float(ba_sim[j, rank])
            rank_ba[(i, j)] = rank

    # Re-score candidates.
    rows: list[dict] = []
    w = cfg.multishot_weight
    for (i, j), cos in cand.items():
        ta, tb = tracks_a[ia[i]], tracks_b[ib[j]]
        if not _compatible(ta, tb, cfg):
            continue
        fused = cos
        if w > 0 and (len(ga[i]) > 1 or len(gb[j]) > 1):
            fused = (1.0 - w) * cos + w * _chamfer(ga[i], wa[i], gb[j], wb[j])
        if cfg.min_similarity is not None and fused < cfg.min_similarity:
            continue
        score = 2.0 * fused - r_a[i] - r_b[j] if cfg.use_csls else fused
        rows.append({
            "_i": i, "_j": j, "track_a": ta, "track_b": tb,
            "similarity": float(np.clip(raw_a[i] @ raw_b[j], -1, 1)), "fused_similarity": float(fused), "score": float(score),
            "retrieval_rank_a": rank_ab.get((i, j), -1) + 1,
            "retrieval_rank_b": rank_ba.get((i, j), -1) + 1,
            "mutual_nn": rank_ab.get((i, j), 1) == 0 and rank_ba.get((i, j), 1) == 0,
        })
    if not rows:
        return [], backend

    # Margins and dual-softmax confidence over the surviving candidate sets.
    by_row: dict[int, list[float]] = {}
    by_col: dict[int, list[float]] = {}
    for r in rows:
        by_row.setdefault(r["_i"], []).append(r["score"])
        by_col.setdefault(r["_j"], []).append(r["score"])

    def _margin(scores: list[float], s: float) -> float:
        others = sorted(scores, reverse=True)
        best_other = others[1] if s >= others[0] and len(others) > 1 else (others[0] if s < others[0] else 0.0)
        return s - best_other

    def _softmax_share(scores: list[float], s: float) -> float:
        arr = np.asarray(scores, dtype=np.float64)
        m = arr.max()
        return float(np.exp((s - m) / cfg.temperature) / np.exp((arr - m) / cfg.temperature).sum())

    for r in rows:
        rs, cs = by_row[r["_i"]], by_col[r["_j"]]
        r["margin"] = float(min(_margin(rs, r["score"]), _margin(cs, r["score"])))
        r["confidence"] = _softmax_share(rs, r["score"]) * _softmax_share(cs, r["score"])

    method = "no assignment"
    chosen: set = set()
    if cfg.one_to_one:
        chosen, method = _assign(rows, na, nb, cfg)

    # Final ordering, top_k per source-A track, stable pair keys.
    rows.sort(key=lambda r: r["score"], reverse=True)
    per_row: dict[int, int] = {}
    output, seen = [], set()
    for r in rows:
        if per_row.get(r["_i"], 0) >= top_k:
            continue
        a, b = r["track_a"], r["track_b"]
        pair_key = hashlib.sha256(f"{a['track_id']}|{b['track_id']}".encode()).hexdigest()[:16]
        if pair_key in seen:
            continue
        seen.add(pair_key)
        per_row[r["_i"]] = per_row.get(r["_i"], 0) + 1
        output.append({
            "track_a": a, "track_b": b, "pair_key": pair_key,
            "similarity": r["similarity"], "fused_similarity": r["fused_similarity"],
            "score": r["score"], "confidence": r["confidence"], "margin": r["margin"],
            "mutual_nn": r["mutual_nn"], "one_to_one": (r["_i"], r["_j"]) in chosen,
            "retrieval_rank_a": r["retrieval_rank_a"], "retrieval_rank_b": r["retrieval_rank_b"],
        })

    parts = [backend, f"{'centered' if do_center else 'uncentered'}"]
    if w > 0:
        parts.append("multi-shot chamfer")
    if cfg.use_csls:
        parts.append(f"CSLS(k={cfg.csls_k})")
    parts.append(method)
    return output, " + ".join(parts)


# ----------------------------------------------------------------------------
# Self-test: synthetic identities with a domain shift between sources
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_id, dim = 60, 128
    ids = rng.normal(size=(n_id, dim))
    shift = rng.normal(size=dim) * 1.2  # source-B domain bias

    def make(prefix, bias):
        out = []
        for k in range(n_id):
            frames = ids[k] + rng.normal(size=(6, dim)) * 1.6 + bias
            out.append({"track_id": f"{prefix}{k}", "embeddings": frames,
                        "quality": rng.uniform(0.3, 1.0, size=6)})
        return out

    A, B = make("a", 0.0), make("b", shift)

    # Baseline: plain cosine on a single frame embedding (original behaviour).
    qa = _l2(np.stack([t["embeddings"][0] for t in A]))
    qb = _l2(np.stack([t["embeddings"][0] for t in B]))
    base_acc = float((np.argmax(qa @ qb.T, axis=1) == np.arange(n_id)).mean())

    pairs, backend = retrieve_and_rank(A, B, top_k=1)
    hits = sum(p["track_a"]["track_id"][1:] == p["track_b"]["track_id"][1:] for p in pairs)
    print(f"backend : {backend}")
    print(f"baseline top-1 accuracy : {base_acc:.2%}")
    print(f"advanced top-1 accuracy : {hits / n_id:.2%}")
    top = pairs[0]
    print({k: (round(v, 3) if isinstance(v, float) else v)
           for k, v in top.items() if k not in ("track_a", "track_b")})
