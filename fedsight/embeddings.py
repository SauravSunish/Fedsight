"""Advanced OSNet (Torchreid) tracklet embedding builder.

Drop-in replacement for the original ``build_osnet_embeddings(tracklets)``:
same input (``track["crops_bgr"]``), same return ``(tracklets, note)``, same
``track["embedding"]`` output. It additionally writes the multi-shot fields that
``reid_retrieval.retrieve_and_rank`` consumes (``embeddings`` and ``quality``).

Upgrades over the original:
 1. Per-crop quality scoring (sharpness, size, aspect, exposure/contrast, optional
    detector confidence); low-quality crops are filtered and the rest are weighted.
 2. Temporal-bin crop selection: caps crops per track while keeping the best crop
    from each time slice, so pose/lighting diversity survives.
 3. Horizontal-flip test-time augmentation (standard Re-ID accuracy boost).
 4. Batched inference with fp16 autocast on CUDA and automatic OOM back-off.
 5. Cached extractor (model loads once) plus an optional on-disk feature cache.
 6. Outlier-crop trimming and quality-weighted aggregation of the track descriptor.
 7. Track consistency score that flags likely ID switches / merged identities.
 8. Pluggable extractor (dependency injection) for testing or other backbones.
 9. Graceful handling of bad crops and tracks with no usable crops.

IMPORTANT: with no ``model_path``, Torchreid loads ImageNet-pretrained weights, which
are NOT trained for person Re-ID (as far as I know). For real accuracy, pass weights
trained on Market1501/MSMT17, e.g. ``OSNetConfig(model_name="osnet_ain_x1_0",
model_path="osnet_ain_x1_0_msmt17.pth")`` (the AIN variant generalises better
across cameras/domains).
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)
_EPS = 1e-12
_EXTRACTORS: dict = {}


@dataclass(frozen=True)
class OSNetConfig:
    model_name: str = "osnet_x1_0"
    model_path: Optional[str] = None
    image_size: tuple = (256, 128)
    device: Optional[str] = None          # None -> cuda if available else cpu
    batch_size: int = 64
    use_fp16: bool = True                 # autocast on CUDA only
    flip_tta: bool = True
    max_crops_per_track: int = 24
    min_crop_height: int = 32
    min_crop_width: int = 16
    min_quality: float = 0.25             # soft filter; best `min_keep` always survive
    min_keep: int = 1
    quality_power: float = 1.0            # aggregation weight = quality ** power
    trim_outliers: bool = True
    consistency_warn: float = 0.55        # mean crop-to-centroid cosine below this -> flag
    cache_dir: Optional[str] = None       # on-disk feature cache (None = memory only)
    keep_crops: bool = False              # original code dropped crops after extraction
    strict: bool = False                  # raise if any track has no usable crop


# ----------------------------------------------------------------------------
# Crop quality
# ----------------------------------------------------------------------------
def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), _EPS)


def _gray_fixed(img: np.ndarray, size=(128, 64)) -> np.ndarray:
    gray = img.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    ys = np.linspace(0, h - 1, size[0]).astype(int)
    xs = np.linspace(0, w - 1, size[1]).astype(int)
    return gray[np.ix_(ys, xs)]


def crop_quality(crop: np.ndarray, det_conf: Optional[float] = None) -> float:
    """Heuristic quality in [0, 1]: size, sharpness, person aspect, exposure."""
    h, w = crop.shape[:2]
    g = _gray_fixed(crop)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    q_sharp = 1.0 - np.exp(-float(lap.var()) / 60.0)
    q_area = min(1.0, (h * w) / (128 * 64)) ** 0.5
    q_aspect = float(np.exp(-abs(np.log(max(h, 1) / max(w, 1) / 2.0))))  # persons ~2:1
    exposure = max(0.0, 1.0 - (abs(float(g.mean()) - 127.5) / 127.5) ** 2)
    contrast = min(1.0, float(g.std()) / 35.0)
    q_expo = float(np.sqrt(exposure * contrast))

    parts = [(q_area, 0.3), (q_sharp, 0.3), (q_aspect, 0.2), (q_expo, 0.2)]
    q = float(np.exp(sum(wt * np.log(max(v, 1e-3)) for v, wt in parts)))
    if det_conf is not None:
        q *= float(np.clip(det_conf, 0.0, 1.0)) ** 0.5
    return float(np.clip(q, 0.0, 1.0))


def _select_diverse(idx: list, quals: np.ndarray, k: int) -> list:
    """Best-quality crop from each of k temporal bins (keeps diversity)."""
    if len(idx) <= k:
        return idx
    bins = np.array_split(np.arange(len(idx)), k)
    return [idx[b[int(np.argmax(quals[np.asarray(idx)[b]]))]] for b in bins]


# ----------------------------------------------------------------------------
# Extractor, batching, cache
# ----------------------------------------------------------------------------
def _get_extractor(cfg: OSNetConfig):
    try:
        import torch
        try:
            from torchreid.utils import FeatureExtractor
        except ModuleNotFoundError as exc:
            if exc.name != "torchreid.utils":
                raise
            # PyPI 0.2.5 nests the official implementation under reid/.
            from torchreid.reid.utils import FeatureExtractor
    except Exception as exc:
        raise RuntimeError(
            "OSNet/Torchreid could not be imported. Install torch + torchvision first, "
            "then run `pip install torchreid`. Details: " + str(exc)
        ) from exc
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights_hash = hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest() if cfg.model_path else "imagenet"
    key = (cfg.model_name, weights_hash, tuple(cfg.image_size), device)
    if key not in _EXTRACTORS:
        _EXTRACTORS[key] = FeatureExtractor(
            model_name=cfg.model_name,
            model_path=cfg.model_path or "",
            device=device,
            image_size=tuple(cfg.image_size),
            verbose=False,
        )
    return _EXTRACTORS[key], device


def _forward(extract: Callable, imgs: list, device: str, fp16: bool) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        feats = extract(imgs)
    else:
        amp = torch.autocast("cuda", dtype=torch.float16) if (fp16 and device == "cuda") \
            else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            feats = extract(imgs)
    if hasattr(feats, "detach"):
        feats = feats.detach().float().cpu().numpy()
    return np.asarray(feats, dtype=np.float32)


def _run_batches(extract, images, batch, device, fp16):
    """Batched inference; halves the batch size on CUDA out-of-memory."""
    out, i = [], 0
    while i < len(images):
        chunk = images[i:i + batch]
        try:
            out.append(_forward(extract, chunk, device, fp16))
            i += len(chunk)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch > 1:
                batch = max(1, batch // 2)
                log.warning("CUDA OOM, retrying with batch size %d", batch)
                with contextlib.suppress(Exception):
                    import torch
                    torch.cuda.empty_cache()
                continue
            raise
    return np.concatenate(out, axis=0), batch


class _FeatureCache:
    def __init__(self, directory: Optional[str], tag: str):
        self.mem: dict = {}
        self.dir = directory
        self.tag = hashlib.sha1(tag.encode()).hexdigest()[:10]
        self.hits = 0
        if directory:
            os.makedirs(directory, exist_ok=True)

    def key(self, img: np.ndarray) -> str:
        return hashlib.sha256(str(img.shape).encode() + np.ascontiguousarray(img).tobytes()).hexdigest() + self.tag

    def get(self, k: str):
        if k in self.mem:
            self.hits += 1
            return self.mem[k]
        if self.dir:
            path = os.path.join(self.dir, k + ".npy")
            if os.path.isfile(path):
                try:
                    self.mem[k] = np.load(path, allow_pickle=False)
                    self.hits += 1
                    return self.mem[k]
                except Exception:
                    return None
        return None

    def put(self, k: str, v: np.ndarray):
        self.mem[k] = v
        if self.dir:
            np.save(os.path.join(self.dir, k + ".npy"), v)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def build_osnet_embeddings(
    tracklets: list[dict],
    config: Optional[OSNetConfig] = None,
    extractor: Optional[Callable] = None,
) -> tuple[list[dict], str]:
    """Create quality-weighted OSNet embeddings for every tracklet.

    Input track keys: ``track_id``, ``crops_bgr`` (list of HxWx3 uint8 BGR arrays),
    optional ``det_confs`` (one detector confidence per crop).

    Output keys written on each track:
        embedding        (D,) L2-normalised, quality-weighted track descriptor
        embeddings       (N, D) per-crop L2-normalised features (for chamfer matching)
        quality          (N,) per-crop quality weights
        embedding_meta   dict: n_crops_in/used, mean_quality, consistency,
                         possible_id_switch, crop_indices
    Tracks with no usable crop get ``embedding_status = "no_usable_crops"`` (or raise
    if ``cfg.strict``) and are skipped downstream by ``retrieve_and_rank``.

    ``extractor`` may be any callable taking a list of HxWx3 RGB uint8 arrays and
    returning an (N, D) array; if omitted, Torchreid's FeatureExtractor is used.
    """
    cfg = config or OSNetConfig()
    if min(cfg.batch_size, cfg.max_crops_per_track, cfg.min_keep, cfg.min_crop_height, cfg.min_crop_width) < 1:
        raise ValueError("Embedding batch/crop limits must be positive.")
    if not 0 <= cfg.min_quality <= 1 or not np.isfinite(cfg.quality_power) or cfg.quality_power <= 0:
        raise ValueError("Invalid embedding quality configuration.")
    if extractor is not None and cfg.cache_dir:
        raise ValueError("Disk caching is disabled for custom extractors without a versioned model identity.")
    weights_hash = hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest() if cfg.model_path else "imagenet-default"
    if extractor is None:
        extract, device = _get_extractor(cfg)
        model_desc = f"{cfg.model_name}{'' if cfg.model_path else ' (ImageNet weights only)'}"
    else:
        extract, device = extractor, cfg.device or "cpu"
        model_desc = "custom extractor"

    # 1) Validate, score, filter, and select crops per track.
    plan: list[tuple[int, int, float, np.ndarray]] = []  # (track_idx, crop_idx, quality, rgb)
    total_in = 0
    for t_idx, track in enumerate(tracklets):
        crops = track.get("crops_bgr") or []
        confs = track.get("det_confs")
        total_in += len(crops)
        valid = []
        for c_idx, crop in enumerate(crops):
            if (crop is None or not hasattr(crop, "ndim") or crop.ndim != 3
                    or crop.shape[2] != 3 or crop.size == 0
                    or crop.shape[0] < cfg.min_crop_height or crop.shape[1] < cfg.min_crop_width):
                continue
            dc = confs[c_idx] if confs is not None and c_idx < len(confs) else None
            valid.append((c_idx, crop, crop_quality(crop, dc)))
        if not valid:
            continue
        quals = np.array([v[2] for v in valid], dtype=np.float32)
        keep = [n for n, q in enumerate(quals) if q >= cfg.min_quality]
        if len(keep) < cfg.min_keep:
            keep = list(np.sort(np.argsort(-quals)[: cfg.min_keep]))
        keep = _select_diverse(keep, quals, cfg.max_crops_per_track)
        for n in keep:
            c_idx, crop, q = valid[n]
            plan.append((t_idx, c_idx, q, np.ascontiguousarray(crop[:, :, ::-1])))

    if not plan:
        raise RuntimeError("No person crops were retained for OSNet feature extraction.")

    # 2) Feature extraction with cache, batching, and flip TTA.
    cache = _FeatureCache(
        cfg.cache_dir,
        f"{cfg.model_name}|{weights_hash}|{tuple(cfg.image_size)}|{cfg.flip_tta}|{model_desc}",
    )
    keys = [cache.key(p[3]) for p in plan]
    feats: list = [cache.get(k) for k in keys]
    todo = [n for n, f in enumerate(feats) if f is None]
    batch = cfg.batch_size
    if todo:
        imgs = [plan[n][3] for n in todo]
        f = _l2(_run_batches(extract, imgs, batch, device, cfg.use_fp16)[0])
        if f.ndim != 2 or len(f) != len(imgs) or not np.isfinite(f).all():
            raise ValueError("Extractor returned malformed or non-finite embeddings.")
        if cfg.flip_tta:
            flipped = [np.ascontiguousarray(im[:, ::-1]) for im in imgs]
            f = _l2(f + _l2(_run_batches(extract, flipped, batch, device, cfg.use_fp16)[0]))
        for n, row in zip(todo, f):
            feats[n] = row.astype(np.float32)
            cache.put(keys[n], feats[n])

    # 3) Aggregate per track.
    per_track: dict[int, list] = {}
    for (t_idx, c_idx, q, _), f in zip(plan, feats):
        per_track.setdefault(t_idx, []).append((c_idx, q, f))

    flagged, used_total, q_all = [], 0, []
    for t_idx, track in enumerate(tracklets):
        items = per_track.get(t_idx)
        if not items:
            if cfg.strict:
                raise RuntimeError(f"No usable crops for track {track.get('track_id', t_idx)}.")
            track["embedding_status"] = "no_usable_crops"
            if not cfg.keep_crops:
                track.pop("crops_bgr", None)
            continue

        crop_idx = [i[0] for i in items]
        quality = np.array([i[1] for i in items], dtype=np.float32)
        F = _l2(np.stack([i[2] for i in items]))
        w = np.maximum(quality, 1e-3) ** cfg.quality_power

        if cfg.trim_outliers and len(F) >= 4:
            centre = _l2((F * w[:, None]).sum(0))
            s = F @ centre
            ok = s >= s.mean() - 2.0 * s.std()
            if ok.sum() >= 2:
                F, w, quality = F[ok], w[ok], quality[ok]
                crop_idx = [c for c, k in zip(crop_idx, ok) if k]

        mean_vec = (F * w[:, None]).sum(0)
        if float(np.linalg.norm(mean_vec)) <= _EPS:
            raise RuntimeError(f"OSNet produced an invalid zero embedding for {track['track_id']}.")
        centre = _l2(mean_vec)
        consistency = float(((F @ centre) * w).sum() / w.sum())
        suspect = bool(len(F) >= 3 and consistency < cfg.consistency_warn)
        if suspect:
            flagged.append(track.get("track_id", t_idx))

        track["embedding"] = centre.astype(np.float32)
        track["embeddings"] = F.astype(np.float32)
        track["quality"] = quality
        track["embedding_meta"] = {
            "model_name": model_desc, "weights_sha256": weights_hash,
            "n_crops_in": len(track.get("crops_bgr") or []),
            "n_crops_used": int(len(F)),
            "mean_quality": float(quality.mean()),
            "consistency": consistency,
            "possible_id_switch": suspect,
            "crop_indices": crop_idx,
        }
        used_total += len(F)
        q_all.append(float(quality.mean()))
        if not cfg.keep_crops:
            track.pop("crops_bgr", None)  # crops are no longer needed after extraction

    n_ok = len(q_all)
    note = (
        f"{model_desc} embeddings on {device.upper()} for {n_ok}/{len(tracklets)} tracklets "
        f"({used_total}/{total_in} crops used after quality filtering"
        f"{', flip-TTA' if cfg.flip_tta else ''}, quality-weighted mean, "
        f"{cache.hits} cache hits). Mean crop quality {np.mean(q_all) if q_all else 0:.2f}."
    )
    if flagged:
        note += f" Low-consistency tracks (possible ID switch): {flagged}."
    return tracklets, note


# ----------------------------------------------------------------------------
# Self-test with a fake extractor (no torch/torchreid needed)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    proj = rng.normal(size=(8 * 4 * 3, 64)).astype(np.float32)

    def fake_extractor(images):
        out = []
        for im in images:
            small = im[np.ix_(np.linspace(0, im.shape[0] - 1, 8).astype(int),
                              np.linspace(0, im.shape[1] - 1, 4).astype(int))]
            out.append(small.astype(np.float32).ravel() @ proj)
        return np.stack(out)

    def person(base, blur=False, dark=False):
        img = np.clip(base + rng.normal(0, 25, size=(128, 64, 3)), 0, 255)
        if blur:
            img = np.full_like(img, img.mean())
        if dark:
            img = img * 0.1
        return img.astype(np.uint8)

    def make(tid, base):
        crops = [person(base) for _ in range(8)] + [person(base, blur=True), person(base, dark=True)]
        return {"track_id": tid, "crops_bgr": crops}

    bases = rng.uniform(40, 215, size=(4, 1, 1, 3))
    tracks = [make(f"t{i}", bases[i]) for i in range(4)]
    tracks, note = build_osnet_embeddings(tracks, OSNetConfig(), extractor=fake_extractor)
    print(note)
    m = tracks[0]["embedding_meta"]
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    print("embedding", tracks[0]["embedding"].shape, "| per-crop", tracks[0]["embeddings"].shape)

    try:  # chain into the retrieval module if it sits beside this file
        from reid_retrieval import retrieve_and_rank
        pairs, backend = retrieve_and_rank(tracks[:2], tracks[2:], top_k=1)
        print("retrieval OK:", backend, "| pairs:", len(pairs))
    except ImportError:
        pass
