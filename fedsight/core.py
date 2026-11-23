"""Shared contracts for the local FedSight workbench."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

VERSION = "2.0.0"
FEATURE_VERSION = "threadsight-2.0-v1"


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite values cannot enter an evidence record.")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclasses.dataclass(frozen=True)
class TimeAnchor:
    original_timestamp: str | None = None
    timezone_name: str = "unknown"
    offset_s: float = 0.0
    uncertainty_s: float = 0.0
    drift_ppm: float = 0.0
    method: str = "unknown"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.offset_s, self.uncertainty_s, self.drift_ppm)) or self.uncertainty_s < 0:
            raise ValueError("Clock values must be finite; uncertainty must be nonnegative.")
        if abs(self.drift_ppm) >= 1_000_000:
            raise ValueError("Clock drift must preserve increasing time.")
        if self.original_timestamp:
            parsed = datetime.fromisoformat(self.original_timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Time anchors need an explicit UTC offset, e.g. +05:30.")
            if self.method == "unknown":
                raise ValueError("Describe how the clock was aligned.")

    def interval(self, relative_s: float) -> tuple[float, float] | None:
        if not self.original_timestamp:
            return None
        if not math.isfinite(relative_s) or relative_s < 0:
            raise ValueError("Relative time must be finite and nonnegative.")
        anchor = datetime.fromisoformat(self.original_timestamp.replace("Z", "+00:00")).timestamp()
        center = anchor + self.offset_s + relative_s * (1 + self.drift_ppm / 1_000_000)
        return center - self.uncertainty_s, center + self.uncertainty_s


@dataclasses.dataclass(frozen=True)
class RejectionPolicy:
    min_similarity: float = 0.55
    min_quality: float = 0.30
    min_margin: float = 0.08

    def __post_init__(self):
        if not (-1 <= self.min_similarity <= 1 and 0 <= self.min_quality <= 1 and 0 <= self.min_margin <= 4):
            raise ValueError("Invalid rejection thresholds.")
