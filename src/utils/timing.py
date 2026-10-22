"""Small helpers for latency measurement and reporting."""
from typing import Dict, Sequence

import numpy as np


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Mean / median / p95 / max of per-query latencies in milliseconds."""
    if not latencies_ms:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(arr.max()), 3),
    }
