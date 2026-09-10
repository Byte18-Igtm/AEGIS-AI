"""
AEGIS-AI — feature engineering (Phase 3).

Turns a *window* of raw telemetry (consecutive samples for one server) into a
single fixed-length feature vector for the anomaly detector and the incident
classifiers.

A window is a DataFrame with at least the five metric columns
(cpu_util, mem_util, latency_ms, error_rate, traffic_rps). WINDOW_SIZE samples
= 1 hour at the dataset's 5-minute cadence.

Public API:
    make_window_features(window, baseline_traffic) -> dict[str, float]
    build_feature_table(df, window_size, stride, baseline_traffic)
        -> (X: DataFrame, meta: DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("cpu_util", "mem_util", "latency_ms", "error_rate", "traffic_rps")
WINDOW_SIZE = 12          # 1 hour at 5-min cadence
STRIDE = 6               # 50% overlap between consecutive labelled windows
DEFAULT_BASELINE_TRAFFIC = 550.0  # rps; overridden with the training-set value

# label a window as an incident only if this fraction of its rows carry a single
# non-normal incident_type (keeps borderline / edge windows partly ambiguous)
LABEL_COVERAGE = 0.40


def _slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def make_window_features(
    window: pd.DataFrame, baseline_traffic: float = DEFAULT_BASELINE_TRAFFIC
) -> dict[str, float]:
    """Compute the feature dict for one telemetry window."""
    f: dict[str, float] = {}
    cols = {m: window[m].to_numpy(dtype=float) for m in METRICS}

    for m, v in cols.items():
        f[f"{m}_mean"] = float(np.mean(v))
        f[f"{m}_std"] = float(np.std(v))
        f[f"{m}_min"] = float(np.min(v))
        f[f"{m}_max"] = float(np.max(v))
        f[f"{m}_last"] = float(v[-1])
        f[f"{m}_range"] = float(np.max(v) - np.min(v))
        f[f"{m}_delta"] = float(v[-1] - v[0])           # net change across window
        f[f"{m}_slope"] = _slope(v)                      # trend
        f[f"{m}_roc_max"] = float(np.max(np.abs(np.diff(v)))) if len(v) > 1 else 0.0

    # ---- threshold-violation fractions ----
    n = len(window)
    f["frac_cpu_gt_80"] = float(np.mean(cols["cpu_util"] > 80))
    f["frac_cpu_gt_90"] = float(np.mean(cols["cpu_util"] > 90))
    f["frac_mem_gt_85"] = float(np.mean(cols["mem_util"] > 85))
    f["frac_mem_gt_90"] = float(np.mean(cols["mem_util"] > 90))
    f["frac_latency_gt_300"] = float(np.mean(cols["latency_ms"] > 300))
    f["frac_latency_gt_600"] = float(np.mean(cols["latency_ms"] > 600))
    f["frac_error_gt_005"] = float(np.mean(cols["error_rate"] > 0.05))
    f["frac_error_gt_02"] = float(np.mean(cols["error_rate"] > 0.20))

    traffic_ratio = f["traffic_rps_mean"] / max(baseline_traffic, 1.0)
    f["traffic_ratio_to_baseline"] = float(traffic_ratio)
    f["frac_traffic_gt_2x"] = float(np.mean(cols["traffic_rps"] > 2 * baseline_traffic))
    f["frac_traffic_gt_4x"] = float(np.mean(cols["traffic_rps"] > 4 * baseline_traffic))

    # ---- cross-metric relationships ----
    tr = f["traffic_rps_mean"]
    f["latency_per_rps"] = f["latency_ms_mean"] / max(tr, 1.0)
    f["cpu_per_rps"] = f["cpu_util_mean"] / max(tr, 1.0)
    f["error_volume"] = f["error_rate_mean"] * tr
    f["cpu_minus_mem"] = f["cpu_util_mean"] - f["mem_util_mean"]
    f["cpu_mem_ratio"] = f["cpu_util_mean"] / max(f["mem_util_mean"], 1.0)
    f["corr_cpu_mem"] = _safe_corr(cols["cpu_util"], cols["mem_util"])
    f["corr_latency_traffic"] = _safe_corr(cols["latency_ms"], cols["traffic_rps"])
    f["corr_cpu_traffic"] = _safe_corr(cols["cpu_util"], cols["traffic_rps"])
    # latency rising while traffic flat => degradation not load-driven
    f["latency_slope_minus_traffic_slope"] = f["latency_ms_slope"] - f["traffic_rps_slope"]
    f["n_samples"] = float(n)
    return f


FEATURE_NAMES: list[str] = list(
    make_window_features(
        pd.DataFrame({m: np.linspace(1, 2, WINDOW_SIZE) for m in METRICS})
    ).keys()
)


def _window_label(types: np.ndarray, sevs: np.ndarray) -> tuple[str, str]:
    """Majority-ish label for a window from its per-row ground truth."""
    mask = types != "normal"
    if mask.mean() < LABEL_COVERAGE:
        return "normal", "none"
    vals, counts = np.unique(types[mask], return_counts=True)
    kind = str(vals[np.argmax(counts)])
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    worst = max(sevs[types == kind], key=lambda s: order.get(s, 0))
    return kind, str(worst)


def build_feature_table(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    baseline_traffic: float = DEFAULT_BASELINE_TRAFFIC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slide a window over every server and build the (X, meta) tables.

    meta columns: server_id, window_start, window_end, incident_type, severity,
    is_incident.
    """
    feat_rows: list[dict[str, float]] = []
    meta_rows: list[dict] = []
    has_labels = "incident_type" in df.columns

    for server_id, g in df.groupby("server_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        types = g["incident_type"].to_numpy() if has_labels else None
        sevs = g["severity"].to_numpy() if has_labels else None
        for start in range(0, len(g) - window_size + 1, stride):
            win = g.iloc[start : start + window_size]
            feat_rows.append(make_window_features(win, baseline_traffic))
            if has_labels:
                kind, sev = _window_label(
                    types[start : start + window_size], sevs[start : start + window_size]
                )
            else:
                kind, sev = "normal", "none"
            meta_rows.append(
                {
                    "server_id": server_id,
                    "window_start": win["timestamp"].iloc[0],
                    "window_end": win["timestamp"].iloc[-1],
                    "incident_type": kind,
                    "severity": sev,
                    "is_incident": kind != "normal",
                }
            )

    X = pd.DataFrame(feat_rows, columns=FEATURE_NAMES).fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    return X, meta
