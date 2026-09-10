"""
AEGIS-AI — synthetic telemetry generator (Phase 2).

Produces a labelled multi-server telemetry dataset for training and evaluating
the anomaly-detection and incident-classification models. Real production
incident-response datasets with per-action outcomes do not exist publicly, so
the dataset is synthetic — but the *baseline* metric ranges are grounded in
published cloud-fleet statistics (see BASELINE GROUNDING below) rather than
being invented from scratch.

Output: data/synthetic_telemetry.csv with columns
    timestamp, server_id, cpu_util, mem_util, latency_ms, error_rate,
    traffic_rps, incident_type, severity

Run:  python data/generate_synthetic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# BASELINE GROUNDING
# --------------------------------------------------------------------------- #
# Microsoft's Azure Public Dataset (Cortez et al., "Resource Central: A Central
# Resource Management System for the Cloud", SOSP 2017) reports that the large
# majority of production cloud VMs are *not* busy: average CPU utilisation sits
# low, with most VMs averaging roughly 10-50% CPU and a heavy left-skew toward
# the 10-30% band. We therefore draw each server's normal-CPU baseline mean from
# U(12, 42)% and keep the normal operating envelope well below saturation.
# Memory, latency, error-rate and traffic baselines are plausible web-service
# values chosen to be internally consistent with that CPU range; only the CPU
# band is claimed to be literature-grounded.
CPU_BASELINE_MEAN_RANGE = (12.0, 42.0)  # percent, per Azure Public Dataset

SEED = 42
N_SERVERS = 18
DAYS = 21
SAMPLE_INTERVAL_MIN = 5
START = pd.Timestamp("2025-06-02 00:00:00")  # a Monday

INCIDENTS_PER_SERVER = (32, 49)   # randint low/high
INCIDENT_DURATION_SAMPLES = (6, 37)  # 30 min .. 3 h at 5-min cadence
BORDERLINE_FRACTION = 0.25         # windows deliberately near a threshold

INCIDENT_TYPES = (
    "cpu_overload",
    "memory_pressure",
    "traffic_spike",
    "latency_degradation",
    "error_burst",
)

OUT_CSV = Path(__file__).resolve().parent / "synthetic_telemetry.csv"
OUT_SUMMARY = Path(__file__).resolve().parent / "synthetic_telemetry_summary.json"


# --------------------------------------------------------------------------- #
# Baseline ("normal") generation
# --------------------------------------------------------------------------- #
def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """First-order autoregressive noise so the series is not white."""
    out = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _baseline(rng: np.random.Generator, ts: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(ts)
    minute_of_day = ts.hour * 60 + ts.minute
    day_frac = minute_of_day / 1440.0
    # smooth daily curve peaking early-afternoon
    daily = np.sin((day_frac - 0.28) * 2 * np.pi)
    is_weekend = (ts.dayofweek >= 5).astype(float)
    weekend_factor = 1.0 - 0.25 * is_weekend

    # ---- traffic: strong daily seasonality, lower on weekends ----
    traffic_mean = rng.uniform(300.0, 900.0)
    traffic_amp = traffic_mean * rng.uniform(0.35, 0.65)
    traffic = (
        traffic_mean
        + traffic_amp * daily
        + _ar1(rng, n, 0.9, traffic_mean * 0.04)
    ) * weekend_factor
    traffic = np.clip(traffic, 5.0, None)

    # ---- cpu: Azure-grounded low baseline + mild daily shape + load coupling ----
    cpu_mean = rng.uniform(*CPU_BASELINE_MEAN_RANGE)
    cpu = (
        cpu_mean
        + rng.uniform(3.0, 9.0) * daily
        + 0.010 * (traffic - traffic_mean)      # busier -> a bit more CPU
        + _ar1(rng, n, 0.85, 3.5)
    )
    cpu = np.clip(cpu, 1.0, 98.0)

    # ---- memory: slow random-walk drift + weak CPU coupling ----
    mem_mean = rng.uniform(40.0, 65.0)
    drift = np.cumsum(rng.normal(0.0, 0.03, n))
    drift = drift - drift.mean()
    mem = mem_mean + drift + 0.05 * (cpu - cpu_mean) + _ar1(rng, n, 0.9, 1.6)
    mem = np.clip(mem, 10.0, 96.0)

    # ---- latency: base + load-dependent term + noise ----
    lat_base = rng.uniform(60.0, 140.0)
    overload = np.clip((traffic - traffic_mean) / max(traffic_mean, 1.0), -0.5, None)
    latency = lat_base * (1.0 + 0.35 * np.clip(overload, 0, None)) + _ar1(rng, n, 0.7, 7.0)
    latency = np.clip(latency, 5.0, None)

    # ---- error rate: low, occasional harmless blips ----
    err_base = rng.uniform(0.002, 0.02)
    err_noise = np.abs(rng.normal(0.0, err_base * 0.5, n))
    blips = (rng.random(n) < 0.002) * rng.uniform(0.01, 0.03, n)
    error_rate = np.clip(err_base + err_noise + blips, 0.0, 1.0)

    return pd.DataFrame(
        {
            "cpu_util": cpu,
            "mem_util": mem,
            "latency_ms": latency,
            "error_rate": error_rate,
            "traffic_rps": traffic,
        }
    )


# --------------------------------------------------------------------------- #
# Incident injection
# --------------------------------------------------------------------------- #
def _severity(kind: str, magnitude: float) -> str:
    """Severity from how far past threshold the injected magnitude sits."""
    if kind == "cpu_overload":
        return "low" if magnitude < 88 else "medium" if magnitude < 94 else "high"
    if kind == "memory_pressure":
        return "low" if magnitude < 88 else "medium" if magnitude < 93 else "high"
    if kind == "traffic_spike":
        return "low" if magnitude < 3.5 else "medium" if magnitude < 6 else "high"
    if kind == "latency_degradation":
        return "low" if magnitude < 350 else "medium" if magnitude < 800 else "high"
    if kind == "error_burst":
        return "low" if magnitude < 0.08 else "medium" if magnitude < 0.2 else "high"
    raise ValueError(kind)


def _envelope(rng: np.random.Generator, dur: int) -> np.ndarray:
    """Raised-cosine ramp in / plateau / ramp out (so onsets are gradual)."""
    ramp = max(2, dur // 4)
    env = np.ones(dur)
    r = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[:ramp] = r
    env[-ramp:] = r[::-1]
    return env


def _inject(
    rng: np.random.Generator, df: pd.DataFrame, kind: str, s: int, dur: int, borderline: bool
) -> str:
    sl = slice(s, s + dur)
    env = _envelope(rng, dur)

    if kind == "cpu_overload":
        target = rng.uniform(78, 88) if borderline else rng.uniform(85, 99)
        cur = df["cpu_util"].to_numpy()[sl]
        df.loc[df.index[sl], "cpu_util"] = cur + env * np.maximum(0.0, target - cur)
        mag = target

    elif kind == "memory_pressure":
        target = rng.uniform(80, 89) if borderline else rng.uniform(85, 98)
        cur = df["mem_util"].to_numpy()[sl]
        df.loc[df.index[sl], "mem_util"] = cur + env * np.maximum(0.0, target - cur)
        cpu = df["cpu_util"].to_numpy()[sl]
        df.loc[df.index[sl], "cpu_util"] = np.clip(cpu + env * rng.uniform(3, 10), 1, 99)
        mag = target

    elif kind == "traffic_spike":
        mult = rng.uniform(2.0, 3.5) if borderline else rng.uniform(3.0, 8.0)
        tr = df["traffic_rps"].to_numpy()[sl]
        df.loc[df.index[sl], "traffic_rps"] = tr * (1.0 + env * (mult - 1.0))
        lat = df["latency_ms"].to_numpy()[sl]
        df.loc[df.index[sl], "latency_ms"] = lat * (1.0 + env * rng.uniform(0.3, 1.2))
        cpu = df["cpu_util"].to_numpy()[sl]
        df.loc[df.index[sl], "cpu_util"] = np.clip(cpu + env * rng.uniform(10, 30), 1, 99)
        mag = mult

    elif kind == "latency_degradation":
        target = rng.uniform(180, 350) if borderline else rng.uniform(280, 1500)
        cur = df["latency_ms"].to_numpy()[sl]
        df.loc[df.index[sl], "latency_ms"] = cur + env * np.maximum(0.0, target - cur)
        mag = target

    elif kind == "error_burst":
        target = rng.uniform(0.02, 0.06) if borderline else rng.uniform(0.04, 0.45)
        cur = df["error_rate"].to_numpy()[sl]
        df.loc[df.index[sl], "error_rate"] = np.clip(cur + env * np.maximum(0.0, target - cur), 0, 1)
        lat = df["latency_ms"].to_numpy()[sl]
        df.loc[df.index[sl], "latency_ms"] = lat * (1.0 + env * rng.uniform(0.1, 0.5))
        mag = target
    else:
        raise ValueError(kind)

    return _severity(kind, mag)


def _place_incidents(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df["incident_type"] = "normal"
    df["severity"] = "none"
    used: list[tuple[int, int]] = []
    target = int(rng.integers(*INCIDENTS_PER_SERVER))
    placed = 0
    attempts = 0
    while placed < target and attempts < target * 12:
        attempts += 1
        dur = int(rng.integers(*INCIDENT_DURATION_SAMPLES))
        s = int(rng.integers(0, n - dur))
        if any(not (s + dur + 6 <= a or s >= b + 6) for a, b in used):
            continue
        kind = INCIDENT_TYPES[int(rng.integers(0, len(INCIDENT_TYPES)))]
        borderline = rng.random() < BORDERLINE_FRACTION
        sev = _inject(rng, df, kind, s, dur, borderline)
        df.loc[df.index[s : s + dur], "incident_type"] = kind
        df.loc[df.index[s : s + dur], "severity"] = sev
        used.append((s, s + dur))
        placed += 1

    # re-clip after injections
    df["cpu_util"] = df["cpu_util"].clip(1.0, 100.0)
    df["mem_util"] = df["mem_util"].clip(1.0, 100.0)
    df["latency_ms"] = df["latency_ms"].clip(1.0, None)
    df["error_rate"] = df["error_rate"].clip(0.0, 1.0)
    df["traffic_rps"] = df["traffic_rps"].clip(1.0, None)
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    master = np.random.default_rng(SEED)
    ts = pd.date_range(START, periods=DAYS * 24 * 60 // SAMPLE_INTERVAL_MIN,
                       freq=f"{SAMPLE_INTERVAL_MIN}min")

    frames = []
    for i in range(N_SERVERS):
        rng = np.random.default_rng(master.integers(0, 2**32 - 1))
        server_id = f"srv-{i:02d}"
        df = _baseline(rng, ts)
        df.insert(0, "server_id", server_id)
        df.insert(0, "timestamp", ts)
        df = _place_incidents(rng, df)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.round(
        {"cpu_util": 3, "mem_util": 3, "latency_ms": 2, "error_rate": 6, "traffic_rps": 2}
    )
    full.to_csv(OUT_CSV, index=False)

    # ---- summary ----
    inc = full[full["incident_type"] != "normal"]
    # count distinct incident windows (contiguous runs per server)
    win_counts: dict[str, int] = {k: 0 for k in INCIDENT_TYPES}
    sev_counts: dict[str, int] = {}
    for _sid, g in full.groupby("server_id", sort=False):
        types = g["incident_type"].to_numpy()
        sevs = g["severity"].to_numpy()
        change = np.where(types[:-1] != types[1:])[0] + 1
        starts = np.concatenate(([0], change))
        for st in starts:
            if types[st] != "normal":
                win_counts[types[st]] += 1
                sev_counts[sevs[st]] = sev_counts.get(sevs[st], 0) + 1

    summary = {
        "rows": int(len(full)),
        "servers": N_SERVERS,
        "days": DAYS,
        "sample_interval_min": SAMPLE_INTERVAL_MIN,
        "time_range": [str(ts[0]), str(ts[-1])],
        "incident_rows": int(len(inc)),
        "incident_row_fraction": round(len(inc) / len(full), 4),
        "incident_windows_per_type": win_counts,
        "incident_windows_by_severity": sev_counts,
        "cpu_baseline_mean_range_pct": list(CPU_BASELINE_MEAN_RANGE),
        "cpu_grounding": "Azure Public Dataset (Cortez et al., SOSP 2017)",
        "borderline_fraction": BORDERLINE_FRACTION,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print(f"wrote {OUT_CSV}  ({len(full):,} rows, {OUT_CSV.stat().st_size/1e6:.1f} MB)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
