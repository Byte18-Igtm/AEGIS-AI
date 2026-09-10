"""
AEGIS-AI pipeline — orchestration of the six decision-support stages.

    generate_telemetry -> detect_anomaly -> classify_incident
    -> generate_actions -> simulate_action (per action) -> rank_actions

Stages 2-6 now use the trained models (models/, built by src/train.py) and the
helper modules features / actions / simulate / risk. detect_anomaly and
classify_incident operate on a *window*: consecutive telemetry samples for a
single server (about features.WINDOW_SIZE rows).

AEGIS-AI is decision-support only. It never executes a response action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# flat src/ layout: make sibling modules importable no matter the entry point
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------- #
# Vocabularies shared across stages
# --------------------------------------------------------------------------- #
INCIDENT_TYPES = (
    "cpu_overload",
    "memory_pressure",
    "traffic_spike",
    "latency_degradation",
    "error_burst",
)
# The incident-type classifier also predicts this "no incident" class.
NORMAL_LABEL = "normal"
SEVERITIES = ("none", "low", "medium", "high")
METRICS = ("cpu_util", "mem_util", "latency_ms", "error_rate", "traffic_rps")
ACTION_VOCAB = (
    "scale_out",
    "restart_service",
    "reroute_traffic",
    "rollback_deployment",
    "isolate_component",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TELEMETRY_CSV = DATA_DIR / "synthetic_telemetry.csv"


# --------------------------------------------------------------------------- #
# Stage 1 — Telemetry loading
# --------------------------------------------------------------------------- #
def generate_telemetry(csv_path: str | Path = TELEMETRY_CSV) -> pd.DataFrame:
    """Load the labelled synthetic telemetry dataset.

    The dataset is produced by data/generate_synthetic.py: 18 simulated servers
    sampled every 5 minutes over 21 days, with injected windows for the five
    incident types. Baseline CPU ranges are grounded in Microsoft's Azure Public
    Dataset (see that script's header); the rest is synthetic. See STATUS_REPORT.md
    for why the dataset is synthetic rather than a real production trace.

    Input:
      csv_path  Path to the telemetry CSV. Defaults to data/synthetic_telemetry.csv.

    Output:
      DataFrame sorted by (server_id, timestamp), one row per sample, columns:
        timestamp     (datetime64[ns])
        server_id     (str)
        cpu_util      (float)  CPU utilisation, percent (0-100)
        mem_util      (float)  memory utilisation, percent (0-100)
        latency_ms    (float)  request latency, milliseconds
        error_rate    (float)  fraction of failed requests (0-1)
        traffic_rps   (float)  requests per second
        incident_type (str)    ground truth: "normal" or one of INCIDENT_TYPES
        severity      (str)    ground truth: one of SEVERITIES

    incident_type / severity are ground-truth labels for training and
    evaluation. The pipeline never reads them at inference time.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    return df


def get_window(
    df: pd.DataFrame, server_id: str, start_ts, size: int | None = None
) -> pd.DataFrame:
    """Slice `size` consecutive rows for one server starting at/after start_ts."""
    from features import WINDOW_SIZE

    size = size or WINDOW_SIZE
    g = (
        df[df["server_id"] == server_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    start_ts = pd.Timestamp(start_ts)
    idx = g.index[g["timestamp"] >= start_ts]
    i = int(idx[0]) if len(idx) else 0
    return g.iloc[i : i + size].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trained-model loading (cached)
# --------------------------------------------------------------------------- #
_MODELS: dict | None = None


def load_models(models_dir: str | Path = MODELS_DIR) -> dict:
    """Load and cache the trained models + metadata from models/.

    Returns a dict: isolation_forest, rf_incident_type, rf_severity, metadata.
    Raises FileNotFoundError (with a hint) if src/train.py has not been run.
    """
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    import joblib

    md = Path(models_dir)
    needed = [
        "isolation_forest.joblib",
        "rf_incident_type.joblib",
        "rf_severity.joblib",
        "metadata.joblib",
    ]
    missing = [n for n in needed if not (md / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {md}: {missing}. Run `python src/train.py` first."
        )
    _MODELS = {
        "isolation_forest": joblib.load(md / "isolation_forest.joblib"),
        "rf_incident_type": joblib.load(md / "rf_incident_type.joblib"),
        "rf_severity": joblib.load(md / "rf_severity.joblib"),
        "metadata": joblib.load(md / "metadata.joblib"),
    }
    return _MODELS


def _features_for_window(window: pd.DataFrame) -> pd.DataFrame:
    from features import make_window_features

    meta = load_models()["metadata"]
    feats = make_window_features(window, meta["baseline_traffic"])
    return pd.DataFrame([feats], columns=meta["feature_names"]).fillna(0.0)


# --------------------------------------------------------------------------- #
# Stage 2 — Anomaly detection (Isolation Forest)
# --------------------------------------------------------------------------- #
def detect_anomaly(window: pd.DataFrame) -> dict:
    """Score one telemetry window for abnormal behaviour.

    Input:
      window  DataFrame of consecutive samples for one server (the five METRICS
              columns required; `timestamp` used for the reported window bounds).

    Output (dict):
      is_anomaly        (bool)       IsolationForest flagged the window (-1)
      anomaly_score     (float 0-1)  raw novelty score mapped through the
                                     training-normal p50..p99 range, clipped
      raw_score         (float)      unnormalised -score_samples value
      anomalous_metrics (list[str])  metrics whose window mean exceeds the
                                     normal mean + 2*std (attribution heuristic)
      window_start / window_end (Timestamp)

    The detector is trained on NORMAL windows only, so it flags "unlike normal",
    not a specific incident type.
    """
    models = load_models()
    meta = models["metadata"]
    iso = models["isolation_forest"]
    X = _features_for_window(window)

    raw = float(-iso.score_samples(X)[0])
    pcts = meta["anomaly_score_percentiles"]
    span = max(pcts["p99"] - pcts["p50"], 1e-9)
    score01 = float(np.clip((raw - pcts["p50"]) / span, 0.0, 1.0))
    is_anomaly = bool(iso.predict(X)[0] == -1)

    stats = meta["normal_metric_stats"]
    anomalous = [
        m
        for m in METRICS
        if float(window[m].mean()) > stats[m]["mean"] + 2.0 * max(stats[m]["std"], 1e-9)
    ]

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": score01,
        "raw_score": raw,
        "anomalous_metrics": anomalous,
        "window_start": window["timestamp"].iloc[0],
        "window_end": window["timestamp"].iloc[-1],
    }


# --------------------------------------------------------------------------- #
# Stage 3 — Incident classification (Random Forest)
# --------------------------------------------------------------------------- #
def classify_incident(window: pd.DataFrame, anomaly: dict | None = None) -> dict:
    """Predict the incident type and severity for one window.

    Inputs:
      window   DataFrame of consecutive samples for one server.
      anomaly  optional dict from detect_anomaly (not required; kept for the
               stage interface and possible future gating).

    Output (dict):
      incident_type       (str)  "normal" or one of INCIDENT_TYPES
      severity            (str)  one of SEVERITIES ("none" when normal)
      confidence          (float 0-1)  max class probability (incident type)
      severity_confidence (float 0-1)  max class probability (severity)
      class_probabilities (dict[str, float])  per incident-type probability
    """
    models = load_models()
    rf_type = models["rf_incident_type"]
    rf_sev = models["rf_severity"]
    X = _features_for_window(window)

    proba = rf_type.predict_proba(X)[0]
    classes = list(rf_type.classes_)
    class_probabilities = {c: float(p) for c, p in zip(classes, proba)}
    incident_type = classes[int(np.argmax(proba))]
    confidence = float(np.max(proba))

    if incident_type == NORMAL_LABEL:
        severity, severity_confidence = "none", 1.0
    else:
        sev_proba = rf_sev.predict_proba(X)[0]
        sev_classes = list(rf_sev.classes_)
        severity = sev_classes[int(np.argmax(sev_proba))]
        severity_confidence = float(np.max(sev_proba))

    return {
        "incident_type": incident_type,
        "severity": severity,
        "confidence": confidence,
        "severity_confidence": severity_confidence,
        "class_probabilities": class_probabilities,
    }


# --------------------------------------------------------------------------- #
# Stage 4 — Candidate response generation
# --------------------------------------------------------------------------- #
def generate_actions(incident: dict) -> list[dict]:
    """Feasible response actions for the detected incident type.

    Delegates to actions.feasible_actions. Always includes the 'do_nothing'
    baseline as the last entry. Each dict: action_id, name, description, params.
    """
    from actions import feasible_actions

    return feasible_actions(incident.get("incident_type", NORMAL_LABEL))


# --------------------------------------------------------------------------- #
# Stage 5 — Monte Carlo simulation of one action
# --------------------------------------------------------------------------- #
def simulate_action(
    action: dict,
    incident: dict,
    telemetry: pd.DataFrame | None = None,
    n_runs: int = 1000,
    seed: int | None = None,
) -> dict:
    """Monte Carlo outcome estimate for one action against the incident.

    Delegates to simulate.simulate. `telemetry` is the pre-action window (its
    mean latency seeds the latency baseline). Returns mean + p90 for downtime,
    latency, cost and recovery probability, plus the raw `draws` arrays.

    NOTE: the simulation uses hand-authored effect priors, not a learned model.
    See src/simulate.py and STATUS_REPORT.md.
    """
    from simulate import simulate

    rng = np.random.default_rng(seed) if seed is not None else None
    return simulate(action, incident, window=telemetry, n_draws=n_runs, rng=rng)


# --------------------------------------------------------------------------- #
# Stage 6 — Risk scoring, ranking and recommendation
# --------------------------------------------------------------------------- #
def rank_actions(
    simulations: list[dict], weights: dict[str, float] | None = None
) -> dict:
    """Weighted risk score + ranking + plain-language recommendation.

    Delegates to risk.rank. Weight keys: downtime, latency, cost, recovery
    (renormalised to sum 1). Returns {ranked, recommendation, weights}; `ranked`
    is sorted ascending by risk_score (lower = safer) with a 'rank' field, and
    `recommendation` is ranked[0] plus an 'explanation' built from the numbers.
    """
    from risk import rank

    return rank(simulations, weights)


# --------------------------------------------------------------------------- #
# End-to-end demo on a real window from the dataset
# --------------------------------------------------------------------------- #
def _demo_window(df: pd.DataFrame) -> pd.DataFrame:
    """Pick a high-severity incident window from a held-out server."""
    meta = load_models()["metadata"]
    test_df = df[df["server_id"].isin(meta["test_servers"])]
    hits = test_df[
        (test_df["incident_type"] != NORMAL_LABEL) & (test_df["severity"] == "high")
    ]
    row = hits.iloc[len(hits) // 2]
    return get_window(
        df, row["server_id"], row["timestamp"] - pd.Timedelta(minutes=20)
    )


if __name__ == "__main__":
    print("=" * 72)
    print("AEGIS-AI  |  decision-support pipeline")
    print("=" * 72)

    df = generate_telemetry()
    window = _demo_window(df)
    truth = window[window["incident_type"] != NORMAL_LABEL]
    true_type = truth["incident_type"].mode().iat[0] if len(truth) else "normal"
    true_sev = truth["severity"].mode().iat[0] if len(truth) else "none"
    print(
        f"\n[1] Window: server {window['server_id'].iat[0]}, "
        f"{window['timestamp'].iat[0]} -> {window['timestamp'].iat[-1]} "
        f"({len(window)} samples)"
    )
    print(f"    ground truth: {true_type} / {true_sev}")

    anomaly = detect_anomaly(window)
    print("\n[2] Anomaly detection (IsolationForest)")
    print(f"    is_anomaly       : {anomaly['is_anomaly']}")
    print(f"    anomaly_score    : {anomaly['anomaly_score']:.2f}")
    print(f"    anomalous_metrics: {', '.join(anomaly['anomalous_metrics']) or '(none)'}")

    incident = classify_incident(window, anomaly)
    print("\n[3] Incident classification (RandomForest)")
    print(f"    incident_type    : {incident['incident_type']}  "
          f"(confidence {incident['confidence']:.2f})")
    print(f"    severity         : {incident['severity']}  "
          f"(confidence {incident['severity_confidence']:.2f})")

    actions = generate_actions(incident)
    print(f"\n[4] Candidate actions - {len(actions)}")
    for a in actions:
        print(f"    - {a['action_id']:<20} {a['name']}")

    sims = [simulate_action(a, incident, window, n_runs=1500, seed=42) for a in actions]
    print("\n[5] Monte Carlo simulation (1500 draws)")
    print(f"    {'action':<20}{'downtime(mean/p90)':>22}{'latency':>10}"
          f"{'cost$':>9}{'recovery':>10}")
    for s in sims:
        print(
            f"    {s['action_id']:<20}"
            f"{s['downtime_min_mean']:>10.1f} /{s['downtime_min_p90']:>8.1f}"
            f"{s['latency_ms_mean']:>10.0f}"
            f"{s['cost_usd_mean']:>9.0f}"
            f"{s['recovery_probability_mean']:>10.2f}"
        )

    ranking = rank_actions(sims)
    print("\n[6] Risk scoring & ranking")
    for e in ranking["ranked"]:
        print(f"    #{e['rank']}  {e['action_id']:<20} risk_score = {e['risk_score']:.3f}")

    print("\n" + "-" * 72)
    print("RECOMMENDATION")
    print("-" * 72)
    print(ranking["recommendation"]["explanation"])
