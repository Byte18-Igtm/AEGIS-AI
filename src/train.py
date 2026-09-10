"""
AEGIS-AI — model training (Phase 3).

Trains and persists:
    models/isolation_forest.joblib   novelty detector (fit on NORMAL windows only)
    models/rf_incident_type.joblib   RandomForest: normal + 5 incident types
    models/rf_severity.joblib        RandomForest: none/low/medium/high
    models/metadata.joblib           feature names, split, normalisation stats
    models/test_set.joblib           (X_test, meta_test) held-out for evaluation

The test split is made BEFORE any fitting, at the server level (whole servers are
held out) so no window from a test server is ever seen in training.

Run:  python src/train.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from features import METRICS, STRIDE, WINDOW_SIZE, build_feature_table
from pipeline import generate_telemetry

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_TEST_SERVERS = 4
RANDOM_STATE = 42


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df = generate_telemetry()

    servers = sorted(df["server_id"].unique())
    test_servers = servers[-N_TEST_SERVERS:]
    train_servers = servers[:-N_TEST_SERVERS]
    print(f"train servers: {train_servers}")
    print(f"test  servers: {test_servers}")

    train_df = df[df["server_id"].isin(train_servers)]
    test_df = df[df["server_id"].isin(test_servers)]

    # baseline traffic = mean rps of genuinely-normal training rows
    baseline_traffic = float(
        train_df.loc[train_df["incident_type"] == "normal", "traffic_rps"].mean()
    )
    print(f"baseline_traffic (train normal mean rps): {baseline_traffic:.1f}")

    X_train, meta_train = build_feature_table(
        train_df, WINDOW_SIZE, STRIDE, baseline_traffic
    )
    X_test, meta_test = build_feature_table(
        test_df, WINDOW_SIZE, STRIDE, baseline_traffic
    )
    print(f"train windows: {len(X_train)}  |  test windows: {len(X_test)}")
    print("train label balance:\n", meta_train["incident_type"].value_counts())

    normal_mask = ~meta_train["is_incident"].to_numpy()

    # ---- anomaly detector: novelty detection, fit on normal windows only ----
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train[normal_mask])

    # normalisation reference for anomaly_score: raw = -score_samples on normal
    raw_normal = -iso.score_samples(X_train[normal_mask])
    score_pcts = {
        "p50": float(np.percentile(raw_normal, 50)),
        "p95": float(np.percentile(raw_normal, 95)),
        "p99": float(np.percentile(raw_normal, 99)),
    }

    # ---- incident-type classifier (includes the "normal" class) ----
    rf_type = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_type.fit(X_train, meta_train["incident_type"])

    # ---- severity classifier (incident windows only) ----
    inc_mask = meta_train["is_incident"].to_numpy()
    rf_sev = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_sev.fit(X_train[inc_mask], meta_train.loc[inc_mask, "severity"])

    # ---- per-metric normal-window stats (for anomalous-metric attribution) ----
    normal_metric_stats = {}
    for m in METRICS:
        col = f"{m}_mean"
        vals = X_train.loc[normal_mask, col].to_numpy()
        normal_metric_stats[m] = {"mean": float(vals.mean()), "std": float(vals.std())}

    metadata = {
        "feature_names": list(X_train.columns),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "baseline_traffic": baseline_traffic,
        "train_servers": train_servers,
        "test_servers": test_servers,
        "anomaly_score_percentiles": score_pcts,
        "normal_metric_stats": normal_metric_stats,
        "incident_type_classes": list(rf_type.classes_),
        "severity_classes": list(rf_sev.classes_),
    }

    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(rf_type, MODELS_DIR / "rf_incident_type.joblib")
    joblib.dump(rf_sev, MODELS_DIR / "rf_severity.joblib")
    joblib.dump(metadata, MODELS_DIR / "metadata.joblib")
    joblib.dump({"X_test": X_test, "meta_test": meta_test}, MODELS_DIR / "test_set.joblib")

    # quick sanity readout on the held-out set
    type_acc = (rf_type.predict(X_test) == meta_test["incident_type"]).mean()
    test_inc = meta_test["is_incident"].to_numpy()
    sev_acc = (
        rf_sev.predict(X_test[test_inc]) == meta_test.loc[test_inc, "severity"]
    ).mean()
    iso_pred_anom = iso.predict(X_test) == -1
    tp = int((iso_pred_anom & test_inc).sum())
    fp = int((iso_pred_anom & ~test_inc).sum())
    fn = int((~iso_pred_anom & test_inc).sum())
    print(f"\nheld-out incident_type accuracy: {type_acc:.3f}")
    print(f"held-out severity accuracy (incident rows): {sev_acc:.3f}")
    print(
        f"held-out anomaly precision={tp/(tp+fp+1e-9):.3f} "
        f"recall={tp/(tp+fn+1e-9):.3f}"
    )
    print(f"\nsaved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
