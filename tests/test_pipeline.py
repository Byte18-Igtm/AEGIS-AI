"""
Scenario sanity tests for the AEGIS-AI pipeline.

These are not accuracy tests (see src/evaluate.py for metrics). They check that
the wired-together pipeline behaves sensibly on clear-cut cases:

  * the ranker output is well-formed and ordered
  * an obvious remediation beats "do nothing" for a severe incident
  * the classifier recognises strong, unambiguous incident windows
  * the anomaly detector scores incident windows above normal ones

Simulation draws are seeded so the ranking assertions are deterministic.
"""

from __future__ import annotations

import numpy as np

from conftest import requires_models

WEIGHTS = {"downtime": 0.30, "latency": 0.20, "cost": 0.15, "recovery": 0.35}


def _risk_by_action(incident_type: str, severity: str = "high") -> dict[str, float]:
    """Run generate -> simulate (seeded) -> rank for a synthetic incident dict."""
    from pipeline import generate_actions, rank_actions, simulate_action

    incident = {"incident_type": incident_type, "severity": severity}
    actions = generate_actions(incident)
    sims = [
        simulate_action(a, incident, telemetry=None, n_runs=2000, seed=100 + i)
        for i, a in enumerate(actions)
    ]
    ranking = rank_actions(sims, WEIGHTS)
    return {e["action_id"]: e["risk_score"] for e in ranking["ranked"]}


def _window(telemetry, meta_test, incident_type: str, severity: str, k: int = 8):
    """Yield up to k held-out windows matching (incident_type, severity)."""
    from pipeline import get_window

    rows = meta_test[
        (meta_test["incident_type"] == incident_type)
        & (meta_test["severity"] == severity)
    ].head(k)
    for _, r in rows.iterrows():
        yield get_window(telemetry, r["server_id"], r["window_start"])


# --------------------------------------------------------------------------- #
@requires_models
def test_ranking_is_wellformed_and_ordered():
    from pipeline import generate_actions, rank_actions, simulate_action

    incident = {"incident_type": "cpu_overload", "severity": "high"}
    actions = generate_actions(incident)
    sims = [simulate_action(a, incident, None, n_runs=1000, seed=7) for a in actions]
    ranking = rank_actions(sims, WEIGHTS)

    ranked = ranking["ranked"]
    assert [e["rank"] for e in ranked] == list(range(1, len(ranked) + 1))
    scores = [e["risk_score"] for e in ranked]
    assert scores == sorted(scores)
    assert ranking["recommendation"]["action_id"] == ranked[0]["action_id"]
    assert isinstance(ranking["recommendation"]["explanation"], str)
    assert len(ranking["recommendation"]["explanation"]) > 40
    assert abs(sum(ranking["weights"].values()) - 1.0) < 1e-9


@requires_models
def test_high_severity_traffic_spike_prefers_scale_out_over_doing_nothing():
    risks = _risk_by_action("traffic_spike", "high")
    assert "scale_out" in risks and "do_nothing" in risks
    assert risks["scale_out"] < risks["do_nothing"]
    # scale_out should be the safest or near-safest remediation here
    best = min(risks, key=risks.get)
    assert best in {"scale_out", "reroute_traffic"}


@requires_models
def test_high_severity_error_burst_prefers_rollback_over_doing_nothing():
    risks = _risk_by_action("error_burst", "high")
    assert risks["rollback_deployment"] < risks["do_nothing"]
    assert min(risks, key=risks.get) != "do_nothing"


@requires_models
def test_classifier_recognises_strong_cpu_overload(telemetry, test_windows):
    from pipeline import classify_incident

    preds = [
        classify_incident(w)["incident_type"]
        for w in _window(telemetry, test_windows, "cpu_overload", "high")
    ]
    assert preds, "no cpu_overload/high windows in the held-out set"
    # majority of clear high-severity windows classify correctly
    assert preds.count("cpu_overload") >= (len(preds) + 1) // 2


@requires_models
def test_anomaly_score_higher_for_incidents_than_normal(telemetry, test_windows):
    from pipeline import detect_anomaly, get_window

    inc_rows = test_windows[test_windows["is_incident"]].head(15)
    norm_rows = test_windows[~test_windows["is_incident"]].head(15)
    inc_scores = [
        detect_anomaly(get_window(telemetry, r["server_id"], r["window_start"]))[
            "anomaly_score"
        ]
        for _, r in inc_rows.iterrows()
    ]
    norm_scores = [
        detect_anomaly(get_window(telemetry, r["server_id"], r["window_start"]))[
            "anomaly_score"
        ]
        for _, r in norm_rows.iterrows()
    ]
    assert np.mean(inc_scores) > np.mean(norm_scores)
