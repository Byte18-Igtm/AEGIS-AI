"""
AEGIS-AI — explainable recommendation dashboard (Phase 5).

Streamlit prototype. Pick a scenario (a telemetry window from the held-out test
servers), and the app runs the full pipeline on it: anomaly detection, incident
classification, candidate actions, Monte Carlo simulation, and weighted risk
ranking — then shows the recommendation with a plain-language explanation
generated from the simulated numbers.

Loads saved models only. It never trains.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import WINDOW_SIZE  # noqa: E402
from pipeline import (  # noqa: E402
    classify_incident,
    detect_anomaly,
    generate_actions,
    generate_telemetry,
    get_window,
    load_models,
    rank_actions,
    simulate_action,
)
from risk import DEFAULT_WEIGHTS  # noqa: E402

METRICS = ("cpu_util", "mem_util", "latency_ms", "error_rate", "traffic_rps")

st.set_page_config(page_title="AEGIS-AI", layout="wide")


@st.cache_resource
def _models():
    return load_models()


@st.cache_data
def _data():
    return generate_telemetry()


@st.cache_data
def _scenarios():
    """Curated scenario windows from the held-out test set."""
    ts = joblib.load(ROOT / "models" / "test_set.joblib")
    meta = ts["meta_test"].copy()
    rows = []
    # a few of each incident type (spread across severities), then some normal
    for itype, g in meta[meta["is_incident"]].groupby("incident_type"):
        g = g.sort_values("severity")
        take = g.iloc[:: max(1, len(g) // 4)].head(4)
        rows.append(take)
    normals = meta[~meta["is_incident"]].sample(6, random_state=0)
    rows.append(normals)
    picks = pd.concat(rows).reset_index(drop=True)
    picks["label"] = (
        picks["incident_type"]
        + " / "
        + picks["severity"]
        + "  ·  "
        + picks["server_id"]
        + "  ·  "
        + picks["window_start"].astype(str)
    )
    return picks


def _run(window, weights, n_draws):
    anomaly = detect_anomaly(window)
    incident = classify_incident(window, anomaly)
    actions = generate_actions(incident)
    sims = [
        simulate_action(a, incident, window, n_runs=n_draws, seed=42 + i)
        for i, a in enumerate(actions)
    ]
    ranking = rank_actions(sims, weights)
    return anomaly, incident, sims, ranking


def _window_chart(window: pd.DataFrame):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 7), sharex=True)
    for ax, m in zip(axes, METRICS):
        ax.plot(window["timestamp"], window[m], marker="o", ms=3, lw=1)
        ax.set_ylabel(m, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("timestamp", fontsize=8)
    fig.tight_layout()
    return fig


def _draw_hist(sim: dict):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, key, title in (
        (axes[0], "downtime_min", "downtime (min)"),
        (axes[1], "latency_ms", "post-action latency (ms)"),
    ):
        ax.hist(sim["draws"][key], bins=40, color="#4C78A8")
        ax.axvline(np.mean(sim["draws"][key]), color="k", lw=1, ls="--")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
st.title("AEGIS-AI - infrastructure incident decision support")
st.caption(
    "Simulation-based recommendation prototype. Decision-support only - it never "
    "executes a response. Models are pre-trained (src/train.py); this app never "
    "retrains."
)

_models()  # warm cache / fail early if models missing
df = _data()
scenarios = _scenarios()

with st.sidebar:
    st.header("Scenario")
    choice = st.selectbox(
        "Telemetry window (held-out servers)",
        options=list(range(len(scenarios))),
        format_func=lambda i: scenarios.loc[i, "label"],
    )
    st.header("Risk weights")
    weights = {
        k: st.slider(k, 0.0, 1.0, float(v), 0.05)
        for k, v in DEFAULT_WEIGHTS.items()
    }
    n_draws = st.select_slider(
        "Monte Carlo draws", options=[500, 1000, 1500, 2000], value=1000
    )

pick = scenarios.loc[choice]
window = get_window(df, pick["server_id"], pick["window_start"], size=WINDOW_SIZE)
anomaly, incident, sims, ranking = _run(window, weights, n_draws)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Detected incident", incident["incident_type"],
          help=f"classifier confidence {incident['confidence']:.2f}")
c2.metric("Severity", incident["severity"],
          help=f"confidence {incident['severity_confidence']:.2f}")
c3.metric("Anomaly score", f"{anomaly['anomaly_score']:.2f}",
          delta="anomaly" if anomaly["is_anomaly"] else "normal")
c4.metric("Ground truth", f"{pick['incident_type']} / {pick['severity']}",
          help="true label for this window (shown for demo verification only)")

if anomaly["anomalous_metrics"]:
    st.write("**Metrics above normal:** " + ", ".join(anomaly["anomalous_metrics"]))

left, right = st.columns([1, 1])
with left:
    st.subheader("Telemetry window")
    st.pyplot(_window_chart(window))

with right:
    st.subheader("Ranked actions")
    table = pd.DataFrame(
        [
            {
                "rank": e["rank"],
                "action": e["action_id"],
                "risk_score": round(e["risk_score"], 3),
                "downtime_mean": round(e["downtime_min_mean"], 1),
                "downtime_p90": round(e["downtime_min_p90"], 1),
                "latency_mean": round(e["latency_ms_mean"], 0),
                "cost_mean": round(e["cost_usd_mean"], 0),
                "recovery_mean": round(e["recovery_probability_mean"], 2),
            }
            for e in ranking["ranked"]
        ]
    ).set_index("rank")
    st.dataframe(table, width="stretch")

    rec = ranking["recommendation"]
    st.success(f"**Recommendation: {rec['name']}**")
    st.write(rec["explanation"])

st.subheader("Simulated outcome distributions")
sims_by_id = {s["action_id"]: s for s in sims}
for e in ranking["ranked"]:
    s = sims_by_id[e["action_id"]]
    with st.expander(f"#{e['rank']}  {e['name']}  -  risk {e['risk_score']:.3f}"):
        cc = st.columns(4)
        cc[0].metric("downtime mean / p90",
                     f"{s['downtime_min_mean']:.1f} / {s['downtime_min_p90']:.1f} min")
        cc[1].metric("latency mean / p90",
                     f"{s['latency_ms_mean']:.0f} / {s['latency_ms_p90']:.0f} ms")
        cc[2].metric("cost mean / p90",
                     f"${s['cost_usd_mean']:.0f} / ${s['cost_usd_p90']:.0f}")
        cc[3].metric("recovery prob mean / p90",
                     f"{s['recovery_probability_mean']:.2f} / {s['recovery_probability_p90']:.2f}")
        st.pyplot(_draw_hist(s))
