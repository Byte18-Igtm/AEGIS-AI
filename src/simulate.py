"""
AEGIS-AI — Monte Carlo response simulation (Phase 4).

For a single (action, incident) pair, draw many possible outcomes and summarise
them. Outcomes covered: downtime (minutes), post-action latency (ms), incremental
cost (USD), and recovery probability (latent success rate).

THIS IS A SYNTHETIC "WHAT-IF" MODEL, NOT A LEARNED ONE.
The per-action effect priors below are hand-authored estimates of how each
action typically behaves against each incident type, scaled by severity. They
encode domain intuition (e.g. scale_out fixes cpu_overload well but not a bad
deploy) so the ranker has something meaningful to compare. They are not fitted
to data and should be read as "plausible defaults", not measured truth.

Public API:
    simulate(action, incident, window=None, n_draws=1000, rng=None) -> dict
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_DRAWS = 1000
_SEV_LEVEL = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Base effect priors per action. Keys:
#   downtime_min      : (mean, cv)  lognormal-ish blip length at severity 0
#   downtime_sev_mult : extra multiplier per severity level
#   latency_factor    : dict incident_type -> (lo, hi) multiplier on current latency
#   latency_factor_default : (lo, hi) for incident types not listed
#   cost_usd          : (mean, sd) incremental cost; may scale with severity
#   cost_sev_scaled   : bool
#   recovery          : dict incident_type -> mean latent recovery prob at sev 0
#   recovery_default  : mean latent recovery prob
#   recovery_sev_penalty : subtracted per severity level
_PRIORS: dict[str, dict] = {
    "scale_out": {
        "downtime_min": (1.2, 0.5),
        "downtime_sev_mult": 0.15,
        "latency_factor": {
            "cpu_overload": (0.45, 0.70),
            "traffic_spike": (0.40, 0.65),
        },
        "latency_factor_default": (0.85, 0.98),
        "cost_usd": (34.0, 8.0),
        "cost_sev_scaled": True,
        "recovery": {
            "cpu_overload": 0.90,
            "traffic_spike": 0.88,
            "memory_pressure": 0.55,
            "latency_degradation": 0.60,
            "error_burst": 0.32,
        },
        "recovery_default": 0.5,
        "recovery_sev_penalty": 0.06,
    },
    "restart_service": {
        "downtime_min": (3.0, 0.5),
        "downtime_sev_mult": 0.35,
        "latency_factor": {
            "memory_pressure": (0.45, 0.70),
            "error_burst": (0.55, 0.85),
        },
        "latency_factor_default": (0.80, 1.0),
        "cost_usd": (4.0, 2.0),
        "cost_sev_scaled": False,
        "recovery": {
            "memory_pressure": 0.82,
            "error_burst": 0.58,
            "cpu_overload": 0.50,
            "latency_degradation": 0.52,
            "traffic_spike": 0.38,
        },
        "recovery_default": 0.5,
        "recovery_sev_penalty": 0.08,
    },
    "reroute_traffic": {
        "downtime_min": (0.8, 0.6),
        "downtime_sev_mult": 0.10,
        "latency_factor": {
            "latency_degradation": (0.50, 0.78),
            "traffic_spike": (0.55, 0.80),
        },
        "latency_factor_default": (0.88, 1.0),
        "cost_usd": (12.0, 4.0),
        "cost_sev_scaled": True,
        "recovery": {
            "traffic_spike": 0.78,
            "latency_degradation": 0.72,
            "cpu_overload": 0.42,
            "memory_pressure": 0.35,
            "error_burst": 0.45,
        },
        "recovery_default": 0.45,
        "recovery_sev_penalty": 0.05,
    },
    "rollback_deployment": {
        "downtime_min": (5.0, 0.45),
        "downtime_sev_mult": 0.25,
        "latency_factor": {
            "latency_degradation": (0.40, 0.68),
            "error_burst": (0.35, 0.60),
        },
        "latency_factor_default": (0.80, 0.95),
        "cost_usd": (18.0, 6.0),
        "cost_sev_scaled": False,
        "recovery": {
            "error_burst": 0.90,
            "latency_degradation": 0.80,
            "cpu_overload": 0.50,
            "memory_pressure": 0.50,
            "traffic_spike": 0.30,
        },
        "recovery_default": 0.5,
        "recovery_sev_penalty": 0.04,
    },
    "isolate_component": {
        "downtime_min": (1.5, 0.5),
        "downtime_sev_mult": 0.15,
        "latency_factor": {
            "error_burst": (0.55, 0.82),
        },
        "latency_factor_default": (0.88, 1.0),
        "cost_usd": (20.0, 5.0),
        "cost_sev_scaled": True,
        "recovery": {
            "error_burst": 0.80,
            "latency_degradation": 0.50,
            "cpu_overload": 0.38,
            "memory_pressure": 0.40,
            "traffic_spike": 0.40,
        },
        "recovery_default": 0.4,
        "recovery_sev_penalty": 0.05,
    },
    # do_nothing: downtime = how long the incident is expected to run unmitigated,
    # scaling hard with severity; latency stays elevated; no direct cost; recovery
    # = chance it self-resolves within the observation horizon.
    "do_nothing": {
        "downtime_min": (9.0, 0.7),
        "downtime_sev_mult": 1.4,
        "latency_factor": {},
        "latency_factor_default": (1.0, 1.35),
        "cost_usd": (0.0, 0.0),
        "cost_sev_scaled": False,
        "recovery": {},
        "recovery_default": 0.55,
        "recovery_sev_penalty": 0.15,
    },
}

_DEFAULT_LATENCY_MS = 300.0


def _draws_lognormal(rng, mean: float, cv: float, n: int) -> np.ndarray:
    """Positive draws with the given mean and coefficient of variation."""
    if mean <= 0:
        return np.zeros(n)
    sigma = np.sqrt(np.log(1.0 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, n)


def simulate(
    action: dict,
    incident: dict,
    window=None,
    n_draws: int = DEFAULT_N_DRAWS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte Carlo outcome estimate for one action against one incident.

    Inputs:
      action    dict from actions.feasible_actions (needs 'action_id', 'name').
      incident  dict from classify_incident (needs 'incident_type', 'severity').
      window    optional telemetry window DataFrame; its mean latency_ms is used
                as the pre-action latency baseline. Falls back to 300 ms.
      n_draws   number of Monte Carlo draws (spec range 500-2000).
      rng       optional numpy Generator for reproducibility.

    Returns a dict with, for each of downtime_min / latency_ms / cost_usd /
    recovery_probability: a `_mean` and a `_p90`, plus `draws` (the raw arrays,
    for charting) and echoes of action_id / name / n_draws / incident_type /
    severity.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_draws = int(np.clip(n_draws, 100, 5000))

    aid = action["action_id"]
    p = _PRIORS.get(aid, _PRIORS["restart_service"])
    itype = incident.get("incident_type", "normal")
    sev = _SEV_LEVEL.get(incident.get("severity", "none"), 1)

    if window is not None and len(window) and "latency_ms" in getattr(window, "columns", []):
        base_latency = float(window["latency_ms"].mean())
    else:
        base_latency = _DEFAULT_LATENCY_MS

    # ---- downtime ----
    dt_mean, dt_cv = p["downtime_min"]
    dt_mean = dt_mean * (1.0 + p["downtime_sev_mult"] * sev)
    downtime = _draws_lognormal(rng, dt_mean, dt_cv, n_draws)

    # ---- post-action latency ----
    lo, hi = p["latency_factor"].get(itype, p["latency_factor_default"])
    # widen a little with severity, clamp to sane band
    spread = 0.05 * sev
    factor = rng.uniform(max(0.05, lo - spread), hi + spread, n_draws)
    latency = np.clip(base_latency * factor, 5.0, None)

    # ---- incremental cost ----
    c_mean, c_sd = p["cost_usd"]
    if p["cost_sev_scaled"]:
        c_mean = c_mean * (1.0 + 0.25 * sev)
    cost = np.clip(rng.normal(c_mean, c_sd, n_draws), 0.0, None)

    # ---- recovery probability (latent, Beta around a prior mean) ----
    r_mean = p["recovery"].get(itype, p["recovery_default"])
    r_mean = float(np.clip(r_mean - p["recovery_sev_penalty"] * sev, 0.03, 0.985))
    conc = 25.0  # Beta concentration; higher => tighter around r_mean
    recovery = rng.beta(r_mean * conc, (1.0 - r_mean) * conc, n_draws)

    def ms(x):
        return float(np.mean(x))

    def p90(x):
        return float(np.percentile(x, 90))

    return {
        "action_id": aid,
        "name": action.get("name", aid),
        "incident_type": itype,
        "severity": incident.get("severity", "none"),
        "n_draws": n_draws,
        "downtime_min_mean": ms(downtime),
        "downtime_min_p90": p90(downtime),
        "latency_ms_mean": ms(latency),
        "latency_ms_p90": p90(latency),
        "cost_usd_mean": ms(cost),
        "cost_usd_p90": p90(cost),
        "recovery_probability_mean": ms(recovery),
        "recovery_probability_p90": p90(recovery),
        "draws": {
            "downtime_min": downtime,
            "latency_ms": latency,
            "cost_usd": cost,
            "recovery_probability": recovery,
        },
    }
