"""
AEGIS-AI — weighted risk scoring & ranking (Phase 4).

Takes the per-action simulation summaries and produces a single risk score per
action (lower = safer), a ranking, and a plain-language explanation built from
the actual simulated numbers.

Scoring method (all factors normalised across the candidate set, so the score is
relative within one incident, not an absolute):

    risk = w_downtime * norm(downtime_blend)
         + w_latency  * norm(latency_blend)
         + w_cost     * norm(cost_mean)
         + w_recovery * norm(1 - recovery_mean)

where *_blend = 0.6 * mean + 0.4 * p90 (penalise bad tails), and norm() is
min-max scaling to [0, 1] across the candidates. Weights are named and adjustable
and are renormalised to sum to 1.
"""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS: dict[str, float] = {
    "downtime": 0.30,
    "latency": 0.20,
    "cost": 0.15,
    "recovery": 0.35,
}

_TAIL_BLEND = 0.4  # weight on p90 vs mean in the blended factor


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_actions(
    sims: list[dict], weights: dict[str, float] | None = None
) -> list[dict]:
    """Attach a 'risk_score' (0-1, lower safer) to each simulation dict.

    Returns new dicts (inputs are not mutated), in the original order.
    """
    if not sims:
        return []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    downtime = np.array(
        [(1 - _TAIL_BLEND) * s["downtime_min_mean"] + _TAIL_BLEND * s["downtime_min_p90"]
         for s in sims]
    )
    latency = np.array(
        [(1 - _TAIL_BLEND) * s["latency_ms_mean"] + _TAIL_BLEND * s["latency_ms_p90"]
         for s in sims]
    )
    cost = np.array([s["cost_usd_mean"] for s in sims])
    recovery_risk = np.array([1.0 - s["recovery_probability_mean"] for s in sims])

    risk = (
        w["downtime"] * _norm(downtime)
        + w["latency"] * _norm(latency)
        + w["cost"] * _norm(cost)
        + w["recovery"] * _norm(recovery_risk)
    )

    out = []
    for s, r, dn, lt in zip(sims, risk, _norm(downtime), _norm(latency)):
        d = dict(s)
        d.pop("draws", None)  # keep the scored dicts lightweight
        d["risk_score"] = float(r)
        d["_norm_downtime"] = float(dn)
        d["_norm_latency"] = float(lt)
        out.append(d)
    return out


def _dominant_factor(best: dict, other: dict, weights: dict[str, float]) -> str:
    """Which factor most explains best beating other (largest weighted gap)."""
    gaps = {
        "shorter downtime": weights["downtime"]
        * (other["_norm_downtime"] - best["_norm_downtime"]),
        "lower post-action latency": weights["latency"]
        * (other["_norm_latency"] - best["_norm_latency"]),
        "lower cost": weights["cost"]
        * ((other["cost_usd_mean"] - best["cost_usd_mean"]) / (abs(other["cost_usd_mean"]) + 1)),
        "higher recovery probability": weights["recovery"]
        * (best["recovery_probability_mean"] - other["recovery_probability_mean"]),
    }
    return max(gaps, key=gaps.get)


def build_explanation(ranked: list[dict], weights: dict[str, float]) -> str:
    """Plain-language rationale, generated from the simulated numbers."""
    if not ranked:
        return "No candidate actions to evaluate."
    best = ranked[0]
    parts = [
        f"Recommended: {best['name']} (risk score {best['risk_score']:.2f}, "
        f"lowest of {len(ranked)} candidates).",
        f"Simulated downtime averages {best['downtime_min_mean']:.1f} min "
        f"(p90 {best['downtime_min_p90']:.1f} min), post-action latency "
        f"~{best['latency_ms_mean']:.0f} ms, incremental cost "
        f"~${best['cost_usd_mean']:.0f}, with an estimated "
        f"{best['recovery_probability_mean'] * 100:.0f}% recovery probability.",
    ]

    do_nothing = next((r for r in ranked if r["action_id"] == "do_nothing"), None)
    if do_nothing is not None and do_nothing is not best:
        parts.append(
            f"For comparison, taking no action risks ~"
            f"{do_nothing['downtime_min_mean']:.0f} min downtime "
            f"(p90 {do_nothing['downtime_min_p90']:.0f} min) at only "
            f"{do_nothing['recovery_probability_mean'] * 100:.0f}% chance of "
            f"self-recovery (risk score {do_nothing['risk_score']:.2f})."
        )

    runner_up = next((r for r in ranked[1:] if r["action_id"] != "do_nothing"), None)
    if runner_up is not None:
        parts.append(
            f"Next best remediation is {runner_up['name']} "
            f"(risk {runner_up['risk_score']:.2f}); {best['name']} wins mainly on "
            f"{_dominant_factor(best, runner_up, weights)}."
        )

    parts.append(
        "Decision-support only: AEGIS-AI does not execute this action."
    )
    return " ".join(parts)


def rank(sims: list[dict], weights: dict[str, float] | None = None) -> dict:
    """Score, sort ascending by risk, and build the recommendation + explanation.

    Returns {ranked, recommendation, weights} where `ranked` is the list of
    scored dicts (with 'rank' added) and `recommendation` is ranked[0] plus an
    'explanation' string.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    scored = score_actions(sims, w)
    scored.sort(key=lambda d: d["risk_score"])
    for i, d in enumerate(scored, start=1):
        d["rank"] = i

    recommendation = dict(scored[0]) if scored else {}
    if recommendation:
        recommendation["explanation"] = build_explanation(scored, w)

    return {"ranked": scored, "recommendation": recommendation, "weights": w}
