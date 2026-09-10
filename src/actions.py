"""
AEGIS-AI — candidate response actions (Phase 4).

Maps a detected incident type to a controlled set of feasible response actions.
Action ids match pipeline.ACTION_VOCAB, plus one extra baseline:

    do_nothing  — not a real remediation. Included as an explicit "wait and see"
                  comparison so the ranker (and the dashboard) can show what the
                  incident is expected to cost if left unmitigated. It is NOT in
                  ACTION_VOCAB by design.
"""

from __future__ import annotations

ACTION_LIBRARY: dict[str, dict[str, str]] = {
    "scale_out": {
        "name": "Scale out service replicas",
        "description": "Add compute capacity by increasing the replica count.",
    },
    "restart_service": {
        "name": "Restart affected service",
        "description": "Rolling restart of the degraded service instances.",
    },
    "reroute_traffic": {
        "name": "Reroute traffic",
        "description": "Shift load to a healthy region / pool via the load balancer.",
    },
    "rollback_deployment": {
        "name": "Roll back last deployment",
        "description": "Revert the service to the previous known-good release.",
    },
    "isolate_component": {
        "name": "Isolate affected component",
        "description": "Quarantine the failing component to stop cascading errors.",
    },
    "do_nothing": {
        "name": "Take no action (wait & observe)",
        "description": "Baseline for comparison only — no remediation is applied.",
    },
}

# feasible remediations per incident type, best-first (do_nothing appended to all)
_FEASIBLE: dict[str, list[str]] = {
    "cpu_overload": ["scale_out", "restart_service", "rollback_deployment"],
    "memory_pressure": ["restart_service", "scale_out", "rollback_deployment"],
    "traffic_spike": ["scale_out", "reroute_traffic", "restart_service"],
    "latency_degradation": [
        "reroute_traffic",
        "rollback_deployment",
        "restart_service",
        "scale_out",
    ],
    "error_burst": ["rollback_deployment", "isolate_component", "restart_service"],
    "normal": [],
}

_DEFAULT_PARAMS: dict[str, dict] = {
    "scale_out": {"replicas": "+2"},
    "restart_service": {"strategy": "rolling"},
    "reroute_traffic": {"target": "healthy_pool"},
    "rollback_deployment": {"to_revision": "previous"},
    "isolate_component": {"mode": "drain"},
    "do_nothing": {},
}


def feasible_actions(incident_type: str) -> list[dict]:
    """Return the candidate action dicts for an incident type.

    Always includes 'do_nothing' as the final baseline entry. For 'normal'
    (no incident) only 'do_nothing' is returned.

    Each dict: {action_id, name, description, params}.
    """
    ids = list(_FEASIBLE.get(incident_type, []))
    ids.append("do_nothing")
    out = []
    for aid in ids:
        meta = ACTION_LIBRARY[aid]
        out.append(
            {
                "action_id": aid,
                "name": meta["name"],
                "description": meta["description"],
                "params": dict(_DEFAULT_PARAMS.get(aid, {})),
            }
        )
    return out
