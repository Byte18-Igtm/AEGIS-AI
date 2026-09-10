"""Shared test fixtures. Adds src/ to the path and loads data/models once."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_MODELS = ROOT / "models"
_REQUIRED = [
    "isolation_forest.joblib",
    "rf_incident_type.joblib",
    "rf_severity.joblib",
    "metadata.joblib",
    "test_set.joblib",
]


def _models_ready() -> bool:
    return all((_MODELS / n).exists() for n in _REQUIRED)


requires_models = pytest.mark.skipif(
    not _models_ready(),
    reason="trained models missing — run `python src/train.py` then `python src/evaluate.py`",
)


@pytest.fixture(scope="session")
def telemetry():
    from pipeline import generate_telemetry

    return generate_telemetry()


@pytest.fixture(scope="session")
def test_windows():
    """meta_test rows (server_id, window_start, incident_type, severity, ...)."""
    return joblib.load(_MODELS / "test_set.joblib")["meta_test"]
