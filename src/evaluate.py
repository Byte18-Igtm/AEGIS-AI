"""
AEGIS-AI — model evaluation (Phase 6).

Evaluates the trained models on the held-out test servers (models/test_set.joblib,
built by src/train.py from servers never seen in training):

  * incident-type classifier : classification_report + confusion matrix
  * severity classifier       : classification_report + confusion matrix
                                (on true-incident windows only)
  * anomaly detector          : precision / recall / F1, treating "any incident
                                window" as the positive class

Writes:
  reports/evaluation_report.md
  reports/confusion_incident_type.png
  reports/confusion_severity.png

Run:  python src/evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def _fmt_report(title: str, report: dict) -> str:
    lines = [f"### {title}", "", "| class | precision | recall | f1 | support |",
             "|---|---|---|---|---|"]
    for key, val in report.items():
        if isinstance(val, dict):
            lines.append(
                f"| {key} | {val['precision']:.3f} | {val['recall']:.3f} | "
                f"{val['f1-score']:.3f} | {int(val['support'])} |"
            )
        else:
            lines.append(f"| **{key}** | | | {val:.3f} | |")
    lines.append("")
    return "\n".join(lines)


def _save_cm(y_true, y_pred, labels, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    iso = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    rf_type = joblib.load(MODELS_DIR / "rf_incident_type.joblib")
    rf_sev = joblib.load(MODELS_DIR / "rf_severity.joblib")
    meta = joblib.load(MODELS_DIR / "metadata.joblib")
    ts = joblib.load(MODELS_DIR / "test_set.joblib")
    X_test, meta_test = ts["X_test"], ts["meta_test"]

    md = ["# AEGIS-AI — Evaluation Report", "",
          f"Held-out test servers: `{meta['test_servers']}` "
          f"({len(X_test)} windows, none seen in training).", ""]

    # ---------------- incident type ----------------
    yt = meta_test["incident_type"].to_numpy()
    yp = rf_type.predict(X_test)
    type_labels = list(rf_type.classes_)
    rep = classification_report(yt, yp, labels=type_labels, output_dict=True, zero_division=0)
    txt = classification_report(yt, yp, labels=type_labels, zero_division=0)
    print("=== incident_type ===\n", txt)
    md.append(_fmt_report("Incident-type classifier", rep))
    md.append("![incident-type confusion](confusion_incident_type.png)\n")
    _save_cm(yt, yp, type_labels, REPORTS_DIR / "confusion_incident_type.png",
             "Incident type — held-out")

    # ---------------- severity (incident windows only) ----------------
    inc = meta_test["is_incident"].to_numpy()
    yts = meta_test.loc[inc, "severity"].to_numpy()
    yps = rf_sev.predict(X_test[inc])
    sev_labels = [s for s in ["low", "medium", "high"] if s in set(yts) | set(yps)]
    rep_s = classification_report(yts, yps, labels=sev_labels, output_dict=True, zero_division=0)
    txt_s = classification_report(yts, yps, labels=sev_labels, zero_division=0)
    print("=== severity (true-incident windows) ===\n", txt_s)
    md.append(_fmt_report("Severity classifier (true-incident windows only)", rep_s))
    md.append("![severity confusion](confusion_severity.png)\n")
    _save_cm(yts, yps, sev_labels, REPORTS_DIR / "confusion_severity.png",
             "Severity — held-out incident windows")

    # ---------------- anomaly detector ----------------
    y_true_anom = inc.astype(int)
    y_pred_anom = (iso.predict(X_test) == -1).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_anom, y_pred_anom, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true_anom, y_pred_anom)
    tn, fp, fn, tp = cm.ravel()
    print(f"=== anomaly detector ===\nprecision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    md += [
        "### Anomaly detector (IsolationForest)",
        "",
        "Positive class = window overlaps a true incident. The detector is "
        "trained on normal windows only, so this measures how well 'unlike "
        "normal' lines up with 'is an incident'.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| precision | {p:.3f} |",
        f"| recall | {r:.3f} |",
        f"| f1 | {f1:.3f} |",
        f"| true pos / false pos | {tp} / {fp} |",
        f"| false neg / true neg | {fn} / {tn} |",
        "",
    ]

    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REPORTS_DIR / 'evaluation_report.md'} and confusion PNGs")


if __name__ == "__main__":
    main()
