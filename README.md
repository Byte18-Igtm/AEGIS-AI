# AEGIS-AI

**Simulation-based AI decision-support for infrastructure incident response.**

AEGIS-AI takes a window of server telemetry, decides whether something is wrong,
identifies the probable incident type and severity, proposes feasible response
actions, simulates the likely outcome of each one, and ranks them by a weighted
risk score — then presents the least-risk recommendation with a plain-language
explanation built from the simulated numbers.

It is a **decision-support prototype only**. It never executes a response action.

---

## Architecture

```
data/synthetic_telemetry.csv
        │
        ▼
generate_telemetry ──► detect_anomaly ──► classify_incident ──► generate_actions
   (load window)        IsolationForest     RandomForest ×2        per-incident
                        (normal-only fit)   type + severity        action catalogue
                                                                        │
                                                                        ▼
                                                                 simulate_action
                                                                 Monte Carlo,
                                                                 500–2000 draws
                                                                        │
                                                                        ▼
                                                                  rank_actions
                                                             weighted risk score +
                                                             generated explanation
                                                                        │
                                                                        ▼
                                                            dashboard/app.py (Streamlit)
```

| Module | Role |
|---|---|
| `src/pipeline.py` | The six stage functions + a CLI demo (`python src/pipeline.py`). Stages 2–6 load trained models / delegate to the modules below. |
| `src/features.py` | Window → fixed feature vector (rolling stats, slopes, rates of change, threshold-violation fractions, cross-metric ratios). |
| `src/train.py` | Server-level train/test split, then fits IsolationForest + two RandomForests, saves to `models/`. |
| `src/actions.py` | Incident type → feasible action list (`ACTION_VOCAB` + a `do_nothing` baseline). |
| `src/simulate.py` | Monte Carlo "what-if" per action. **Hand-authored effect priors, not a learned model.** |
| `src/risk.py` | Weighted, normalised risk score; ranking; explanation generator. |
| `src/evaluate.py` | Held-out metrics + confusion matrices → `reports/`. |
| `dashboard/app.py` | Streamlit UI. Loads saved models only; never trains. |
| `tests/test_pipeline.py` | Scenario sanity tests (pytest). |

---

## Setup

The project uses a virtual environment named `.venv/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run each piece

```powershell
# 1. (re)generate the synthetic dataset  -> data/synthetic_telemetry.csv
python data/generate_synthetic.py

# 2. train + persist models              -> models/*.joblib
python src/train.py

# 3. evaluate on held-out servers        -> reports/evaluation_report.md + PNGs
python src/evaluate.py

# 4. run the end-to-end pipeline on one real window (prints to console)
python src/pipeline.py

# 5. scenario sanity tests
python -m pytest tests/ -v

# 6. dashboard
streamlit run dashboard/app.py
```

Steps 1–2 must run before 3–6. The repo already ships generated data and trained
models, so you can go straight to step 3 or 6.

---

## Dataset

`data/synthetic_telemetry.csv` — **synthetic**, produced by
`data/generate_synthetic.py`.

| | |
|---|---|
| Servers | 18 simulated hosts (`srv-00` … `srv-17`) |
| Duration | 21 days, one sample every 5 minutes |
| Rows | 108,864 |
| Metrics | `cpu_util`, `mem_util`, `latency_ms`, `error_rate`, `traffic_rps` |
| Labels | `incident_type` (`normal` + 5 types), `severity` (`none/low/medium/high`) |
| Incident windows | 700 total — cpu_overload 161, error_burst 137, memory_pressure 134, traffic_spike 134, latency_degradation 134 |
| Incident rows | 13.2% of all rows |
| Severity mix (windows) | low 249, medium 200, high 251 |
| Split | first 14 servers train, last 4 servers held out for test (whole servers, no leakage) |

### Why the dataset is synthetic (honest note)

Public cloud workload traces exist, but a **labelled** dataset that contains both
infrastructure incidents *and* the outcomes of multiple candidate response
actions does not exist publicly for this scope. So the dataset is generated.

To avoid inventing numbers wholesale, the **normal-operation CPU baseline is
grounded in Microsoft's Azure Public Dataset** (Cortez et al., *Resource
Central*, SOSP 2017), which reports that most production cloud VMs average
roughly 10–50% CPU with a heavy skew toward the low end. Each server's baseline
CPU mean is drawn from U(12, 42)%. The other baselines (memory, latency, error
rate, traffic, daily seasonality, weekend dips) are plausible web-service values
chosen to be internally consistent with that CPU range — only the CPU band is
claimed to be literature-grounded. Incident windows are injected with
raised-cosine ramps and a deliberate ~25% "borderline" fraction so classes are
**not** perfectly separable.

An earlier real trace (`data/system_performance_metrics.csv`, AWS EC2
cpu/mem/disk, 24 h, unlabelled) is kept in the repo but is **not used** by the
pipeline — it has no incident labels and no latency/traffic/error metrics, so it
cannot support classification.

---

## Evaluation results (held-out test servers, 4,028 windows)

Full report: `reports/evaluation_report.md` (regenerate with `python src/evaluate.py`).

**Incident-type classifier** — accuracy **0.97**, macro-F1 **0.92**

| class | precision | recall | f1 |
|---|---|---|---|
| normal | 0.99 | 0.97 | 0.98 |
| cpu_overload | 0.96 | 0.98 | 0.97 |
| memory_pressure | 0.90 | 0.99 | 0.94 |
| error_burst | 0.86 | 0.93 | 0.89 |
| latency_degradation | 0.79 | 0.97 | 0.87 |
| traffic_spike | 0.80 | 0.95 | 0.87 |

**Severity classifier** (true-incident windows) — accuracy **0.88**

| class | precision | recall | f1 |
|---|---|---|---|
| low | 0.88 | 0.89 | 0.88 |
| medium | 0.80 | 0.88 | 0.84 |
| high | 0.97 | 0.88 | 0.92 |

**Anomaly detector** (positive = window overlaps a true incident)

| precision | recall | f1 |
|---|---|---|
| 0.71 | 0.91 | 0.80 |

The anomaly detector over-flags (precision 0.71): it is trained on normal
windows only and reports "unlike normal", which is broader than "is an incident".
Borderline synthetic windows are the main source of the traffic_spike /
latency_degradation precision dip in the type classifier.

---

## Limitations

See `STATUS_REPORT.md` for the full, honest breakdown. In short: the dataset and
the Monte Carlo simulation are synthetic; classification is the only component
validated against held-out data. The simulation's action-effect priors encode
domain intuition, not measured outcomes.
