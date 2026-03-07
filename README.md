# novawireless-governance-pipeline

### Every call gets a trust score. Every rep gets a risk profile. Every month gets a drift velocity. Your dashboard shows none of this.

---

Resolution rate tells you what the system recorded. This pipeline tells you what actually happened — and how fast the gap is growing.

The NovaWireless Governance Pipeline scores every call on five trust signal dimensions, rolls those scores up to rep, scenario, and customer levels, tracks month-over-month drift, evaluates configurable WATCH/VETO governance thresholds, and produces nine diagnostic visualizations plus a machine-readable governance report.

It's a seven-layer diagnostic architecture that turns 82,000 call records into a single question with a clear answer: **is this operation still safe to optimize?**

---

## The Numbers That Should Concern You

Across 82,440 calls over 12 months:

| Finding | Value | What It Means |
|---|---|---|
| Proxy resolution rate | 89.2% | What the dashboard says |
| True resolution rate | 46.9% | What actually happened |
| **Goodhart gap** | **42.4 pp** | The distance between the KPI and reality |
| Gamed metric trust score | 46.0 / 100 | Lowest of all 10 scenario types |
| Bandaid credit rate (gamed) | 71.1% | 7 in 10 "resolved" calls used unauthorized credits |
| Rep-level correlation: true resolution vs. gap | **-0.92** | Reps who resolve less game more — almost perfectly |
| Trust score vs. call volume | r = -0.06 | Workload isn't the driver. Gaming is. |

The trust score distribution is **bimodal**. Clean calls cluster above 80. Gaming and fraud calls create the left tail. If your operation has a left tail and you can't explain it, you have a governance problem.

---

## How It Works

### Layer 1 — Integrity Gate
Every row is validated against structural expectations. Hard failures (missing required columns, impossible values) are quarantined. Soft analytical signals (proxy-true divergence, detection flag concentrations) are flagged for scoring — not discarded.

### Layer 2 — Trust Signal Scoring
Five components, one composite score per call:

| Component | Weight | What It Measures |
|---|---|---|
| Proxy-vs-true gap | 25 pts | Direct Goodhart signal — did the label match reality? |
| Detection signal density | 20 pts | Concentration of fraud and gaming indicators |
| Credit risk score | 20 pts | Unauthorized credit pattern (bandaid = maximum risk) |
| Rep drift score | 15 pts | Behavioral risk from gaming propensity + burnout |
| Outcome risk score | 20 pts | Downstream consequences — unresolved, repeat contact, escalation |

### Layer 3 — Multi-Level Aggregation
Call scores roll up to 250 reps, 10 scenario types, and per-customer trust trajectories.

### Layer 4 — Monthly Trend Tracking
Trust score, resolution gap, bandaid rate, and detection density tracked month-over-month. **Drift velocity** — the rate of change per month — detects accelerating degradation before it shows up in quarterly reviews.

### Layer 5 — Threshold Alerts
Configurable WATCH and VETO thresholds at scenario, rep, and monthly levels. A VETO signal means stop optimizing. A WATCH signal means a human needs to look at this before the next cycle.

| Signal | WATCH | VETO |
|---|---|---|
| Scenario trust score | < 65 | < 50 |
| Resolution gap | > 50% | > 70% |
| Bandaid rate | > 20% | > 50% |
| Rep trust score | < 65 | < 60 |
| Rep resolution gap | > 45% | > 55% |
| Drift velocity | > 2 pts/month | — |

### Layer 6 — Churn Integration
Trust scores are binned into deciles and tested against customer retention. If the trust score predicts churn better than the resolution flag, you know which signal to govern with.

### Layer 7 — Evidence Artifacts
Nine diagnostic charts, a narrative summary report, a machine-readable governance JSON export, and structured threshold alert reports. Everything a reviewer needs to understand the state of the operation without running the pipeline themselves.

---

## Quick Start

```bash
pip install -r requirements.txt
python novawireless_trust_signals_pipeline.py
```

Copy the 12 monthly `calls_sanitized_2025-*.csv` files from the NovaWireless Call Center Lab into `data/`. Runtime is 1–3 minutes on 82,000+ records.

```bash
python novawireless_trust_signals_pipeline.py --dupe_policy quarantine_extras_keep_latest
python novawireless_trust_signals_pipeline.py --min_calls_for_rank 30
```

---

## What It Produces

```
output/
├── data/
│   ├── calls_clean.csv                     Integrity-passed rows
│   ├── calls_quarantine.csv                Flagged rows (preserved, not deleted)
│   ├── integrity_flags.csv                 Per-row flag detail
│   ├── calls_scored.csv                    Trust signal scores per call
│   ├── rep_summary.csv                     Rep-level risk profiles
│   ├── scenario_summary.csv                Scenario-level aggregation
│   ├── customer_summary.csv                Customer trust trajectories
│   ├── monthly_trends.csv                  Drift tracking
│   └── churn_by_trust_decile.csv           Churn integration
├── figures/
│   ├── trust_score_distribution.png        The bimodal trust histogram
│   ├── proxy_vs_true_by_scenario.png       The Goodhart gap, visualized
│   ├── scenario_drift_heatmap.png          Where drift concentrates
│   ├── credit_type_by_scenario.png         Bandaid = unauthorized suppression
│   ├── rep_trust_landscape.png             Trust vs. volume, colored by gap
│   ├── rep_signal_correlations.png         Rep-level correlation matrix
│   ├── monthly_trust_trend.png             Trust score + drift velocity
│   ├── monthly_gap_trend.png               Gap, bandaid, repeat, churn
│   └── churn_by_trust_decile.png           Does trust predict retention?
└── reports/
    ├── integrity_summary.json              Gate statistics
    ├── summary_report.txt                  Narrative report
    ├── governance_report.json              Machine-readable governance export
    ├── threshold_alerts.json               Alert data (structured)
    └── threshold_alerts.txt                Alert report (human-readable)
```

---

## Repository Structure

```
novawireless-governance-pipeline/
├── novawireless_trust_signals_pipeline.py   Main pipeline (~945 lines)
├── trust_signal_health_assessment.pdf       Companion paper
├── data/                                    Input CSVs (not committed)
├── output/                                  Pipeline outputs (gitignored)
└── README.md
```

---

## Companion Paper

> Aulabaugh, G. (2026). *Trust Signal Health Assessment of the NovaWireless Synthetic Call Center: A Diagnostic Pipeline for Detecting Proxy-Outcome Divergence in AI-Optimized Operations.* PixelKraze LLC.

---

## Ecosystem Position

This repo detects divergence through **structured metadata** — credit types, resolution flags, detection signals. The companion `novawireless-transcript-analysis` repo detects divergence through **transcript language**. Two independent signal families, one conclusion: the proxy is broken.

| Repository | Signal Family | Level |
|---|---|---|
| **novawireless-governance-pipeline** | **Structured metadata** | **Call → Rep → Scenario → Month** |
| novawireless-transcript-analysis | Transcript language | Call → Term → Category |
| NovaWireless_KPI_Drift_Observatory | Composite integrity | System-level SII |
| NovaFabric Validation Checklist | Causal validation | Friction → Outcome |

---

## Requirements

Python 3.10+ with `pandas`, `numpy`, and `matplotlib`.

---

<p align="center">
  <b>Gina Aulabaugh</b><br>
  <a href="https://www.pixelkraze.com">www.pixelkraze.com</a>
</p>
