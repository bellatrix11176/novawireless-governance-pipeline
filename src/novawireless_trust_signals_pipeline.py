#!/usr/bin/env python3
"""
novawireless_trust_signals_pipeline.py

TRUST SIGNAL HEALTH PIPELINE (Lab-Aligned, Enhanced v2)

Output structure:
  output/
  ├── data/           (CSVs: scored calls, summaries, trends)
  ├── figures/        (PNG charts, 9 total)
  └── reports/        (JSON + TXT: governance, alerts, summary)

Usage:
  python novawireless_trust_signals_pipeline.py
  python novawireless_trust_signals_pipeline.py --dupe_policy quarantine_extras_keep_latest
  python novawireless_trust_signals_pipeline.py --min_calls_for_rank 30
"""

from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo-root helpers ──

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(50):
        if (cur / "data").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not find repo root containing data/.")

def ensure_output_dirs(repo_root: Path) -> Dict[str, Path]:
    base = repo_root / "output"
    dirs = {"base": base, "data": base / "data", "figures": base / "figures", "reports": base / "reports"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

# ── Constants ──

REQUIRED_COLUMNS = [
    "call_id", "call_date", "scenario", "call_type", "rep_id", "customer_id",
    "true_resolution", "resolution_flag", "credit_applied", "credit_type", "credit_authorized",
]
KNOWN_SCENARIOS = {
    "clean", "unresolvable_clean", "gamed_metric", "fraud_store_promo", "fraud_line_add",
    "fraud_hic_exchange", "fraud_care_promo", "activation_clean", "activation_failed", "line_add_legitimate",
}
VALID_CREDIT_TYPES = {"none", "courtesy", "service_credit", "bandaid", "dispute_credit", "fee_waiver"}
DETECTION_FLAG_COLS = [
    "imei_mismatch_flag", "nrf_generated_flag", "promo_override_post_call",
    "line_added_no_usage_flag", "line_added_same_day_store", "rep_aware_gaming",
]
FRAUD_SCENARIOS = {"fraud_store_promo", "fraud_line_add", "fraud_hic_exchange", "fraud_care_promo"}
GAMING_SCENARIOS = {"gamed_metric"}

# ── Threshold config ──

@dataclass
class ThresholdConfig:
    trust_score_veto: float = 50.0
    trust_score_watch: float = 65.0
    resolution_gap_veto: float = 0.70
    resolution_gap_watch: float = 0.50
    bandaid_rate_veto: float = 0.50
    bandaid_rate_watch: float = 0.20
    rep_trust_veto: float = 60.0
    rep_trust_watch: float = 65.0
    rep_gap_veto: float = 0.55
    rep_gap_watch: float = 0.45
    drift_velocity_watch: float = 2.0

# ── Data loading ──

def load_monthly_files(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("calls_sanitized_*.csv"))
    if not files:
        raise FileNotFoundError(f"No calls_sanitized_*.csv in {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["_source_file"] = f.name
        frames.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(files)} files\n")
    return combined

# ── Integrity Gate ──

@dataclass
class IntegrityConfig:
    required_columns: List[str] = field(default_factory=lambda: list(REQUIRED_COLUMNS))
    unique_key: str = "call_id"
    known_scenarios: set = field(default_factory=lambda: set(KNOWN_SCENARIOS))
    valid_credit_types: set = field(default_factory=lambda: set(VALID_CREDIT_TYPES))

def _coerce_flag(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    s = s.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "t", "yes", "y"}).astype(int)

def build_integrity_flags(df: pd.DataFrame, cfg: IntegrityConfig) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    missing = [c for c in cfg.required_columns if c not in df.columns]
    flags["flag_missing_required_columns"] = len(missing) > 0
    for c in cfg.required_columns:
        if c in df.columns:
            flags[f"flag_null_{c}"] = df[c].isna() | (df[c].astype(str).str.strip() == "")
    if "scenario" in df.columns:
        flags["flag_unknown_scenario"] = ~df["scenario"].isin(cfg.known_scenarios)
    if "credit_type" in df.columns:
        flags["flag_invalid_credit_type"] = ~df["credit_type"].astype(str).str.strip().isin(cfg.valid_credit_types)
    if {"credit_type", "credit_authorized"}.issubset(df.columns):
        flags["flag_bandaid_marked_authorized"] = (
            (df["credit_type"].astype(str).str.strip() == "bandaid") & (_coerce_flag(df["credit_authorized"]) == 1)
        )
    if {"resolution_flag", "true_resolution"}.issubset(df.columns):
        flags["flag_proxy_true_divergence"] = (
            (_coerce_flag(df["resolution_flag"]) == 1) & (_coerce_flag(df["true_resolution"]) == 0)
        )
    if "scenario" in df.columns:
        is_clean = df["scenario"].isin({"clean", "activation_clean", "line_add_legitimate"})
        det = [c for c in DETECTION_FLAG_COLS if c in df.columns]
        if det:
            any_det = df[det].apply(lambda col: _coerce_flag(col), axis=0).any(axis=1)
            flags["flag_detection_on_clean_scenario"] = is_clean & any_det
    flags["flag_duplicate_call_id"] = False
    return flags

def apply_dupe_policy(df, cfg, dupe_policy):
    stats = {"dupe_policy": dupe_policy, "duplicate_ids_count": 0,
             "duplicate_rows_involved": 0, "duplicate_rows_quarantined": 0}
    if cfg.unique_key not in df.columns:
        return pd.Series(False, index=df.index), stats
    dup_mask = df.duplicated(subset=[cfg.unique_key], keep=False)
    involved = df.index[dup_mask]
    stats["duplicate_rows_involved"] = int(len(involved))
    if len(involved) == 0:
        return pd.Series(False, index=df.index), stats
    stats["duplicate_ids_count"] = int(df.loc[dup_mask, cfg.unique_key].nunique())
    if dupe_policy == "quarantine_all":
        out = pd.Series(False, index=df.index); out.loc[involved] = True
        stats["duplicate_rows_quarantined"] = stats["duplicate_rows_involved"]
        return out, stats
    if dupe_policy in {"quarantine_extras_keep_latest", "quarantine_extras_keep_first"}:
        sort_col = "call_date" if "call_date" in df.columns else cfg.unique_key
        asc = dupe_policy == "quarantine_extras_keep_first"
        sdf = df.sort_values([cfg.unique_key, sort_col], ascending=[True, asc])
        keeper = sdf.drop_duplicates(subset=[cfg.unique_key], keep="first").index
        extras = sdf.index.difference(keeper)
        out = pd.Series(False, index=df.index); out.loc[extras] = True
        stats["duplicate_rows_quarantined"] = int(len(extras))
        return out, stats
    raise ValueError(f"Unknown dupe_policy: {dupe_policy}")

def run_integrity_gate(df, out_dirs, cfg, dupe_policy):
    flags = build_integrity_flags(df, cfg)
    dupe_q, dupe_stats = apply_dupe_policy(df, cfg, dupe_policy)
    flags["flag_duplicate_call_id"] = dupe_q
    hard = [c for c in flags.columns if c.startswith("flag_") and c not in
            {"flag_proxy_true_divergence", "flag_detection_on_clean_scenario"}]
    flags["any_flag"] = flags[hard].any(axis=1) if hard else False
    clean_df = df.loc[~flags["any_flag"]].copy()
    quar_df = df.loc[flags["any_flag"]].copy()
    clean_df.to_csv(out_dirs["data"] / "calls_clean.csv", index=False)
    quar_df.to_csv(out_dirs["data"] / "calls_quarantine.csv", index=False)
    flags.to_csv(out_dirs["data"] / "integrity_flags.csv", index=False)
    all_fc = [c for c in flags.columns if c.startswith("flag_") and c != "any_flag"]
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_total": int(len(df)), "rows_clean": int(len(clean_df)),
        "rows_quarantined": int(len(quar_df)),
        "quarantine_rate": float(flags["any_flag"].mean()),
        "flag_rates": {c: float(flags[c].mean()) for c in all_fc},
        "soft_signal_rates": {
            "proxy_true_divergence": float(flags.get("flag_proxy_true_divergence", pd.Series(0)).mean()),
            "detection_on_clean_scenario": float(flags.get("flag_detection_on_clean_scenario", pd.Series(0)).mean()),
        },
        "months_loaded": int(df["_source_file"].nunique()) if "_source_file" in df.columns else 0,
        **dupe_stats,
    }
    save_json(out_dirs["reports"] / "integrity_summary.json", summary)
    return {"clean_df": clean_df, "quarantine_df": quar_df, "summary": summary}

# ── Trust Signal Scoring ──

def _safe_coerce(df, col, default=0):
    return _coerce_flag(df[col]) if col in df.columns else pd.Series(default, index=df.index)

def _safe_numeric(df, col, default=0.0):
    return pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df.columns else pd.Series(default, index=df.index, dtype=float)

def compute_trust_signals(df):
    out = df.copy()
    proxy = _safe_coerce(out, "resolution_flag"); true = _safe_coerce(out, "true_resolution")
    out["proxy_vs_true_gap"] = ((proxy == 1) & (true == 0)).astype(int)
    det = [c for c in DETECTION_FLAG_COLS if c in out.columns]
    out["detection_signal_density"] = (
        out[det].apply(lambda col: _coerce_flag(col), axis=0).sum(axis=1) / len(det) if det else 0.0
    )
    ca = _safe_coerce(out, "credit_applied"); cauth = _safe_coerce(out, "credit_authorized")
    is_b = (out["credit_type"].astype(str).str.strip() == "bandaid") if "credit_type" in out.columns else pd.Series(False, index=out.index)
    is_u = (ca == 1) & (cauth == 0)
    cr = pd.Series(0.0, index=out.index, dtype=float)
    cr = cr.where(~(ca == 1), 0.1); cr = cr.where(~is_u, 0.7); cr = cr.where(~is_b, 1.0)
    out["credit_risk_score"] = cr
    out["rep_drift_score"] = np.clip(0.65 * _safe_numeric(out, "rep_gaming_propensity") + 0.35 * _safe_numeric(out, "rep_burnout_level"), 0, 1)
    out["outcome_risk_score"] = np.clip(
        0.30 * (true == 0).astype(int) + 0.30 * _safe_coerce(out, "repeat_contact_30d")
        + 0.15 * _safe_coerce(out, "repeat_contact_31_60d") + 0.25 * _safe_coerce(out, "escalation_flag"), 0, 1)
    out["call_trust_score"] = np.clip(
        100 - 25*out["proxy_vs_true_gap"] - 20*out["detection_signal_density"]
        - 20*out["credit_risk_score"] - 15*out["rep_drift_score"] - 20*out["outcome_risk_score"], 0, 100).round(2)
    return out

# ── Aggregation ──

def summarize_by_rep(df, min_calls=1):
    if "rep_id" not in df.columns: return pd.DataFrame()
    agg = df.groupby("rep_id", dropna=False).agg(
        calls=("call_id","count"),
        proxy_resolution_rate=("resolution_flag", lambda s: _coerce_flag(s).mean()),
        true_resolution_rate=("true_resolution", lambda s: _coerce_flag(s).mean()),
        proxy_true_gap_rate=("proxy_vs_true_gap","mean"),
        repeat_30d_rate=("repeat_contact_30d", lambda s: _coerce_flag(s).mean()) if "repeat_contact_30d" in df.columns else ("call_id", lambda s: 0.0),
        escalation_rate=("escalation_flag", lambda s: _coerce_flag(s).mean()) if "escalation_flag" in df.columns else ("call_id", lambda s: 0.0),
        credit_risk_avg=("credit_risk_score","mean"),
        detection_density_avg=("detection_signal_density","mean"),
        rep_drift_avg=("rep_drift_score","mean"),
        outcome_risk_avg=("outcome_risk_score","mean"),
        trust_score_avg=("call_trust_score","mean"),
        gaming_propensity_avg=("rep_gaming_propensity", lambda s: pd.to_numeric(s,errors="coerce").mean()) if "rep_gaming_propensity" in df.columns else ("call_id", lambda s: np.nan),
        burnout_avg=("rep_burnout_level", lambda s: pd.to_numeric(s,errors="coerce").mean()) if "rep_burnout_level" in df.columns else ("call_id", lambda s: np.nan),
    ).reset_index()
    agg["resolution_gap"] = (agg["proxy_resolution_rate"] - agg["true_resolution_rate"]).round(4)
    return agg[agg["calls"] >= min_calls].sort_values("trust_score_avg").reset_index(drop=True)

def summarize_by_scenario(df):
    if "scenario" not in df.columns: return pd.DataFrame()
    agg = df.groupby("scenario", dropna=False).agg(
        calls=("call_id","count"), pct_of_total=("call_id","count"),
        proxy_resolution_rate=("resolution_flag", lambda s: _coerce_flag(s).mean()),
        true_resolution_rate=("true_resolution", lambda s: _coerce_flag(s).mean()),
        proxy_true_gap_rate=("proxy_vs_true_gap","mean"),
        credit_applied_rate=("credit_applied", lambda s: _coerce_flag(s).mean()),
        bandaid_rate=("credit_type", lambda s: (s.astype(str).str.strip()=="bandaid").mean()),
        detection_density_avg=("detection_signal_density","mean"),
        trust_score_avg=("call_trust_score","mean"),
        outcome_risk_avg=("outcome_risk_score","mean"),
    ).reset_index()
    agg["pct_of_total"] = (agg["calls"] / max(len(df),1) * 100).round(2)
    agg["resolution_gap"] = (agg["proxy_resolution_rate"] - agg["true_resolution_rate"]).round(4)
    return agg.sort_values("trust_score_avg").reset_index(drop=True)

def summarize_by_customer(df):
    if "customer_id" not in df.columns: return pd.DataFrame()
    spec = {"calls":("call_id","count"), "avg_trust_score":("call_trust_score","mean"),
            "min_trust_score":("call_trust_score","min"), "max_trust_score":("call_trust_score","max"),
            "trust_score_std":("call_trust_score","std"),
            "avg_outcome_risk":("outcome_risk_score","mean"), "avg_proxy_gap":("proxy_vs_true_gap","mean")}
    if "scenario" in df.columns: spec["distinct_scenarios"] = ("scenario","nunique")
    if "repeat_contact_30d" in df.columns:
        spec["any_repeat_30d"] = ("repeat_contact_30d", lambda s: _coerce_flag(s).max())
        spec["repeat_30d_rate"] = ("repeat_contact_30d", lambda s: _coerce_flag(s).mean())
    if "customer_churn_risk_effective" in df.columns:
        spec["avg_churn_risk"] = ("customer_churn_risk_effective", lambda s: pd.to_numeric(s,errors="coerce").mean())
    if "customer_is_churned" in df.columns:
        spec["is_churned"] = ("customer_is_churned", lambda s: _coerce_flag(s).max())
    if "customer_trust_baseline" in df.columns:
        spec["min_trust_baseline"] = ("customer_trust_baseline", lambda s: pd.to_numeric(s,errors="coerce").min())
    if "call_date" in df.columns:
        spec["first_call"] = ("call_date","min"); spec["last_call"] = ("call_date","max")
    agg = df.groupby("customer_id", dropna=False).agg(**spec).reset_index()
    return agg.sort_values(["calls","avg_trust_score"], ascending=[False,True]).reset_index(drop=True)

# ── Monthly Trend Tracking ──

def compute_monthly_trends(df):
    if "call_date" not in df.columns: return pd.DataFrame()
    df = df.copy()
    df["_month"] = pd.to_datetime(df["call_date"], errors="coerce").dt.to_period("M")
    df = df.dropna(subset=["_month"])
    spec = {"calls":("call_id","count"), "trust_score_avg":("call_trust_score","mean"),
            "trust_score_median":("call_trust_score","median"),
            "trust_score_p10":("call_trust_score", lambda s: s.quantile(0.10)),
            "proxy_resolution_rate":("resolution_flag", lambda s: _coerce_flag(s).mean()),
            "true_resolution_rate":("true_resolution", lambda s: _coerce_flag(s).mean()),
            "proxy_true_gap_rate":("proxy_vs_true_gap","mean"),
            "detection_density_avg":("detection_signal_density","mean"),
            "credit_risk_avg":("credit_risk_score","mean"),
            "outcome_risk_avg":("outcome_risk_score","mean"),
            "rep_drift_avg":("rep_drift_score","mean")}
    if "credit_type" in df.columns:
        spec["bandaid_rate"] = ("credit_type", lambda s: (s.astype(str).str.strip()=="bandaid").mean())
    if "repeat_contact_30d" in df.columns:
        spec["repeat_30d_rate"] = ("repeat_contact_30d", lambda s: _coerce_flag(s).mean())
    if "escalation_flag" in df.columns:
        spec["escalation_rate"] = ("escalation_flag", lambda s: _coerce_flag(s).mean())
    if "customer_is_churned" in df.columns:
        spec["churn_rate"] = ("customer_is_churned", lambda s: _coerce_flag(s).mean())
    m = df.groupby("_month").agg(**spec).reset_index()
    m["_month"] = m["_month"].astype(str)
    m["resolution_gap"] = (m["proxy_resolution_rate"] - m["true_resolution_rate"]).round(4)
    m["trust_velocity"] = m["trust_score_avg"].diff().round(3)
    m["gap_velocity"] = m["resolution_gap"].diff().round(4)
    if "bandaid_rate" in m.columns: m["bandaid_velocity"] = m["bandaid_rate"].diff().round(4)
    return m

# ── Threshold Alerts ──

def _check_threshold(alerts, level, etype, entity, signal, val, th, direction):
    triggered = (val < th) if direction == "below" else (val > th)
    if triggered:
        fmt = f"{val:.1f}" if signal == "trust_score" else f"{val:.1%}"
        th_fmt = f"{th}" if signal == "trust_score" else f"{th:.0%}"
        alerts.append({"level": level, "entity_type": etype, "entity": entity,
                       "signal": signal, "value": round(val, 4), "threshold": th,
                       "message": f"{etype.title()} '{entity}' {signal} {fmt} {'below' if direction=='below' else 'above'} {level} threshold {th_fmt}"})

def run_threshold_alerts(scenario_summary, rep_summary, monthly_trends, thresholds):
    alerts = []
    if not scenario_summary.empty:
        for _, r in scenario_summary.iterrows():
            s, t, g, b = r["scenario"], r.get("trust_score_avg",100), r.get("resolution_gap",0), r.get("bandaid_rate",0)
            _check_threshold(alerts, "VETO", "scenario", s, "trust_score", t, thresholds.trust_score_veto, "below")
            _check_threshold(alerts, "WATCH", "scenario", s, "trust_score", t, thresholds.trust_score_watch, "below")
            _check_threshold(alerts, "VETO", "scenario", s, "resolution_gap", g, thresholds.resolution_gap_veto, "above")
            _check_threshold(alerts, "WATCH", "scenario", s, "resolution_gap", g, thresholds.resolution_gap_watch, "above")
            _check_threshold(alerts, "VETO", "scenario", s, "bandaid_rate", b, thresholds.bandaid_rate_veto, "above")
            _check_threshold(alerts, "WATCH", "scenario", s, "bandaid_rate", b, thresholds.bandaid_rate_watch, "above")
    if not rep_summary.empty:
        for _, r in rep_summary.iterrows():
            rid, t, g = r["rep_id"], r.get("trust_score_avg",100), r.get("resolution_gap",0)
            _check_threshold(alerts, "VETO", "rep", rid, "trust_score", t, thresholds.rep_trust_veto, "below")
            _check_threshold(alerts, "WATCH", "rep", rid, "trust_score", t, thresholds.rep_trust_watch, "below")
            _check_threshold(alerts, "VETO", "rep", rid, "resolution_gap", g, thresholds.rep_gap_veto, "above")
            _check_threshold(alerts, "WATCH", "rep", rid, "resolution_gap", g, thresholds.rep_gap_watch, "above")
    if not monthly_trends.empty and "trust_velocity" in monthly_trends.columns:
        for _, r in monthly_trends.iterrows():
            vel = r.get("trust_velocity", 0)
            if pd.notna(vel) and vel < -thresholds.drift_velocity_watch:
                alerts.append({"level":"WATCH","entity_type":"month","entity":str(r["_month"]),
                    "signal":"trust_velocity","value":round(vel,3),"threshold":-thresholds.drift_velocity_watch,
                    "message":f"Month {r['_month']}: trust dropped {abs(vel):.2f} pts (>{thresholds.drift_velocity_watch}/month)"})
    # Deduplicate: keep highest severity per entity+signal
    seen = {}
    for a in alerts:
        key = (a["entity_type"], a["entity"], a["signal"])
        if key not in seen or (a["level"] == "VETO" and seen[key]["level"] == "WATCH"):
            seen[key] = a
    deduped = sorted(seen.values(), key=lambda a: (0 if a["level"]=="VETO" else 1, a["entity_type"], a["entity"]))
    return {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_alerts": len(deduped), "veto_count": sum(1 for a in deduped if a["level"]=="VETO"),
            "watch_count": sum(1 for a in deduped if a["level"]=="WATCH"),
            "thresholds_used": asdict(thresholds), "alerts": deduped}

def write_threshold_alerts_txt(alert_summary, outpath):
    lines = ["GOVERNANCE THRESHOLD ALERTS", "="*60,
             f"Generated: {alert_summary['timestamp']}",
             f"Total alerts: {alert_summary['total_alerts']}",
             f"  VETO:  {alert_summary['veto_count']}", f"  WATCH: {alert_summary['watch_count']}", ""]
    if alert_summary["veto_count"]:
        lines.append("─── VETO ALERTS (optimization halted) ───")
        for a in alert_summary["alerts"]:
            if a["level"]=="VETO": lines.append(f"  [{a['entity_type'].upper()}] {a['message']}")
        lines.append("")
    if alert_summary["watch_count"]:
        lines.append("─── WATCH ALERTS (human review required) ───")
        for a in alert_summary["alerts"]:
            if a["level"]=="WATCH": lines.append(f"  [{a['entity_type'].upper()}] {a['message']}")
        lines.append("")
    if alert_summary["total_alerts"] == 0:
        lines.append("No threshold alerts triggered. All governance signals within bounds.\n")
    th = alert_summary["thresholds_used"]
    lines += ["THRESHOLD CONFIGURATION",
              f"  Scenario trust VETO/WATCH:  < {th['trust_score_veto']} / < {th['trust_score_watch']}",
              f"  Scenario gap VETO/WATCH:    > {th['resolution_gap_veto']:.0%} / > {th['resolution_gap_watch']:.0%}",
              f"  Bandaid rate VETO/WATCH:    > {th['bandaid_rate_veto']:.0%} / > {th['bandaid_rate_watch']:.0%}",
              f"  Rep trust VETO/WATCH:       < {th['rep_trust_veto']} / < {th['rep_trust_watch']}",
              f"  Rep gap VETO/WATCH:         > {th['rep_gap_veto']:.0%} / > {th['rep_gap_watch']:.0%}",
              f"  Drift velocity WATCH:       > {th['drift_velocity_watch']} pts/month"]
    outpath.write_text("\n".join(lines)+"\n", encoding="utf-8")

# ── Customer Churn Integration ──

def compute_churn_by_trust_decile(df):
    if "customer_is_churned" not in df.columns: return pd.DataFrame()
    out = df.copy()
    out["_churned"] = _safe_coerce(out, "customer_is_churned")
    out["trust_decile"] = pd.qcut(out["call_trust_score"], 10, labels=False, duplicates="drop") + 1
    spec = {"calls":("call_id","count"), "trust_score_min":("call_trust_score","min"),
            "trust_score_max":("call_trust_score","max"), "trust_score_avg":("call_trust_score","mean"),
            "churn_rate":("_churned","mean"),
            "proxy_resolution_rate":("resolution_flag", lambda s: _coerce_flag(s).mean()),
            "true_resolution_rate":("true_resolution", lambda s: _coerce_flag(s).mean())}
    if "repeat_contact_30d" in out.columns:
        spec["repeat_30d_rate"] = ("repeat_contact_30d", lambda s: _coerce_flag(s).mean())
    dec = out.groupby("trust_decile").agg(**spec).reset_index()
    dec["resolution_gap"] = (dec["proxy_resolution_rate"] - dec["true_resolution_rate"]).round(4)
    if len(dec) > 2:
        corr = dec["trust_score_avg"].corr(dec["churn_rate"])
        dec.attrs["trust_churn_correlation"] = round(corr, 4) if pd.notna(corr) else None
    return dec

# ── Governance JSON Export ──

def build_governance_report(integrity_summary, scenario_summary, rep_summary,
                            monthly_trends, alert_summary, churn_decile, df_scored):
    proxy_rate = _safe_coerce(df_scored, "resolution_flag").mean()
    true_rate = _safe_coerce(df_scored, "true_resolution").mean()
    trust = df_scored["call_trust_score"]
    report = {
        "meta": {"pipeline_version": "2.0.0",
                 "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "rows_analyzed": len(df_scored),
                 "months_loaded": integrity_summary.get("months_loaded", 0)},
        "integrity_gate": {
            "rows_total": integrity_summary.get("rows_total", 0),
            "rows_clean": integrity_summary.get("rows_clean", 0),
            "rows_quarantined": integrity_summary.get("rows_quarantined", 0),
            "quarantine_rate": round(integrity_summary.get("quarantine_rate", 0), 4),
            "proxy_true_divergence_rate": round(
                integrity_summary.get("soft_signal_rates", {}).get("proxy_true_divergence", 0), 4),
        },
        "overall_signals": {
            "proxy_resolution_rate": round(float(proxy_rate), 4),
            "true_resolution_rate": round(float(true_rate), 4),
            "resolution_inflation_pp": round(float(100 * (proxy_rate - true_rate)), 2),
            "trust_score_mean": round(float(trust.mean()), 2),
            "trust_score_median": round(float(trust.median()), 2),
            "trust_score_p10": round(float(trust.quantile(0.10)), 2),
            "trust_score_p90": round(float(trust.quantile(0.90)), 2),
            "repeat_contact_30d_rate": round(float(_safe_coerce(df_scored, "repeat_contact_30d").mean()), 4),
            "escalation_rate": round(float(_safe_coerce(df_scored, "escalation_flag").mean()), 4),
            "bandaid_rate": round(float(
                (df_scored["credit_type"].astype(str).str.strip() == "bandaid").mean()
                if "credit_type" in df_scored.columns else 0), 4),
        },
        "scenario_health": [], "rep_health": {"total_reps": len(rep_summary),
            "lowest_trust_reps": [], "highest_trust_reps": []},
        "monthly_trends": [], "alerts": alert_summary, "churn_integration": {},
    }
    if not scenario_summary.empty:
        for _, row in scenario_summary.iterrows():
            t = row.get("trust_score_avg", 100)
            report["scenario_health"].append({
                "scenario": row["scenario"], "calls": int(row["calls"]),
                "trust_score": round(t, 2), "resolution_gap": round(row.get("resolution_gap", 0), 4),
                "bandaid_rate": round(row.get("bandaid_rate", 0), 4),
                "status": "VETO" if t < 50 else "WATCH" if t < 65 else "OK",
            })
    if not rep_summary.empty:
        for _, r in rep_summary.head(5).iterrows():
            report["rep_health"]["lowest_trust_reps"].append(
                {"rep_id": r["rep_id"], "calls": int(r["calls"]),
                 "trust_score": round(r["trust_score_avg"], 2),
                 "resolution_gap": round(r.get("resolution_gap", 0), 4)})
        for _, r in rep_summary.tail(5).iloc[::-1].iterrows():
            report["rep_health"]["highest_trust_reps"].append(
                {"rep_id": r["rep_id"], "calls": int(r["calls"]),
                 "trust_score": round(r["trust_score_avg"], 2),
                 "resolution_gap": round(r.get("resolution_gap", 0), 4)})
    if not monthly_trends.empty:
        for _, row in monthly_trends.iterrows():
            report["monthly_trends"].append({
                "month": str(row["_month"]), "calls": int(row["calls"]),
                "trust_score": round(row.get("trust_score_avg", 0), 2),
                "resolution_gap": round(row.get("resolution_gap", 0), 4),
                "bandaid_rate": round(row.get("bandaid_rate", 0), 4),
                "trust_velocity": round(row.get("trust_velocity", 0), 3) if pd.notna(row.get("trust_velocity")) else None,
            })
    if not churn_decile.empty:
        corr = churn_decile.attrs.get("trust_churn_correlation", None)
        report["churn_integration"] = {
            "trust_churn_correlation": corr,
            "interpretation": (
                "Trust score is predictive of churn (negative correlation)." if corr is not None and corr < -0.3 else
                "Trust score shows weak/no relationship with churn at call level." if corr is not None else
                "Churn data not available."),
            "deciles": [{"decile": int(r["trust_decile"]),
                         "trust_range": f"{r['trust_score_min']:.1f}-{r['trust_score_max']:.1f}",
                         "calls": int(r["calls"]), "churn_rate": round(r.get("churn_rate", 0), 4),
                         "resolution_gap": round(r.get("resolution_gap", 0), 4)}
                        for _, r in churn_decile.iterrows()],
        }
    return report

# ── Charts & Evidence ──

CHART_DPI = 180
CHART_BG = "#1a1a2e"
CHART_FG = "#e0e0e0"
CHART_ACCENT = "#00d4aa"
CHART_WARN = "#ff6b6b"

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": CHART_BG, "axes.facecolor": "#16213e",
        "axes.edgecolor": CHART_FG, "axes.labelcolor": CHART_FG,
        "text.color": CHART_FG, "xtick.color": CHART_FG, "ytick.color": CHART_FG,
        "grid.color": "#2a2a4a", "grid.alpha": 0.3, "font.size": 10,
    })

def chart_trust_distribution(df, fig_dir):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    scores = df["call_trust_score"].dropna()
    ax.hist(scores, bins=40, color=CHART_ACCENT, alpha=0.7, edgecolor="none")
    ax.axvline(scores.mean(), color=CHART_WARN, ls="--", lw=2, label=f"Mean: {scores.mean():.1f}")
    ax.axvline(scores.median(), color="#ffd93d", ls=":", lw=2, label=f"Median: {scores.median():.1f}")
    ax.set_xlabel("Call Trust Score (0-100)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Call Trust Scores")
    ax.legend(facecolor="#16213e", edgecolor=CHART_FG); ax.grid(True, axis="y")
    plt.tight_layout(); plt.savefig(fig_dir/"trust_score_distribution.png", dpi=CHART_DPI); plt.close()

def chart_proxy_truth_gap(df, fig_dir):
    if "scenario" not in df.columns: return
    _apply_dark_style()
    scen = df.groupby("scenario").agg(
        proxy=("resolution_flag", lambda s: _coerce_flag(s).mean()),
        true=("true_resolution", lambda s: _coerce_flag(s).mean()),
    ).reset_index().sort_values("proxy", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.5*len(scen)+1)))
    y = np.arange(len(scen)); h = 0.35
    ax.barh(y-h/2, scen["proxy"], h, label="Proxy Resolution", color=CHART_WARN, alpha=0.85)
    ax.barh(y+h/2, scen["true"], h, label="True Resolution", color=CHART_ACCENT, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(scen["scenario"]); ax.set_xlabel("Rate")
    ax.set_title("Proxy vs True Resolution by Scenario — The Gap is Goodhart's Law")
    ax.legend(facecolor="#16213e", edgecolor=CHART_FG); ax.grid(True, axis="x")
    plt.tight_layout(); plt.savefig(fig_dir/"proxy_vs_true_by_scenario.png", dpi=CHART_DPI); plt.close()

def chart_rep_landscape(rep_summary, fig_dir):
    if rep_summary.empty: return
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(rep_summary["calls"], rep_summary["trust_score_avg"],
                    c=rep_summary["resolution_gap"], cmap="RdYlGn_r", s=40, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Resolution Gap (proxy - true)")
    ax.set_xlabel("Calls Handled"); ax.set_ylabel("Avg Trust Score")
    ax.set_title("Rep Landscape: Trust Score vs Volume (color = resolution inflation)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir/"rep_trust_landscape.png", dpi=CHART_DPI); plt.close()

def chart_scenario_drift_heatmap(scenario_summary, fig_dir):
    if scenario_summary.empty: return
    _apply_dark_style()
    cols = ["proxy_true_gap_rate","bandaid_rate","detection_density_avg","outcome_risk_avg","credit_applied_rate"]
    avail = [c for c in cols if c in scenario_summary.columns]
    if len(avail) < 2: return
    data = scenario_summary.set_index("scenario")[avail].astype(float)
    fig, ax = plt.subplots(figsize=(max(8, len(avail)*1.5), max(5, len(data)*0.6)))
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(avail))); ax.set_xticklabels([c.replace("_","\n") for c in avail], fontsize=8)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=9)
    for i in range(len(data)):
        for j in range(len(avail)):
            v = data.values[i,j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                       color="black" if v > 0.5 else CHART_FG)
    ax.set_title("Scenario × Signal Heatmap — Where Does Drift Concentrate?")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(fig_dir/"scenario_drift_heatmap.png", dpi=CHART_DPI); plt.close()

def chart_credit_analysis(df, fig_dir):
    if "credit_type" not in df.columns or "scenario" not in df.columns: return
    _apply_dark_style()
    credited = df[_safe_coerce(df, "credit_applied") == 1].copy()
    if len(credited) < 10: return
    ct = pd.crosstab(credited["scenario"], credited["credit_type"], normalize="index")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.5*len(ct)+1)))
    ct.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", edgecolor="none")
    ax.set_xlabel("Proportion of Credits")
    ax.set_title("Credit Type Mix by Scenario — Bandaid = Unauthorized Suppression")
    ax.legend(title="Credit Type", bbox_to_anchor=(1.02,1), loc="upper left",
              facecolor="#16213e", edgecolor=CHART_FG)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir/"credit_type_by_scenario.png", dpi=CHART_DPI); plt.close()

def chart_corr_heatmap(rep_summary, fig_dir):
    if rep_summary.empty: return
    _apply_dark_style()
    ncols = [c for c in rep_summary.columns if c != "rep_id"
             and pd.api.types.is_numeric_dtype(rep_summary[c]) and rep_summary[c].notna().sum() > 5]
    if len(ncols) < 3: return
    corr = rep_summary[ncols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(ncols)*0.8), max(6, len(ncols)*0.7)))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ncols))); ax.set_xticklabels([c.replace("_","\n") for c in ncols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ncols))); ax.set_yticklabels(ncols, fontsize=7)
    for i in range(len(ncols)):
        for j in range(len(ncols)):
            v = corr.values[i,j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                       color="black" if abs(v)>0.5 else CHART_FG)
    ax.set_title("Rep-Level Signal Correlations"); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(fig_dir/"rep_signal_correlations.png", dpi=CHART_DPI); plt.close()

def chart_monthly_trust_trend(monthly, fig_dir):
    if monthly.empty: return
    _apply_dark_style()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    months = monthly["_month"].astype(str); x = np.arange(len(months))
    ax1.plot(x, monthly["trust_score_avg"], "o-", color=CHART_ACCENT, lw=2.5, ms=8, label="Trust Score (avg)", zorder=3)
    if "trust_score_p10" in monthly.columns:
        ax1.fill_between(x, monthly["trust_score_p10"], monthly["trust_score_avg"], alpha=0.15, color=CHART_ACCENT)
    ax1.set_ylabel("Trust Score", color=CHART_ACCENT); ax1.set_ylim(0, 100); ax1.grid(True, axis="y", alpha=0.2)
    if "trust_velocity" in monthly.columns:
        ax2 = ax1.twinx()
        vel = monthly["trust_velocity"].fillna(0)
        colors = [CHART_WARN if v < 0 else CHART_ACCENT for v in vel]
        ax2.bar(x, vel, alpha=0.4, color=colors, width=0.5, label="Drift Velocity")
        ax2.set_ylabel("Drift Velocity (pts/month)", color=CHART_FG)
        ax2.axhline(0, color=CHART_FG, lw=0.5, alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Month"); ax1.set_title("Monthly Trust Score Trend with Drift Velocity")
    ax1.legend(loc="upper left", facecolor="#16213e", edgecolor=CHART_FG)
    plt.tight_layout(); plt.savefig(fig_dir/"monthly_trust_trend.png", dpi=CHART_DPI); plt.close()

def chart_monthly_gap_trend(monthly, fig_dir):
    if monthly.empty: return
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    months = monthly["_month"].astype(str); x = np.arange(len(months))
    ax.plot(x, monthly["resolution_gap"]*100, "s-", color=CHART_WARN, lw=2, ms=7, label="Resolution Gap (pp)")
    if "bandaid_rate" in monthly.columns:
        ax.plot(x, monthly["bandaid_rate"]*100, "D-", color="#ffd93d", lw=2, ms=6, label="Bandaid Rate (%)")
    if "repeat_30d_rate" in monthly.columns:
        ax.plot(x, monthly["repeat_30d_rate"]*100, "^-", color="#a78bfa", lw=2, ms=6, label="Repeat Contact 30d (%)")
    if "churn_rate" in monthly.columns and monthly["churn_rate"].notna().any():
        ax.plot(x, monthly["churn_rate"]*100, "v-", color="#f97316", lw=2, ms=6, label="Churn Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Month"); ax.set_ylabel("Rate (%)")
    ax.set_title("Monthly Governance Signals — Gap, Bandaid, Repeat Contact, Churn")
    ax.legend(loc="upper right", facecolor="#16213e", edgecolor=CHART_FG, fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(fig_dir/"monthly_gap_trend.png", dpi=CHART_DPI); plt.close()

def chart_churn_by_trust_decile(decile_df, fig_dir):
    if decile_df.empty: return
    _apply_dark_style()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = decile_df["trust_decile"].values
    ax1.bar(x, decile_df["churn_rate"]*100, 0.5, color=CHART_WARN, alpha=0.85, label="Churn Rate (%)", zorder=2)
    ax1.set_ylabel("Churn Rate (%)", color=CHART_WARN); ax1.set_xlabel("Trust Score Decile (1=lowest, 10=highest)")
    ax2 = ax1.twinx()
    ax2.plot(x, decile_df["resolution_gap"]*100, "o-", color=CHART_ACCENT, lw=2.5, ms=8, label="Resolution Gap (pp)", zorder=3)
    ax2.set_ylabel("Resolution Gap (pp)", color=CHART_ACCENT)
    labels = [f"D{int(r['trust_decile'])}\n{r['trust_score_min']:.0f}-{r['trust_score_max']:.0f}" for _, r in decile_df.iterrows()]
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    corr = decile_df.attrs.get("trust_churn_correlation", None)
    if corr is not None:
        ax1.annotate(f"Trust-Churn r = {corr:.3f}", xy=(0.02, 0.95), xycoords="axes fraction",
                     fontsize=11, color=CHART_FG, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e", edgecolor=CHART_FG, alpha=0.8))
    ax1.set_title("Churn Rate by Trust Score Decile — Does Trust Predict Retention?")
    ax1.grid(True, axis="y", alpha=0.2)
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc="upper right", facecolor="#16213e", edgecolor=CHART_FG)
    plt.tight_layout(); plt.savefig(fig_dir/"churn_by_trust_decile.png", dpi=CHART_DPI); plt.close()

# ── Summary Report ──

def write_summary_report(df, rep_summary, scenario_summary, monthly_trends,
                         alert_summary, churn_decile, integrity_summary, outpath,
                         min_calls_for_rank=20):
    n = len(df)
    if n == 0: outpath.write_text("No rows found.\n", encoding="utf-8"); return
    def pct(x): return f"{100*x:.1f}%"
    lines = ["TRUST SIGNAL HEALTH PIPELINE — SUMMARY REPORT (v2)", "="*60,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"Rows analyzed: {n:,}",
             f"Months loaded: {integrity_summary.get('months_loaded','?')}", ""]
    # Integrity gate
    lines += ["INTEGRITY GATE",
              f"  Total rows:       {integrity_summary.get('rows_total',n):,}",
              f"  Clean rows:       {integrity_summary.get('rows_clean',n):,}",
              f"  Quarantined:      {integrity_summary.get('rows_quarantined',0):,} ({pct(integrity_summary.get('quarantine_rate',0))})"]
    soft = integrity_summary.get("soft_signal_rates", {})
    if soft.get("proxy_true_divergence", 0) > 0:
        lines.append(f"  Proxy-true divergence rate: {pct(soft['proxy_true_divergence'])} (soft signal, not quarantined)")
    lines.append("")
    # Overall signals
    pr = _safe_coerce(df, "resolution_flag").mean(); tr = _safe_coerce(df, "true_resolution").mean()
    lines += ["OVERALL SIGNAL RATES",
              f"  Proxy resolution rate:      {pct(pr)}",
              f"  True resolution rate:       {pct(tr)}",
              f"  Resolution inflation:       {pct(pr-tr)} ← this is the Goodhart gap",
              f"  Repeat contact (30d) rate:  {pct(_safe_coerce(df,'repeat_contact_30d').mean())}",
              f"  Escalation rate:            {pct(_safe_coerce(df,'escalation_flag').mean())}",
              f"  Credit applied rate:        {pct(_safe_coerce(df,'credit_applied').mean())}"]
    if "credit_type" in df.columns:
        lines.append(f"  Bandaid credit rate:        {pct((df['credit_type'].astype(str).str.strip()=='bandaid').mean())} ← unauthorized suppression signal")
    lines.append("")
    # Trust score
    trust = df["call_trust_score"]
    lines += ["TRUST SCORE SUMMARY",
              f"  Mean:   {trust.mean():.1f}", f"  Median: {trust.median():.1f}",
              f"  Std:    {trust.std():.1f}", f"  P10:    {trust.quantile(0.10):.1f}",
              f"  P90:    {trust.quantile(0.90):.1f}", ""]
    # Scenario health
    if not scenario_summary.empty:
        lines.append("SCENARIO HEALTH (sorted by trust score, lowest first)")
        lines.append(f"  {'Scenario':<28} {'Calls':>7} {'Trust':>7} {'Gap':>7} {'Bandaid':>8}")
        lines.append("  " + "-"*60)
        for _, r in scenario_summary.iterrows():
            lines.append(f"  {r['scenario']:<28} {int(r['calls']):>7,} {r['trust_score_avg']:>7.1f} "
                         f"{pct(r.get('resolution_gap',0)):>7} {pct(r.get('bandaid_rate',0)):>8}")
        lines.append("")
    # Monthly trends
    if not monthly_trends.empty:
        lines.append("MONTHLY TREND SUMMARY")
        lines.append(f"  {'Month':<12} {'Calls':>7} {'Trust':>7} {'Gap':>7} {'Bandaid':>8} {'Velocity':>10}")
        lines.append("  " + "-"*60)
        for _, r in monthly_trends.iterrows():
            vel = r.get("trust_velocity", 0)
            vs = f"{vel:>+.2f}" if pd.notna(vel) else "    —"
            lines.append(f"  {str(r['_month']):<12} {int(r['calls']):>7,} {r['trust_score_avg']:>7.1f} "
                         f"{pct(r.get('resolution_gap',0)):>7} {pct(r.get('bandaid_rate',0)):>8} {vs:>10}")
        lines.append("")
    # Rep rankings
    if not rep_summary.empty:
        qual = rep_summary[rep_summary["calls"] >= min_calls_for_rank]
        lines += [f"REP RANKINGS (>= {min_calls_for_rank} calls)",
                  f"  Total reps: {len(rep_summary)}  |  Qualified: {len(qual)}", ""]
        if not qual.empty:
            lines.append("  LOWEST TRUST (most concerning):")
            for _, r in qual.head(5).iterrows():
                lines.append(f"    {r['rep_id']} | calls={int(r['calls'])} | trust={r['trust_score_avg']:.1f} "
                             f"| gap={pct(r.get('resolution_gap',0))} | credit_risk={r.get('credit_risk_avg',0):.2f}")
            lines.append("")
            lines.append("  HIGHEST TRUST (healthiest):")
            for _, r in qual.tail(5).iloc[::-1].iterrows():
                lines.append(f"    {r['rep_id']} | calls={int(r['calls'])} | trust={r['trust_score_avg']:.1f} "
                             f"| gap={pct(r.get('resolution_gap',0))} | credit_risk={r.get('credit_risk_avg',0):.2f}")
        lines.append("")
    # Churn integration
    if not churn_decile.empty:
        corr = churn_decile.attrs.get("trust_churn_correlation", None)
        lines.append("CHURN INTEGRATION — Trust Score Predictive Power")
        if corr is not None:
            lines.append(f"  Trust-Churn Correlation: r = {corr:.4f}")
            if corr < -0.3: lines.append("  ► Trust score IS predictive of churn. Lower trust = higher churn.")
            elif corr < -0.1: lines.append("  ► Weak negative relationship. Trust has marginal predictive value.")
            else:
                lines += ["  ► Trust score shows minimal call-level churn prediction.",
                          "    Consistent with finding that churn signal is structurally diffuse",
                          "    and requires account-level longitudinal signals."]
        lines.append("")
        lines.append(f"  {'Decile':<8} {'Trust Range':<14} {'Calls':>7} {'Churn':>7} {'Gap':>7}")
        lines.append("  " + "-"*50)
        for _, r in churn_decile.iterrows():
            lines.append(f"  D{int(r['trust_decile']):<7} {r['trust_score_min']:.0f}-{r['trust_score_max']:.0f}{'':<8} "
                         f"{int(r['calls']):>7,} {pct(r.get('churn_rate',0)):>7} {pct(r.get('resolution_gap',0)):>7}")
        lines.append("")
    # Alert summary
    lines += ["THRESHOLD ALERTS",
              f"  Total: {alert_summary.get('total_alerts',0)}  "
              f"(VETO: {alert_summary.get('veto_count',0)}, WATCH: {alert_summary.get('watch_count',0)})"]
    if alert_summary.get("veto_count", 0) > 0:
        lines.append("  ⚠ VETO conditions present — see threshold_alerts.txt for details")
    lines.append("")
    # Interpretation guide
    lines += ["INTERPRETATION GUIDE",
              "  - Resolution inflation = proxy - true rate. Positive = system looks better than it is.",
              "  - Bandaid credits are unauthorized credits to suppress repeat contacts.",
              "  - Detection signal density = fraction of fraud/gaming indicators firing per call.",
              "  - Rep drift score = gaming propensity + burnout composite.",
              "  - Call trust score (0-100): 100 = fully trustworthy, 0 = total divergence.",
              "  - Drift velocity = month-over-month trust score change. Negative = degrading.",
              "  - VETO = halt AI optimization; re-audit labels. WATCH = human review required."]
    outpath.write_text("\n".join(lines)+"\n", encoding="utf-8")

# ── Pipeline Command ──

def cmd_run(args):
    repo_root = find_repo_root(Path.cwd())
    data_dir = repo_root / "data"
    out_dirs = ensure_output_dirs(repo_root)
    fig_dir = out_dirs["figures"]

    print("\n" + "="*60)
    print("TRUST SIGNAL HEALTH PIPELINE v2.0")
    print("="*60)
    print(f"Repo root: {repo_root}")
    print(f"Data dir:  {data_dir}")
    print(f"Output:    {out_dirs['base']}")
    print(f"  data/    {out_dirs['data']}")
    print(f"  figures/ {out_dirs['figures']}")
    print(f"  reports/ {out_dirs['reports']}")
    print("")

    # Load
    print("Loading monthly files...")
    df_raw = load_monthly_files(data_dir)

    # Integrity gate
    print("Running integrity gate...")
    cfg = IntegrityConfig()
    gate = run_integrity_gate(df_raw, out_dirs, cfg=cfg, dupe_policy=args.dupe_policy)
    s = gate["summary"]
    print(f"  Clean: {s['rows_clean']:,} | Quarantined: {s['rows_quarantined']:,} ({s['quarantine_rate']:.2%})\n")

    # Trust signal scoring
    print("Computing trust signals...")
    df_clean = gate["clean_df"].copy()
    if "_source_file" in df_clean.columns: df_clean = df_clean.drop(columns=["_source_file"])
    df_scored = compute_trust_signals(df_clean)
    df_scored.to_csv(out_dirs["data"] / "calls_scored.csv", index=False)
    print(f"  Scored {len(df_scored):,} rows | Mean trust: {df_scored['call_trust_score'].mean():.1f}\n")

    # Aggregation
    print("Building summaries...")
    rep_summary = summarize_by_rep(df_scored, min_calls=args.min_calls_for_rank)
    scenario_summary = summarize_by_scenario(df_scored)
    customer_summary = summarize_by_customer(df_scored)
    rep_summary.to_csv(out_dirs["data"] / "rep_summary.csv", index=False)
    scenario_summary.to_csv(out_dirs["data"] / "scenario_summary.csv", index=False)
    customer_summary.to_csv(out_dirs["data"] / "customer_summary.csv", index=False)
    print(f"  Reps: {len(rep_summary)} | Scenarios: {len(scenario_summary)} | Customers: {len(customer_summary)}\n")

    # Monthly trends
    print("Computing monthly trends...")
    monthly_trends = compute_monthly_trends(df_scored)
    if not monthly_trends.empty:
        monthly_trends.to_csv(out_dirs["data"] / "monthly_trends.csv", index=False)
        print(f"  {len(monthly_trends)} months tracked")
        if "trust_velocity" in monthly_trends.columns:
            vel = monthly_trends["trust_velocity"].dropna()
            if len(vel) > 0: print(f"  Drift velocity range: {vel.min():+.2f} to {vel.max():+.2f} pts/month")
    print("")

    # Churn integration
    print("Computing churn by trust decile...")
    churn_decile = compute_churn_by_trust_decile(df_scored)
    if not churn_decile.empty:
        churn_decile.to_csv(out_dirs["data"] / "churn_by_trust_decile.csv", index=False)
        corr = churn_decile.attrs.get("trust_churn_correlation", None)
        print(f"  Trust-Churn correlation: r = {corr:.4f}" if corr is not None else "  Trust-Churn correlation: N/A")
    else:
        print("  Churn data not available — skipping")
    print("")

    # Threshold alerts
    print("Evaluating governance thresholds...")
    thresholds = ThresholdConfig()
    alert_summary = run_threshold_alerts(scenario_summary, rep_summary, monthly_trends, thresholds)
    save_json(out_dirs["reports"] / "threshold_alerts.json", alert_summary)
    write_threshold_alerts_txt(alert_summary, out_dirs["reports"] / "threshold_alerts.txt")
    print(f"  Alerts: {alert_summary['total_alerts']} (VETO: {alert_summary['veto_count']}, WATCH: {alert_summary['watch_count']})\n")

    # Charts (9 total)
    print("Generating charts...")
    chart_trust_distribution(df_scored, fig_dir)
    chart_proxy_truth_gap(df_scored, fig_dir)
    chart_rep_landscape(rep_summary, fig_dir)
    chart_scenario_drift_heatmap(scenario_summary, fig_dir)
    chart_credit_analysis(df_scored, fig_dir)
    chart_corr_heatmap(rep_summary, fig_dir)
    chart_monthly_trust_trend(monthly_trends, fig_dir)
    chart_monthly_gap_trend(monthly_trends, fig_dir)
    chart_churn_by_trust_decile(churn_decile, fig_dir)
    print("  9 charts written to output/figures/\n")

    # Reports
    print("Writing reports...")
    write_summary_report(df_scored, rep_summary=rep_summary, scenario_summary=scenario_summary,
                         monthly_trends=monthly_trends, alert_summary=alert_summary,
                         churn_decile=churn_decile, integrity_summary=s,
                         outpath=out_dirs["reports"] / "summary_report.txt",
                         min_calls_for_rank=args.min_calls_for_rank)
    gov_report = build_governance_report(
        integrity_summary=s, scenario_summary=scenario_summary, rep_summary=rep_summary,
        monthly_trends=monthly_trends, alert_summary=alert_summary,
        churn_decile=churn_decile, df_scored=df_scored)
    save_json(out_dirs["reports"] / "governance_report.json", gov_report)
    print("  Reports written to output/reports/\n")

    # Console summary
    print("="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Input:  {s.get('months_loaded','?')} monthly files ({s['rows_total']:,} rows)")
    print(f"Clean:  {s['rows_clean']:,} rows → scored → summarized → trended → alerted")
    print("")
    print("Output structure:")
    print("  output/data/")
    print("    calls_clean.csv              calls_quarantine.csv")
    print("    integrity_flags.csv          calls_scored.csv")
    print("    rep_summary.csv              scenario_summary.csv")
    print("    customer_summary.csv         monthly_trends.csv")
    print("    churn_by_trust_decile.csv")
    print("  output/figures/")
    print("    trust_score_distribution.png proxy_vs_true_by_scenario.png")
    print("    rep_trust_landscape.png      scenario_drift_heatmap.png")
    print("    credit_type_by_scenario.png  rep_signal_correlations.png")
    print("    monthly_trust_trend.png      monthly_gap_trend.png")
    print("    churn_by_trust_decile.png")
    print("  output/reports/")
    print("    integrity_summary.json       summary_report.txt")
    print("    governance_report.json       threshold_alerts.json")
    print("    threshold_alerts.txt")
    print("")
    proxy = _safe_coerce(df_scored, "resolution_flag").mean()
    true = _safe_coerce(df_scored, "true_resolution").mean()
    print(f"Mean trust score: {df_scored['call_trust_score'].mean():.1f} / 100")
    print(f"Resolution inflation: {100*(proxy-true):.1f}pp (proxy {100*proxy:.1f}% vs true {100*true:.1f}%)")
    print(f"Governance alerts: {alert_summary['total_alerts']} (VETO: {alert_summary['veto_count']}, WATCH: {alert_summary['watch_count']})")
    print("Done.")
    return 0

# ── CLI ──

def build_parser():
    p = argparse.ArgumentParser(description="NovaWireless Trust Signal Health Pipeline v2.0")
    p.add_argument("--dupe_policy", default="quarantine_all",
                   choices=["quarantine_all","quarantine_extras_keep_latest","quarantine_extras_keep_first"])
    p.add_argument("--min_calls_for_rank", type=int, default=20)
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return cmd_run(args)

if __name__ == "__main__":
    raise SystemExit(main())
