"""Build the self-contained Kaggle notebook (model ladder version).

The notebook must run top-to-bottom on a fresh Kaggle kernel, so this builder
inlines every ``src`` module into its own cell (cross-module imports stripped).

    python build_notebook.py                 # full run notebook
    python build_notebook.py --smoke 60000   # smoke notebook on a sample of cases
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODULES = [
    "hc_config.py", "hc_metrics.py", "hc_data.py", "hc_features.py",
    "hc_model.py", "hc_stability.py", "hc_pipeline.py", "hc_ladder.py",
]


def strip_project_imports(source: str) -> str:
    source = re.sub(r"(?m)^[ \t]*from hc_[a-z_]+ import \([^)]*\)\n", "", source)
    source = re.sub(r"(?m)^[ \t]*from hc_[a-z_]+ import [^\n(]*\n", "", source)
    source = re.sub(r"(?m)^[ \t]*import hc_[a-z_]*\n", "", source)
    return source


def module_source(name: str) -> str:
    src = strip_project_imports((SRC / name).read_text(encoding="utf-8"))
    return src + f'\nprint("module {name} loaded")\n'


INTRO = """# Home Credit: out-of-time default model ladder (LR, WoE scorecard, random forest, XGBoost, LightGBM)

**Question.** Can gradient boosting beat a logistic regression and a regulatory-style WoE scorecard on applications
that arrive *after* the model was built, and does the answer survive a bootstrap, a recalibration and a simple
credit-policy cut-off?

**Data.** Kaggle competition `home-credit-credit-risk-model-stability` (real Home Credit applications, multi-table
parquet). One row per `case_id`; prediction time = `date_decision`; target = default flag observed later.

## Design in five lines

1. **Time split by `WEEK_NUM`** (no random split): ~70% / 15% / 15% of consecutive weeks = TRAIN / VALID / OOT. No `case_id` appears twice.
2. **Point-in-time features.** Every history record dated after `date_decision` is dropped before aggregation (mean/max/min/std, counts, recency).
3. **Fit on TRAIN rows only**: column screening, imputation, scaling, WoE binning, encoders. VALID is used for tuning, calibration and the cut-off; OOT is scored only after the champion, calibrator and cut-off are frozen and written to `artifacts/champion.json`.
4. **Same features, same split for every model**: dummy, L2 logistic regression, WoE scorecard (monotone bins, wrong-sign variables dropped until all signs are consistent), random forest, XGBoost (main advanced model, plus a monotone-constrained variant), LightGBM (check).
5. **Metrics on VALID and OOT for every model**: AUC, Gini, KS, PR-AUC, Brier, weekly Gini and the competition stability metric
   `mean(weekly Gini) + 88*min(0, slope) - 0.5*std(residuals)`.

Credit-policy numbers are observed OOT outcomes (approval rate, bad rate, swap sets, risk grades). The expected-loss table is illustrative: LGD is an assumption (0.45, Basel foundation-IRB senior unsecured) and EAD is the application credit amount.
All numbers written to `artifacts/` are the only source for the README.
"""

INSTALL = r"""import importlib, subprocess, sys

def _ensure(pkg, spec=None, module=None):
    module = module or pkg
    try:
        importlib.import_module(module)
        return "present"
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec or pkg], check=True)
        importlib.import_module(module)
        return "installed"

print("optbinning:", _ensure("optbinning", "optbinning==1.0.0"))
print("polars:", _ensure("polars"))
print("statsmodels:", _ensure("statsmodels"))
"""

IMPORTS = r"""import json, math, os, platform, sys, time, warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

warnings.simplefilter("ignore")
SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 90)

ART = Path("artifacts")
(ART / "figures").mkdir(parents=True, exist_ok=True)
(ART / "models").mkdir(parents=True, exist_ok=True)
T_START = time.time()
print("cwd:", Path.cwd(), "| cpu count:", os.cpu_count())
"""

DISCOVERY = r"""if Path("/kaggle/input").exists():
    print("mounted inputs:", sorted(p.name for p in Path("/kaggle/input").iterdir())[:10])
root = data_root()
print("data root:", root)
train_files = sorted(p.name for p in (root / "train").glob("*.parquet"))
print(f"{len(train_files)} train parquet files, e.g. {train_files[:4]}")

# plain-English variable descriptions shipped with the competition
defs = {}
cands = [root.parent / "feature_definitions.csv", root / "feature_definitions.csv"]
if Path("/kaggle/input").exists():
    cands += sorted(Path("/kaggle/input").rglob("feature_definitions.csv"))
for c in cands:
    if Path(c).exists():
        fd = pd.read_csv(c)
        defs = dict(zip(fd["Variable"], fd["Description"]))
        print("feature definitions:", c, len(defs))
        break

base = read_base(SAMPLE_CASES)
print("base:", base.shape, "| weeks", base[WEEK_COL].min(), "-", base[WEEK_COL].max(),
      "| default rate", round(float(base[TARGET].mean()), 5))

prov = [file_fingerprint(root / "train" / f) for f in train_files]
write_json(ART / "data_provenance.json", {"root": str(root), "sample_cases": SAMPLE_CASES, "files": prov})
print("provenance (sha256) written for", len(prov), "files")
"""

PROFILE = r"""weeks = sorted(base[WEEK_COL].unique().to_list())
bounds = split_bounds(weeks, SPLIT_FRACTIONS)
split_tbl = (
    base.with_columns(
        pl.when(pl.col(WEEK_COL) <= bounds["train_weeks"][-1]).then(pl.lit("train"))
        .when(pl.col(WEEK_COL) <= bounds["valid_weeks"][-1]).then(pl.lit("valid"))
        .otherwise(pl.lit("oot")).alias("split")
    )
    .group_by("split")
    .agg(pl.len().alias("n"), pl.col(TARGET).mean().alias("default_rate"),
         pl.col(WEEK_COL).min().alias("week_min"), pl.col(WEEK_COL).max().alias("week_max"),
         pl.col(WEEK_COL).n_unique().alias("n_weeks"),
         pl.col(DATE_COL).min().alias("date_min"), pl.col(DATE_COL).max().alias("date_max"))
    .sort("week_min")
)
split_tbl = split_tbl.with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
print(split_tbl)
write_json(ART / "split.json", {"bounds": bounds, "table": split_tbl.to_dicts(),
                                "fractions_requested": list(SPLIT_FRACTIONS), "sample_cases": SAMPLE_CASES})
"""

FEATURES = r"""ds = build_dataset(sample_cases=SAMPLE_CASES, verbose=True)
print(f"\nbuilt in {ds['elapsed_s']:.0f}s: {ds['X'].shape[0]} rows x {ds['X'].shape[1]} columns, {len(ds['cat_cols'])} categorical")
prof = split_profile(ds)
print(pd.DataFrame(prof).T.to_string())
pit = pd.DataFrame(ds["audit"]["pit"])
if not pit.empty:
    cols = [c for c in ["table", "depth", "rows_in", "rows_kept", "rows_dropped", "drop_share"] if c in pit.columns]
    print("\npoint-in-time filter (history rows dropped because dated after date_decision):")
    print(pit[cols].to_string(index=False))
write_json(ART / "pit_audit.json", {"tables": ds["audit"]["pit"], "missing_tables": ds["audit"]["tables_missing"]})
write_json(ART / "feature_screening.json", {"tables_used": ds["audit"]["tables_used"],
           "n_features_built": int(ds["X"].shape[1]), "n_categorical": len(ds["cat_cols"]),
           "screening": ds["audit"]["screening"], "split_profile": prof})
tr_m, va_m, oo_m = ds["train_mask"], ds["valid_mask"], ds["oot_mask"]
y_all, week_all = ds["y"], ds["week"]
"""

GATE = r"""screen = feature_screen(ds)
iv_tbl = screen["iv_table"]
print(f"candidates after the univariate screen: {screen['n_candidates']}")
print(f"selected: trees={len(screen['tree_features'])} linear={len(screen['linear_numeric'])} woe={len(screen['woe_features'])}")
print("quarantined (single-feature AUC >", UNIVARIATE_AUC_SCREEN, "):", screen["quarantined"])
print(iv_tbl.head(12).to_string(index=False))
iv_tbl.to_csv(ART / "iv_table.csv", index=False)
print(f"dropped by the TRAIN recent-window coverage rule (< {MIN_RECENT_COVERAGE:.0%} non-null in the last {RECENT_TRAIN_WEEKS} train weeks): {len(screen['low_recent_coverage'])}")

# ---- coverage gate (v2): no final model feature may be empty in VALID or OOT ----------------
cov = feature_coverage(ds, {"tree": screen["tree_features"], "linear": screen["linear_numeric"], "woe": screen["woe_features"]})
cov["low_recent_coverage_dropped"] = screen["low_recent_coverage"]
cov["recent_train_weeks"] = screen["recent_train_weeks"]
write_json(ART / "feature_coverage.json", cov)
print(f"final model features: {cov['n_final_features']} | null in VALID: {cov['n_null_in_valid']} | null in OOT: {cov['n_null_in_oot']}")
print(pd.DataFrame(cov["by_source"]).T.round(3).to_string())
cov_gate_row = coverage_gate(cov, raise_on_fail=True)
screen["calendar_proxy"].to_csv(ART / "calendar_proxy_screen.csv", index=False)
print(f"calendar-proxy screen (TRAIN, |Spearman rho with WEEK_NUM| >= {CALENDAR_PROXY_RHO}): {int(screen['calendar_proxy']['dropped'].sum())} of {len(screen['calendar_proxy'])} relative-date features dropped")
proxy_gate_row = calendar_proxy_gate(screen["calendar_proxy"], sorted(set(screen["tree_features"]) | set(screen["linear_numeric"]) | set(screen["woe_features"])), raise_on_fail=True)
cal_gate_row = calendar_gate(sorted(set(screen["tree_features"]) | set(screen["linear_numeric"]) | set(screen["woe_features"])), raise_on_fail=True)
print("calendar gate:", cal_gate_row["status"], cal_gate_row["detail"]["n_relative_features"], "relative-time features among the final features")

design, onehot_cats = build_linear_design(ds, screen["linear_numeric"])
print("\nlinear design:", design.shape)

sc_raw = build_scorecard(ds, screen["woe_features"])
print("scorecard before sign fix: kept", len(sc_raw["process_vars"]), "after IV;", len(sc_raw["features"]), "after VIF")

sc, sign_log, sign_first = fix_scorecard_signs(ds, sc_raw)
print(f"sign consistency BEFORE fix: {sign_first['consistent']}/{sign_first['total']}")
signs = pd.DataFrame(sc["sign_check"])
print(f"sign consistency AFTER fix : {int(signs['consistent'].sum())}/{len(signs)} "
      f"(dropped {len(sign_log)} wrong-sign variables)")

lin_pipe = linear_pipeline()
lin_pipe.fit(design[tr_m], y_all[tr_m])
gate = leakage_gate(ds, screen, list(design.columns) + sc["features"], sc=sc, linear_pipe=lin_pipe, linear_design=design)
gate_tbl = pd.DataFrame(gate["checks"])
print()
print(gate_tbl[["check", "status"]].to_string(index=False))
write_json(ART / "leakage_gate.json", gate)
"""

SCORECARD = r"""coef = sc["coef_table"].copy()
print("statsmodels Logit on the WoE design:", len(coef) - 1, "variables (all coefficients negative = more WoE (safer) lowers PD)")
print(coef.round(4).to_string(index=False))
print("\nwrong-sign drop log (largest p-value dropped each round):")
print(pd.DataFrame(sign_log).round(4).to_string(index=False) if sign_log else "none needed")

points = sc["points"].copy()
points["feature_plain"] = points["feature"].map(lambda f: plain_name(f, defs))
print("\nscorecard points table:", points.shape, "| base points", round(sc["base_points"], 2))
print(points[["feature_plain", "bin", "count", "event_rate", "woe", "points"]].head(14).round(3).to_string(index=False))

# reconstruct one training score from the bin points
i0 = int(np.where(tr_m)[0][0])
row = woe_design(sc["process"], ds["X"].iloc[[i0]][sc["process_vars"]], columns=sc["features"]).iloc[0]
recon = float(sc["base_points"])
for var in sc["features"]:
    pts = points[points["feature"] == var]
    recon += float(pts.iloc[(pts["woe"] - row[var]).abs().argmin()]["points"])
model_score = float(pd_to_score(scorecard_predict(sc, ds["X"].iloc[[i0]]), PDO, BASE_SCORE, BASE_ODDS)[0])
print(f"\nscore reconstruction: sum(points)={recon:.2f} vs model score={model_score:.2f} delta={recon - model_score:+.2f}")

write_json(ART / "scorecard.json", {
    "features": sc["features"], "process_vars": sc["process_vars"], "base_points": sc["base_points"],
    "pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS,
    "coefficients": coef.to_dict(orient="records"), "sign_check": sc["sign_check"],
    "sign_consistent_before_fix": sign_first["consistent"], "sign_total_before_fix": sign_first["total"],
    "sign_consistent": int(signs["consistent"].sum()), "sign_total": int(len(signs)),
    "sign_fix_log": sign_log, "vif_drops": sc["vif_log"],
    "reconstruction_delta_points": recon - model_score, "train_rows": sc["train_rows"],
    "woe_fit_rows": sc["woe_fit_rows"]})
coef.to_csv(ART / "scorecard_coefficients.csv", index=False)
points.to_csv(ART / "scorecard_points.csv", index=False)
pd.DataFrame(sign_log).to_csv(ART / "sign_fix_log.csv", index=False)
signs.to_csv(ART / "sign_check.csv", index=False)
"""

LR = r"""t0 = time.time()
Ztr = lin_pipe.transform(design[tr_m]).astype("float32")
Zva = lin_pipe.transform(design[va_m]).astype("float32")
Zoo = lin_pipe.transform(design[oo_m]).astype("float32")
lr_model, lr_C, lr_tbl, lr_iter = tune_lr(Ztr, y_all[tr_m], Zva, y_all[va_m])
print("L2 logistic regression: C chosen on VALID AUC")
print(lr_tbl.round(5).to_string(index=False))
print(f"chosen C={lr_C}, final fit n_iter={lr_iter}, {time.time() - t0:.0f}s")
"""

TREES_DATA = r"""tree_features = [f for f in screen["tree_features"] if f in ds["X"].columns]
Xt = ds["X"][tree_features]
Xtr, Xva, Xoo = Xt[tr_m], Xt[va_m], Xt[oo_m]
ytr, yva, yoo = y_all[tr_m], y_all[va_m], y_all[oo_m]
print("tree design:", Xtr.shape, Xva.shape, Xoo.shape)

t0 = time.time()
rf_model, rf_rows = fit_rf(Xtr, ytr)
print(f"random forest fitted on a {rf_rows}-row TRAIN subsample (compute cap) in {time.time() - t0:.0f}s | "
      f"VALID AUC {auc(yva, rf_model.predict_proba(Xva)[:, 1]):.4f}")
"""

XGB = r"""t0 = time.time()
xgb_best, xgb_tbl = xgb_random_search(Xtr, ytr, Xva, yva, n_iter=6, sub=500_000)
print("XGBoost random search (6 configs, 500k-row TRAIN subsample, scored on VALID AUC):")
print(xgb_tbl.round(5).to_string(index=False))
print("best:", xgb_best, f"| {time.time() - t0:.0f}s")

t0 = time.time()
xgb_model = fit_xgb(xgb_best, Xtr, ytr, Xva, yva)
print(f"final XGBoost on full TRAIN: best_iteration={xgb_model.best_iteration} | "
      f"VALID AUC {auc(yva, xgb_model.predict_proba(Xva)[:, 1]):.4f} | {time.time() - t0:.0f}s")

# monotone-constrained variant: direction = sign of the TRAIN Spearman correlation (weak features unconstrained)
t0 = time.time()
mono = monotone_signs(Xtr, ytr)
xgbm_model = fit_xgb(xgb_best, Xtr, ytr, Xva, yva, monotone=mono)
print(f"monotone XGBoost: {sum(1 for m in mono if m != 0)} of {len(mono)} features constrained | "
      f"VALID AUC {auc(yva, xgbm_model.predict_proba(Xva)[:, 1]):.4f} | {time.time() - t0:.0f}s")

t0 = time.time()
lgb_fit = fit_tree("lgbm", Xtr, ytr, Xva, yva)
lgb_model = lgb_fit["model"]
print(f"LightGBM (fixed documented parameters, no search): best_iter={lgb_fit['best_iter']} | "
      f"VALID AUC {auc(yva, lgb_model.predict_proba(Xva)[:, 1]):.4f} | {time.time() - t0:.0f}s")
write_json(ART / "tuning.json", {
    "logistic_C_grid": lr_tbl.to_dict(orient="records"), "logistic_C": lr_C,
    "xgb_search": xgb_tbl.to_dict(orient="records"), "xgb_best": xgb_best,
    "xgb_best_iteration": int(xgb_model.best_iteration), "xgb_monotone_constrained_features": int(sum(1 for m in mono if m != 0)),
    "rf": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 50, "max_features": 0.3, "train_rows": rf_rows},
    "lightgbm_best_iter": lgb_fit["best_iter"]})
"""

FREEZE = r"""# ---- predictions on VALID only; the OOT rows are NOT scored yet ----------------
P_V = {
    "dummy": np.full(len(yva), float(ytr.mean())),
    "woe_scorecard": scorecard_predict(sc, ds["X"][va_m]),
    "penalised_lr": lr_model.predict_proba(Zva)[:, 1],
    "random_forest": rf_model.predict_proba(Xva)[:, 1],
    "xgboost": xgb_model.predict_proba(Xva)[:, 1],
    "xgboost_monotone": xgbm_model.predict_proba(Xva)[:, 1],
    "lightgbm": lgb_model.predict_proba(Xva)[:, 1],
}
valid_eval = {n: eval_split(yva, p, week_all[va_m]) for n, p in P_V.items()}
vtab = pd.DataFrame({n: {k: v[k] for k in ("auc", "gini", "ks", "pr_auc", "brier")} for n, v in valid_eval.items()}).T
print("VALID metrics (OOT not yet scored):")
print(vtab.round(4).to_string())

cand = {n: valid_eval[n]["gini"] for n in P_V if n not in ("dummy", "xgboost_monotone")}
frozen = pick_champion(cand, main="xgboost", tol=0.002)
champion = frozen["champion"]
print("\nCHAMPION (rule frozen on VALID):", champion, "| tied set:", frozen["tied"])

cal_sel = choose_calibrator(P_V[champion], yva, week_all[va_m])
print("\ncalibrator selection on VALID (fit early VALID weeks, score late VALID weeks):")
print(cal_sel["selection_table"].round(6).to_string(index=False))
calibrator = cal_sel["calibrators"][cal_sel["choice"]]
print("chosen calibrator:", cal_sel["choice"])

pc_valid = calibrator.predict(P_V[champion]) if calibrator is not None else P_V[champion]
TARGET_BAD = (0.015, 0.020)   # stated target bad rates among approved loans
TARGET_APPROVAL = 0.80        # stated target approval rate
cutoffs = {}
for n in ("penalised_lr", "woe_scorecard", "xgboost", "lightgbm"):
    cutoffs[n] = {f"bad_rate_{int(t * 1000)}": cutoff_for_bad_rate(P_V[n], yva, t) for t in TARGET_BAD}
    cutoffs[n]["approval_80"] = cutoff_for_approval_rate(P_V[n], TARGET_APPROVAL)
print("\ncut-offs chosen on VALID (approve if raw PD < cut-off):")
print(pd.DataFrame(cutoffs).T.round(5).to_string())

frozen_out = {"champion": champion, "rule": frozen["rule"], "valid_gini": frozen["valid_gini"], "tied": frozen["tied"],
              "calibrator": cal_sel["choice"], "calibrator_selection": cal_sel["selection_table"].to_dict(orient="records"),
              "calibrator_split": cal_sel["split"], "targets": {"bad_rate_among_approved": list(TARGET_BAD), "approval_rate": TARGET_APPROVAL},
              "cutoffs_valid_raw_pd": cutoffs,
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "oot_scored_after_this_point": True}
write_json(ART / "champion.json", frozen_out)
print("frozen -> artifacts/champion.json")
"""

OOT = r"""P_O = {
    "dummy": np.full(len(yoo), float(ytr.mean())),
    "woe_scorecard": scorecard_predict(sc, ds["X"][oo_m]),
    "penalised_lr": lr_model.predict_proba(Zoo)[:, 1],
    "random_forest": rf_model.predict_proba(Xoo)[:, 1],
    "xgboost": xgb_model.predict_proba(Xoo)[:, 1],
    "xgboost_monotone": xgbm_model.predict_proba(Xoo)[:, 1],
    "lightgbm": lgb_model.predict_proba(Xoo)[:, 1],
}
oot_eval = {n: eval_split(yoo, p, week_all[oo_m]) for n, p in P_O.items()}
LABEL = {"dummy": "Dummy (base rate)", "penalised_lr": "Logistic regression (L2)", "woe_scorecard": "WoE scorecard",
         "random_forest": "Random forest", "xgboost": "XGBoost", "xgboost_monotone": "XGBoost (monotone)", "lightgbm": "LightGBM"}
otab = pd.DataFrame({LABEL[n]: {k: v[k] for k in ("auc", "gini", "ks", "pr_auc", "brier")} for n, v in oot_eval.items()}).T
print(f"OOT weeks {int(week_all[oo_m].min())}-{int(week_all[oo_m].max())}, n={int(oo_m.sum())}, default rate {yoo.mean():.4f}")
print(otab.round(4).to_string())
print("\nweekly-Gini stability metric (OOT):")
print(pd.Series({LABEL[n]: v["stability"]["stability_metric"] for n, v in oot_eval.items()}).round(4).to_string())

rows = []
for n in P_V:
    r = {"model": n}
    for sp, ev in (("valid", valid_eval[n]), ("oot", oot_eval[n])):
        for k in ("auc", "gini", "pr_auc", "brier", "logloss", "ks"):
            r[f"{sp}_{k}"] = ev[k]
        r[f"{sp}_stability"] = ev["stability"]["stability_metric"]
        r[f"{sp}_mean_weekly_gini"] = ev["stability"]["mean_gini"]
    rows.append(r)
metrics_tbl = pd.DataFrame(rows).sort_values("oot_auc", ascending=False)
metrics_tbl.to_csv(ART / "metrics_table.csv", index=False)

def pack(ev, y, p):
    d = {k: ev[k] for k in ("auc", "gini", "ks", "pr_auc", "brier", "logloss", "stability")}
    d["calibration"] = calibration_summary(y, p)
    d["top5"] = topk_metrics(y, p, 0.05); d["top10"] = topk_metrics(y, p, 0.10)
    return d
write_json(ART / "metrics.json", {
    "models": {n: {"model": n, "valid": pack(valid_eval[n], yva, P_V[n]), "oot": pack(oot_eval[n], yoo, P_O[n]),
                   "weekly": {"valid": valid_eval[n]["weekly"], "oot": oot_eval[n]["weekly"]}} for n in P_V},
    "champion": champion, "champion_rule": frozen["rule"],
    "main_advanced_model": "xgboost",
    "model_meta": {"penalised_lr": {"C": lr_C, "penalty": "l2"}, "xgboost": xgb_best, "random_forest": {"train_rows": rf_rows}}})

fig, ax = plt.subplots(figsize=(11, 5))
for n in ("penalised_lr", "woe_scorecard", "random_forest", "xgboost", "lightgbm"):
    wk = pd.DataFrame(oot_eval[n]["weekly"])
    ax.plot(wk["week"], wk["gini"], marker="o", ms=3, lw=1.3, label=LABEL[n])
ax.set_xlabel("week (WEEK_NUM)"); ax.set_ylabel("weekly Gini"); ax.set_title("Weekly Gini on the out-of-time weeks")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(ART / "figures" / "weekly_gini_oot.png", dpi=130); plt.show()
stab_rows = []
for n in P_V:
    for sp, ev in (("valid", valid_eval[n]), ("oot", oot_eval[n])):
        st = ev["stability"]
        stab_rows.append({"model": n, "split": sp, **{k: st[k] for k in ("mean_gini", "slope", "std_residual", "stability_metric")}})
pd.DataFrame(stab_rows).to_csv(ART / "stability.csv", index=False)
write_json(ART / "weekly_gini.json", {n: oot_eval[n]["weekly"] for n in P_V})

from sklearn.metrics import precision_recall_curve, roc_curve
curves = {"roc": {}, "pr": {}}
for n, p in P_O.items():
    if n == "dummy":
        continue
    fpr, tpr, _ = roc_curve(yoo, p); s = max(1, len(fpr) // 120)
    curves["roc"][n] = [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::s], tpr[::s])]
    pr_, rc_, _ = precision_recall_curve(yoo, p); s = max(1, len(pr_) // 120)
    curves["pr"][n] = [{"recall": float(a), "precision": float(b)} for a, b in zip(rc_[::s], pr_[::s])]
write_json(ART / "curves.json", curves)
"""

BOOT = r"""t0 = time.time()
boot_preds = {n: P_O[n] for n in ("xgboost", "penalised_lr", "woe_scorecard", "random_forest", "lightgbm")}
boot = paired_bootstrap(yoo, boot_preds, ref="xgboost", n_boot=N_BOOT, seed=SEED)
print(f"paired bootstrap of OOT Gini difference, {N_BOOT} resamples ({time.time() - t0:.0f}s):")
bt = pd.DataFrame(boot).T
print(bt.round(4).to_string())
write_json(ART / "bootstrap_ci.json", {"metric": "OOT Gini difference (XGBoost minus other)", "n_boot": N_BOOT, "results": boot})
"""

CALIB = r"""pv_raw, po_raw = P_V[champion], P_O[champion]
variants = {"none (raw)": po_raw}
for nm, cal in cal_sel["calibrators"].items():
    if cal is not None:
        variants[nm] = cal.predict(po_raw)
rows = []
for nm, p in variants.items():
    cs = calibration_summary(yoo, p)
    rows.append({"calibration": nm, "expected_over_actual": cs["expected_actual_ratio"], "mean_pd": cs["mean_predicted"],
                 "actual_rate": cs["observed_rate"], "ece": ece(yoo, p), "brier": brier(yoo, p), "logloss": logloss(yoo, p),
                 "chosen_on_valid": nm.startswith(cal_sel["choice"])})
cal_tbl = pd.DataFrame(rows)
print(f"champion = {champion}; calibrator chosen on VALID = {cal_sel['choice']}")
print("OOT calibration, before and after (Platt vs isotonic fitted on VALID):")
print(cal_tbl.round(5).to_string(index=False))

chosen_p = variants.get(cal_sel["choice"], po_raw)
dec_before = ea_deciles(yoo, po_raw); dec_after = ea_deciles(yoo, chosen_p)
dec = dec_before.merge(dec_after, on="decile", suffixes=("_before", "_after"))
print("\nexpected vs actual by decile (before / after):")
print(dec[["decile", "n_before", "expected_before", "actual_before", "expected_over_actual_before",
           "expected_after", "actual_after", "expected_over_actual_after"]].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
mx = float(max(dec_before["actual"].max(), dec_before["expected"].max()))
for a, d, ttl in ((ax[0], dec_before, "before"), (ax[1], dec_after, f"after ({cal_sel['choice']})")):
    a.plot([0, mx], [0, mx], "k--", lw=1); a.plot(d["expected"], d["actual"], "o-")
    a.set_xlabel("mean predicted PD"); a.set_ylabel("observed default rate"); a.set_title(f"OOT calibration {ttl}"); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ART / "figures" / "calibration.png", dpi=130); plt.show()
write_json(ART / "calibration.json", {"champion": champion, "chosen_on_valid": cal_sel["choice"], "oot": cal_tbl.to_dict(orient="records"),
           "deciles_before": dec_before.to_dict(orient="records"), "deciles_after": dec_after.to_dict(orient="records")})
"""

COVID = r"""# why OOT default rate is lower than TRAIN: calendar view and mix-vs-rate decomposition
Xtr_champ = Xtr  # same tree design
champ_model = {"xgboost": xgb_model, "lightgbm": lgb_model, "random_forest": rf_model}.get(champion)
s_tr = champ_model.predict_proba(Xtr)[:, 1] if champ_model is not None else lr_model.predict_proba(Ztr)[:, 1]
dates = pd.to_datetime(pd.Series(ds["date"]))
cal_df = pd.DataFrame({"q": dates.dt.to_period("Q").astype(str), "split": ds["split"], "y": y_all})
q_tbl = cal_df.groupby("q").agg(n=("y", "size"), default_rate=("y", "mean"), split=("split", lambda s: "/".join(sorted(set(s))))).reset_index()
print("default rate by calendar quarter of date_decision:")
print(q_tbl.round(4).to_string(index=False))
print(f"\nTRAIN default rate {ytr.mean():.4f} | VALID {yva.mean():.4f} | OOT {yoo.mean():.4f}")

dec_oot = decile_decomposition(s_tr, ytr, po_raw, yoo)
dec_val = decile_decomposition(s_tr, ytr, pv_raw, yva)
print("\nTRAIN -> OOT, champion-score deciles fitted on TRAIN:")
print(dec_oot["table"].round(4).to_string(index=False))
print(f"total change in default rate {dec_oot['total_change']:+.4f} = score-mix effect {dec_oot['mix_effect']:+.4f} + within-band rate effect {dec_oot['rate_effect']:+.4f}")
print(f"TRAIN -> VALID: total {dec_val['total_change']:+.4f} = mix {dec_val['mix_effect']:+.4f} + rate {dec_val['rate_effect']:+.4f}")
print("Reading: a near-zero mix effect means applicants did not get safer on the model's own scale; the drop is inside the same risk bands.")
print("ASSUMPTION (context, not tested here): 2020 payment moratoria / COVID-period relief plausibly lowered observed defaults.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
wk_rate = pd.DataFrame({"w": week_all, "y": y_all}).groupby("w")["y"].mean()
ax[0].plot(wk_rate.index, wk_rate.values, lw=1.2)
ax[0].axvline(bounds["train_weeks"][-1] + 0.5, c="gray", ls="--"); ax[0].axvline(bounds["valid_weeks"][-1] + 0.5, c="gray", ls="--")
ax[0].set_title("weekly default rate (dashed = split cut-offs)"); ax[0].set_xlabel("WEEK_NUM"); ax[0].grid(alpha=0.3)
t_ = dec_oot["table"]
ax[1].plot(t_["band"], t_["rate_train"], "o-", label="TRAIN"); ax[1].plot(t_["band"], t_["rate_other"], "o-", label="OOT")
ax[1].set_title("default rate by TRAIN-score decile"); ax[1].set_xlabel("score decile (10 = riskiest)"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ART / "figures" / "default_rate_drop.png", dpi=130); plt.show()
write_json(ART / "default_rate_drop.json", {
    "train_default_rate": float(ytr.mean()), "valid_default_rate": float(yva.mean()), "oot_default_rate": float(yoo.mean()),
    "train_to_oot": {k: dec_oot[k] for k in ("total_change", "mix_effect", "rate_effect")},
    "train_to_valid": {k: dec_val[k] for k in ("total_change", "mix_effect", "rate_effect")},
    "oot_deciles": dec_oot["table"].to_dict(orient="records"), "by_quarter": q_tbl.to_dict(orient="records"),
    "assumption_note": "COVID-period relief as context is an assumption; the numbers above are measured."})
"""

BUSINESS = r"""# (a) capture of defaulters at fixed review capacity
cap = capture_table(yoo, {LABEL[n]: P_O[n] for n in ("penalised_lr", "woe_scorecard", "random_forest", "xgboost", "lightgbm")})
print("share of OOT defaulters caught when the riskiest 5/10/20% of applications are reviewed:")
print(cap.round(4).to_string(index=False))

# (b) cut-off strategies: cut-offs frozen on VALID, outcomes observed on OOT
all_ = np.ones(len(yoo), dtype=bool)
rows = [{"strategy": "approve all", "model": "-", **policy_stats(yoo, all_)}]
strategies = {f"bad rate <= {t:.1%} (VALID)": f"bad_rate_{int(t * 1000)}" for t in TARGET_BAD}
strategies[f"approval rate {TARGET_APPROVAL:.0%} (VALID)"] = "approval_80"
for sname, key in strategies.items():
    for n in ("penalised_lr", "xgboost", "lightgbm"):
        rows.append({"strategy": sname, "model": LABEL[n], **policy_stats(yoo, P_O[n] < cutoffs[n][key])})
strat = pd.DataFrame(rows)
print("\ncut-off strategies on the SAME OOT loans (observed defaults):")
print(strat[["strategy", "model", "approval_rate", "bad_rate_approved", "bad_rate_all", "defaulters_declined_share"]].round(4).to_string(index=False))

# (c) swap sets at the same approval rate: XGBoost vs logistic regression
swap = swap_sets(yoo, P_O["xgboost"], P_O["penalised_lr"], TARGET_APPROVAL)
print(f"\nswap sets at {TARGET_APPROVAL:.0%} approval on OOT (A = XGBoost, B = logistic regression, same number of loans approved by each):")
print(pd.DataFrame({k: v for k, v in swap.items() if isinstance(v, dict)}).T.round(4).to_string())
print(f"portfolio bad rate of approved loans: XGBoost {swap['portfolio_bad_rate_A']:.4f} | logistic regression {swap['portfolio_bad_rate_B']:.4f}")

# (d) risk grades / master scale (10 grades from VALID quantiles of the champion score)
dfx = ds["diagnostics"].to_pandas().set_index(ID_COL).reindex(ds["case_ids"])
ead_all = dfx["credit_amount"].astype(float)
ead_fill = float(np.nanmedian(ead_all[tr_m]))
ead_oot = ead_all[oo_m].fillna(ead_fill).to_numpy()
LGD = LGD_ASSUMED
ms = master_scale(pv_raw, yva, po_raw, yoo, n_grades=10, ead_oot=ead_oot, lgd=LGD)
mono = bool(np.all(np.diff(ms["oot_default_rate"].to_numpy()) >= 0))
print("\nmaster scale: 10 grades from VALID quantiles of the champion score; grade PD = VALID observed default rate")
print(ms.round(5).to_string(index=False))
print("OOT observed default rate monotone across grades:", mono)
print(f"grades where the binomial test rejects the grade PD at 5%: {int((ms['binomial_p'] < 0.05).sum())} of {len(ms)}")
tot_el, tot_real = ms["expected_loss"].sum(), ms["realised_loss"].sum()
print(f"ILLUSTRATIVE expected loss = PD x LGD x EAD (LGD {LGD} assumed, EAD = application credit amount, median-filled where missing): "
      f"expected {tot_el:,.0f} vs realised {tot_real:,.0f} (ratio {tot_el / tot_real:.3f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ms["grade"], ms["grade_pd_valid"], "o-", label="grade PD (VALID)"); ax[0].plot(ms["grade"], ms["oot_default_rate"], "o-", label="OOT observed")
ax[0].set_xlabel("risk grade (1 = safest)"); ax[0].set_title("master scale"); ax[0].legend(); ax[0].grid(alpha=0.3)
for n in ("penalised_lr", "xgboost"):
    o = np.argsort(P_O[n], kind="stable"); cum = np.cumsum(yoo[o]) / np.arange(1, len(yoo) + 1)
    ax[1].plot(np.arange(1, len(yoo) + 1) / len(yoo), cum, label=LABEL[n])
ax[1].set_xlabel("approval rate (lowest PD first)"); ax[1].set_ylabel("bad rate among approved"); ax[1].set_title("OOT approval vs bad rate"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(ART / "figures" / "credit_policy.png", dpi=130); plt.show()

write_json(ART / "business.json", {
    "capture": cap.to_dict(orient="records"), "cutoff_strategies_oot": strat.to_dict(orient="records"),
    "swap_sets": swap, "master_scale": ms.to_dict(orient="records"), "master_scale_oot_monotone": mono,
    "expected_loss_illustrative": {"lgd_ASSUMED": LGD, "ead": "application credit amount (credamount_770A), median-filled when missing",
                                  "expected_loss_total": float(tot_el), "realised_loss_total": float(tot_real), "ratio": float(tot_el / tot_real)},
    "note": "all rates are observed OOT outcomes; LGD is an assumption; expected loss is illustrative"})
ms.to_csv(ART / "master_scale.csv", index=False)
strat.to_csv(ART / "cutoff_strategies.csv", index=False)
write_json(ART / "decision.json", {
    "targets": {"bad_rate_among_approved": list(TARGET_BAD), "approval_rate": TARGET_APPROVAL},
    "cutoffs_valid_raw_pd": cutoffs, "oot": strat.to_dict(orient="records"), "swap_sets": swap})
"""

MONITOR = r"""s_oo_raw = po_raw
score_psi = psi(s_tr, s_oo_raw)
score_psi_valid = psi(s_tr, pv_raw)
print(f"score PSI TRAIN -> VALID {score_psi_valid:.4f} | TRAIN -> OOT {score_psi:.4f} ({psi_band(score_psi)})")

gain = pd.Series(xgb_model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
top = gain.head(15)
top_tbl = pd.DataFrame({"feature": top.index, "gain": top.values})
top_tbl["gain_share"] = top_tbl["gain"] / gain.sum()
top_tbl["plain_name"] = top_tbl["feature"].map(lambda f: plain_name(f, defs))
print("\ntop 15 XGBoost features by gain:")
print(top_tbl[["plain_name", "gain_share", "feature"]].round(4).to_string(index=False))
top_tbl.to_csv(ART / "top_features.csv", index=False)

csi_tab = psi_table(Xtr, Xoo, top_tbl["feature"].tolist())
csi_tab["plain_name"] = csi_tab["feature"].map(lambda f: plain_name(f, defs))
print("\nfeature stability (PSI/CSI, TRAIN -> OOT) for those top 15 features:")
print(csi_tab[["plain_name", "psi", "band"]].round(4).to_string(index=False))
psi_all = psi_table(Xtr, Xoo, tree_features)
psi_all.to_csv(ART / "psi.csv", index=False)
print("\nall tree features PSI bands:", psi_all["band"].value_counts().to_dict())

adv = adversarial_validation(Xtr, Xoo, tree_features)
print("adversarial AUC TRAIN vs OOT:", round(adv["auc"], 4), "-", adv["interpretation"])
write_json(ART / "drift.json", {"adversarial": adv, "psi": psi_all.to_dict(orient="records"), "psi_bands": psi_all["band"].value_counts().to_dict()})

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(csi_tab["plain_name"].str.slice(0, 48)[::-1], csi_tab["psi"][::-1], color="#b0483a")
ax[0].axvline(PSI_WATCH, ls="--", c="orange"); ax[0].axvline(PSI_ALERT, ls="--", c="red"); ax[0].set_title("top-15 feature PSI (TRAIN->OOT)")
ax[1].hist(s_tr, bins=50, alpha=0.5, density=True, label="TRAIN"); ax[1].hist(po_raw, bins=50, alpha=0.5, density=True, label="OOT")
ax[1].set_title(f"champion score distribution, PSI={score_psi:.4f}"); ax[1].legend()
fig.tight_layout(); fig.savefig(ART / "figures" / "psi.png", dpi=130); plt.show()

mon = build_monitoring(pv_raw, po_raw, ds, tree_features[:60])
mon["score_drift"]["psi_train_to_oot"] = float(score_psi)
mon["score_drift"]["psi_band"] = psi_band(score_psi)
mon["score_psi_train_to_valid"] = float(score_psi_valid)
mon["top15_feature_psi"] = csi_tab.to_dict(orient="records")
write_json(ART / "monitoring.json", mon)
write_json(ART / "evaluation.json", {
    "calibration": dec_after.rename(columns={"expected": "mean_pd", "actual": "observed"}).to_dict(orient="records"),
    "bootstrap": boot, "shap": {"top_features": top_tbl[["feature", "plain_name", "gain_share"]].to_dict(orient="records")},
    "note": "'shap' key kept for the app; values are XGBoost gain shares"})
"""

FINAL = r"""gates = []
gates += [{"gate": f"leakage:{c['check']}", "status": c["status"], "detail": c["detail"]} for c in gate["checks"]]
dummy_auc = oot_eval["dummy"]["auc"]
beats = {n: oot_eval[n]["auc"] > dummy_auc for n in P_O if n != "dummy"}
gates.append({"gate": "dummy_baseline_beaten", "status": "PASS" if all(beats.values()) else "FAIL", "detail": beats})
a_ch = oot_eval[champion]["auc"]
gates.append({"gate": "plausible_oot_range", "status": "PASS" if 0.74 <= a_ch <= 0.86 else "WARN",
              "detail": {"champion_oot_auc": a_ch, "range": [0.74, 0.86]}})
gates.append({"gate": "valid_oot_gap_under_0.05", "status": "PASS" if abs(valid_eval[champion]["auc"] - a_ch) < 0.05 else "WARN",
              "detail": {"valid_auc": valid_eval[champion]["auc"], "oot_auc": a_ch}})
gates.append({"gate": "scorecard_all_signs_consistent", "status": "PASS" if bool(signs["consistent"].all()) else "FAIL",
              "detail": {"consistent": int(signs["consistent"].sum()), "total": int(len(signs)), "before_fix": sign_first}})
ea_after = float(cal_tbl.loc[cal_tbl["chosen_on_valid"], "expected_over_actual"].iloc[0])
gates.append({"gate": "calibration_expected_actual_after", "status": "PASS" if 0.8 <= ea_after <= 1.25 else "WARN",
              "detail": {"after": ea_after, "before": float(cal_tbl.iloc[0]["expected_over_actual"])}})
gates.append({"gate": "score_psi_train_to_oot", "status": "PASS" if score_psi < PSI_ALERT else "WARN", "detail": {"psi": float(score_psi)}})
b = boot["xgboost_minus_penalised_lr"]
gates.append({"gate": "xgb_minus_lr_ci_excludes_zero", "status": "PASS" if b["ci95_lo"] > 0 else "WARN", "detail": b})
gates.append(cov_gate_row)
gates.append(cal_gate_row)
gates.append(proxy_gate_row)
gates.append({"gate": "adversarial_auc_and_top_drivers_reported", "status": "PASS",
              "detail": {"auc": adv["auc"], "top_drivers": [{"feature": r["feature"], "gain_share": r["gain_share"]} for r in adv["top_drifting"][:8]],
                         "note": "reported, not gated on its value"}})
gates.append({"gate": "champion_frozen_before_oot", "status": "PASS", "detail": {"file": "artifacts/champion.json"}})
gt = pd.DataFrame([{"gate": g["gate"], "status": g["status"]} for g in gates])
print(gt.to_string(index=False))
write_json(ART / "quality_gates.json", {"gates": gates, "n_pass": int((gt["status"] == "PASS").sum()),
                                        "n_fail": int((gt["status"] == "FAIL").sum()), "n_warn": int((gt["status"] == "WARN").sum())})

# ---- models + scored sample for the Streamlit app ------------------------------
xgb_model.save_model(str(ART / "models" / "champion_xgb.txt"))
joblib.dump({"process": sc["process"], "coef_table": sc["coef_table"], "features": sc["features"],
             "process_vars": sc["process_vars"], "base_points": sc["base_points"]}, ART / "models" / "scorecard.joblib", compress=3)
joblib.dump({"pipeline": lin_pipe, "model": lr_model}, ART / "models" / "penalised_lr.joblib", compress=3)
champ_kind = {"xgboost": "xgb", "lightgbm": "lgbm", "random_forest": "rf", "penalised_lr": "linear", "woe_scorecard": "scorecard"}[champion]
if champion == "lightgbm":
    lgb_model.save_model(str(ART / "models" / "champion_lgbm.txt")) if hasattr(lgb_model, "save_model") else lgb_model.booster_.save_model(str(ART / "models" / "champion_lgbm.txt"))
model_file = {"xgboost": "models/champion_xgb.txt", "lightgbm": "models/champion_lgbm.txt"}.get(champion, "models/champion_xgb.txt")

oot_pos = np.where(oo_m)[0]
rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(len(oot_pos), min(3000, len(oot_pos)), replace=False))
sample_df = ds["X"].iloc[oot_pos[pick]][tree_features].copy()
sample_df.insert(0, "case_id", ds["case_ids"][oot_pos[pick]]); sample_df.insert(1, "week", week_all[oot_pos[pick]]); sample_df.insert(2, "target", yoo[pick])
for n in (champion, "woe_scorecard"):
    sample_df[f"pd_{n}"] = P_O[n][pick]
sample_df.to_parquet(ART / "scored_oot_sample.parquet", index=False)

write_json(ART / "feature_meta.json", {
    "champion": champion, "kind": champ_kind, "features": tree_features, "cat_features": [f for f in tree_features if f in ds["cat_cols"]],
    "model_file": model_file, "scorecard_file": "models/scorecard.joblib", "linear_file": "models/penalised_lr.joblib",
    "typical_values": {f: {"p05": float(np.nanpercentile(ds["X"][f].to_numpy(), 5)), "p50": float(np.nanpercentile(ds["X"][f].to_numpy(), 50)),
                           "p95": float(np.nanpercentile(ds["X"][f].to_numpy(), 95)), "missing": float(np.mean(~np.isfinite(ds["X"][f].to_numpy())))}
                       for f in tree_features},
    "scorecard_features": sc["features"], "pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS})

import sklearn
env = {"python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
       "numpy": np.__version__, "polars": pl.__version__, "lightgbm": __import__("lightgbm").__version__,
       "xgboost": __import__("xgboost").__version__, "statsmodels": __import__("statsmodels").__version__,
       "optbinning": __import__("optbinning").__version__, "kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local"),
       "sample_cases": SAMPLE_CASES, "runtime_seconds": round(time.time() - T_START, 1),
       "devices": {"xgboost": DEVICE_LOG.get("xgboost", "cpu"), "xgboost_monotone": DEVICE_LOG.get("xgboost", "cpu"), "lightgbm": "cpu", "random_forest": "cpu",
                   "penalised_lr": "cpu", "woe_scorecard": "cpu"}, "gpu_visible": xgb_device() == "cuda"}
write_json(ART / "environment.json", env)
print(json.dumps(env, indent=2))

import hashlib
manifest = [{"file": str(p.relative_to(ART)).replace("\\", "/"), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ART.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
write_json(ART / "MANIFEST.json", {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": manifest})
print(f"manifest: {len(manifest)} files | total runtime {time.time() - T_START:.0f}s")
print("DONE")
"""


def build(smoke: int | None = None) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [new_markdown_cell(INTRO), new_code_cell(INSTALL), new_code_cell(IMPORTS)]
    cells.append(new_markdown_cell("## Project modules\n\nThe tested `src/` package inlined verbatim (cross-module imports stripped)."))
    for m in MODULES:
        cells.append(new_code_cell(module_source(m)))
    cells.append(new_code_cell(
        f"SAMPLE_CASES = {smoke if smoke else None}  # None = full run; an integer = smoke run on a sample of cases\n"
        "N_BOOT = 1000 if SAMPLE_CASES is None else 100\n"
        "print('SAMPLE_CASES =', SAMPLE_CASES, '| N_BOOT =', N_BOOT)"))

    def stage(title, body, note):
        cells.append(new_markdown_cell(f"## {title}\n\n{note}"))
        cells.append(new_code_cell(body))

    stage("1. Data discovery", DISCOVERY, "Resolve the parquet root, fingerprint every file, read the base table.")
    stage("2. Time split by week", PROFILE, "Consecutive weeks are divided 70/15/15; every week lands in exactly one partition.")
    stage("3. Point-in-time feature build", FEATURES, "History tables are aggregated to one row per application after dropping records dated after `date_decision`. Column screening uses TRAIN labels only.")
    stage("4. Leakage gate, scorecard bins and sign fix", GATE, "IV screen, single-feature AUC quarantine, WoE monotone binning, then wrong-sign variables are dropped one by one until every coefficient sign matches its bins. The gate raises on failure.")
    stage("5. Scorecard coefficients and points", SCORECARD, "Coefficient table, drop log, points table (PDO scaling) and a score-reconstruction check.")
    stage("6. Logistic regression", LR, "L2 logistic regression on the imputed, standardised design (imputer and scaler fitted on TRAIN). C chosen on VALID.")
    stage("7. Random forest", TREES_DATA, "Random forest on a TRAIN subsample (documented compute cap).")
    stage("8. XGBoost, monotone XGBoost, LightGBM", XGB, "Small random search for XGBoost on VALID; LightGBM is a check with fixed parameters; the monotone variant constrains each feature to its TRAIN Spearman direction.")
    stage("9. Freeze: champion, calibrator, cut-off (VALID only)", FREEZE, "Champion, calibrator and cut-offs are decided here from VALID, and written to `champion.json` before OOT is scored.")
    stage("10. Score OOT: metrics and weekly Gini", OOT, "AUC, Gini, KS, PR-AUC, Brier for every model, weekly Gini and the competition stability metric.")
    stage("11. Paired bootstrap of the Gini difference", BOOT, "Same resampled applications for both models in each draw.")
    stage("12. Calibration (Platt vs isotonic)", CALIB, "Both fitted on VALID, compared on OOT before and after, with an expected-vs-actual table by decile.")
    stage("13. Why the default rate fell", COVID, "Calendar view and an exact mix-vs-rate decomposition on TRAIN-score deciles.")
    stage("14. Credit-policy layer", BUSINESS, "Capture at fixed review capacity, cut-offs chosen on VALID at stated bad-rate / approval-rate targets, swap sets, 10-grade master scale with a binomial test per grade, illustrative expected loss.")
    stage("15. Monitoring and interpretability", MONITOR, "Score PSI, top-feature PSI/CSI, adversarial validation, top-15 XGBoost features in plain English.")
    stage("16. Quality gates, models, manifest", FINAL, "Gate table, saved models for the app, hashed manifest.")

    k = 0
    for c in cells:  # end-of-cell markers let scripts/assemble_executed.py split the Kaggle log per cell
        if c.cell_type == "code":
            k += 1
            c.source = c.source.rstrip("\n") + f'\nprint("[end of code cell {k}]")'
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11"}}
    return nb


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "HomeCreditStability.ipynb")
    args = ap.parse_args()
    nb = build(args.smoke)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(args.out))
    print(f"wrote {args.out} with {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
