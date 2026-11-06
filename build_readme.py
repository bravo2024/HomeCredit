"""Render README.md from artifacts/ only. No metric is typed by hand.

    python build_readme.py [--artifacts artifacts] [--out README.md]

Every number is read from ``artifacts/`` (or from the constants in ``src/hc_config.py``).
The run history compares against ``artifacts_v1_partsbug/``, ``kaggle/pull_v3_gpu/`` and
``kaggle/pull_v4_gpu/`` when they exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import hc_config as C  # noqa: E402

LABEL = {"dummy": "Dummy (base rate)", "penalised_lr": "Logistic regression (L2)", "woe_scorecard": "WoE scorecard",
         "random_forest": "Random forest", "xgboost": "XGBoost", "xgboost_monotone": "XGBoost (monotone)",
         "lightgbm": "LightGBM"}
ORDER = ["penalised_lr", "woe_scorecard", "random_forest", "xgboost", "xgboost_monotone", "lightgbm", "dummy"]
COMP_URL = "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability"
KERNEL = "vivekkopthsd/home-credit-stability-pd-gpu"


def J(d: Path, name: str):
    return json.loads((d / name).read_text(encoding="utf-8"))


def md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.3f}") -> str:
    fmt = fmt or {}
    df = df.astype(object)
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("-" if pd.isna(v) else fmt.get(c, default).format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def short(f: str) -> str:
    """Feature name without the table prefix."""
    return f.split("__", 1)[1] if "__" in f else f


def history_row(label: str, change: str, d: Path) -> dict | None:
    if not d.exists() or not (d / "metrics_table.csv").exists():
        return None
    mt = pd.read_csv(d / "metrics_table.csv").set_index("model")
    ch = J(d, "champion.json")["champion"]
    adv = J(d, "drift.json")["adversarial"]
    top = adv["top_drifting"][0]
    if (d / "feature_coverage.json").exists():
        cv = J(d, "feature_coverage.json")
        n_feat, n_null = cv["n_final_features"], str(cv["n_null_in_oot"])
    else:
        fm = J(d, "feature_meta.json")
        n_feat = len(fm["features"])
        s = pd.read_parquet(d / "scored_oot_sample.parquet")
        n_null = f"{sum(1 for f in fm['features'] if f in s.columns and s[f].isna().all())} (OOT sample)"
    return {"Run": label, "Change": change, "Final features": n_feat, "Empty in OOT": n_null,
            "Selected model": LABEL[ch], "Selected OOT Gini": float(mt.loc[ch, "oot_gini"]),
            "XGBoost OOT Gini": float(mt.loc["xgboost", "oot_gini"]), "Adversarial AUC": float(adv["auc"]),
            "Top adversarial driver (gain share)": f"`{short(top['feature'])}` ({top['gain_share']:.2f})"}


def render(art: Path) -> str:  # noqa: C901
    mt = pd.read_csv(art / "metrics_table.csv").set_index("model")
    split = J(art, "split.json")
    champ = J(art, "champion.json")
    boot = J(art, "bootstrap_ci.json")["results"]
    cal = J(art, "calibration.json")
    biz = J(art, "business.json")
    drop = J(art, "default_rate_drop.json")
    scj = J(art, "scorecard.json")
    mon = J(art, "monitoring.json")
    gates = J(art, "quality_gates.json")
    leak = J(art, "leakage_gate.json")
    env = J(art, "environment.json")
    tun = J(art, "tuning.json")
    fm = J(art, "feature_meta.json")
    fs = J(art, "feature_screening.json")
    pit = J(art, "pit_audit.json")
    drift = J(art, "drift.json")
    cov = J(art, "feature_coverage.json")
    manifest = J(art, "MANIFEST.json")
    stab = pd.read_csv(art / "stability.csv")
    cut = pd.read_csv(art / "cutoff_strategies.csv")
    ms = pd.read_csv(art / "master_scale.csv")
    iv = pd.read_csv(art / "iv_table.csv")
    psi_csv = pd.read_csv(art / "psi.csv")
    top = pd.read_csv(art / "top_features.csv")
    proxy = pd.read_csv(art / "calendar_proxy_screen.csv")
    parts_path = art / "table_parts_check.json"
    parts = J(art, "table_parts_check.json") if parts_path.exists() else None

    sp = pd.DataFrame(split["table"]).set_index("split")
    tr, va, oo = sp.loc["train"], sp.loc["valid"], sp.loc["oot"]
    n_tot = int(sp["n"].sum())
    sel = champ["champion"]
    sd = mon["score_drift"]
    adv = drift["adversarial"]
    features = fm["features"]
    n_feat = len(features)
    cov_rows = pd.DataFrame(cov["features"]).set_index("feature")
    b = split["bounds"]
    cs80 = cut[cut["strategy"].str.startswith("approval rate")].set_index("model")
    cap = pd.DataFrame(biz["capture"]).set_index("model")
    el = biz["expected_loss_illustrative"]
    sw = biz["swap_sets"]
    wk_tr = f"{b['train_weeks'][0]}-{b['train_weeks'][-1]}"
    wk_va = f"{b['valid_weeks'][0]}-{b['valid_weeks'][-1]}"
    wk_oo = f"{b['oot_weeks'][0]}-{b['oot_weeks'][-1]}"
    n_pass = sum(1 for g in gates["gates"] if g["status"] == "PASS")
    n_gates = len(gates["gates"])
    dev = env.get("devices", {})
    gpu_models = sorted(LABEL[m] for m, d in dev.items() if d == "cuda")
    cpu_models = sorted(LABEL[m] for m, d in dev.items() if d == "cpu")
    n_tables = len(pit["tables"]) + len(pit["missing_tables"])

    L: list[str] = []
    P = L.append

    # ------------------------------------------------------------------ title and summary
    P("# Credit Risk Modelling | Loan Default Prediction with Out-of-Time Validation\n")
    P(f"This project predicts loan default on the Kaggle competition data *Home Credit - Credit Risk Model Stability* "
      f"({n_tot:,} applications, overall default rate {pct(float((sp['n'] * sp['default_rate']).sum() / n_tot))}). "
      f"The split is by calendar week. Models are fitted on {tr['date_min']} to {tr['date_max']} (weeks {wk_tr}, {int(tr['n']):,} applications), "
      f"tuned on {va['date_min']} to {va['date_max']} (weeks {wk_va}, {int(va['n']):,}) and scored once on {oo['date_min']} to {oo['date_max']} "
      f"(weeks {wk_oo}, {int(oo['n']):,} applications, default rate {pct(oo['default_rate'])}). "
      f"Seven models share one feature set and one split: a base-rate dummy, L2 logistic regression, a WoE scorecard, random forest, XGBoost, monotone XGBoost and LightGBM. "
      f"The selection rule was frozen on VALID before OOT was scored and picked **{LABEL[sel]}**. "
      f"On OOT it reaches AUC {mt.loc[sel, 'oot_auc']:.3f}, Gini {mt.loc[sel, 'oot_gini']:.3f} and KS {mt.loc[sel, 'oot_ks']:.3f}, "
      f"against Gini {mt.loc['penalised_lr', 'oot_gini']:.3f} for logistic regression "
      f"(paired-bootstrap difference {boot['xgboost_minus_penalised_lr']['point_gini_diff']:+.3f}, 95% interval "
      f"{boot['xgboost_minus_penalised_lr']['ci95_lo']:+.3f} to {boot['xgboost_minus_penalised_lr']['ci95_hi']:+.3f}). "
      f"The model uses {n_feat} features built from {len(cov['by_source'])} of the {n_tables} source tables.\n")
    P("The data are used under the competition rules and are not in this repository. Results are specific to this split and to one run, and they are not comparable with the Kaggle leaderboard "
      "(see the limitations). This document describes version 5 of the pipeline. Versions 1 to 4 contained defects that are described in the run history.\n")

    # ------------------------------------------------------------------ results
    P("## 1. Results\n")
    P(f"Source: `artifacts/metrics_table.csv`. OOT is weeks {wk_oo}. VALID is weeks {wk_va} and also drives early stopping, tuning, calibrator and cut-offs, so VALID figures are optimistic for the tuned models.\n")
    rows = []
    for m in ORDER:
        r = mt.loc[m]
        rows.append({"Model": LABEL[m], "OOT AUC": r["oot_auc"], "VALID AUC": r["valid_auc"], "OOT Gini": r["oot_gini"], "VALID Gini": r["valid_gini"],
                     "OOT KS": r["oot_ks"], "VALID KS": r["valid_ks"], "OOT PR-AUC": r["oot_pr_auc"], "VALID PR-AUC": r["valid_pr_auc"]})
    P(md(pd.DataFrame(rows)))
    P(f"\nThe OOT base rate is {pct(oo['default_rate'])}, which is the PR-AUC of a random ranking. The dummy KS is not zero because tied scores are split by row order.\n")
    P(f"Paired bootstrap of the OOT Gini difference with {J(art, 'bootstrap_ci.json')['n_boot']} resamples of applications (`artifacts/bootstrap_ci.json`). The resampling ignores week-level dependence and refit variability.\n")
    brow = []
    for k, lab in (("xgboost_minus_penalised_lr", "logistic regression"), ("xgboost_minus_woe_scorecard", "WoE scorecard"),
                   ("xgboost_minus_random_forest", "random forest"), ("xgboost_minus_lightgbm", "LightGBM")):
        v = boot[k]
        brow.append({"XGBoost minus": lab, "Gini difference": v["point_gini_diff"], "95% lower": v["ci95_lo"], "95% upper": v["ci95_hi"], "Share of resamples above 0": v["share_positive"]})
    P(md(pd.DataFrame(brow)))
    P("\n![weekly Gini on OOT](artifacts/figures/weekly_gini_oot.png)\n")

    # ------------------------------------------------------------------ data
    P("## 2. Data and point-in-time design\n")
    P(f"* Source: Kaggle competition [{COMP_URL.split('/')[-1]}]({COMP_URL}), labelled train files only. The competition page restricts the data to competition use, so `data/raw/` is git-ignored. "
      f"Size and sha256 of each of the {len(J(art, 'data_provenance.json')['files'])} train parquet files are in `artifacts/data_provenance.json`, and the sha256 of {len(manifest['files'])} output files is in `artifacts/MANIFEST.json`.\n"
      f"* Unit: one row per credit application (`case_id`). Prediction time: `date_decision`. Target: the competition `target`, a default flag observed after the decision.\n"
      f"* Point-in-time rule (`pit_filter` in `src/hc_data.py`): a history record is dropped if its latest date, or its latest (year, month) pair, is after `date_decision`. Records with no usable date are kept. Drop counts per table are in `artifacts/pit_audit.json`.\n")
    P(f"**Tables.** `TABLE_SPECS` in `src/hc_config.py` lists {n_tables} logical tables. "
      f"Each logical table is the union of all its numbered part files (`table_parts` and `scan_table` in `src/hc_data.py`). Schemas are aligned across parts, missing columns become null and dtypes are relaxed to a common type. "
      f"Depth 1 and 2 tables are aggregated part by part and the results are stacked. This is exact because each `case_id` lives in exactly one part, and the loader raises if that ever fails.\n")
    pit_by = {t["table"]: t for t in pit["tables"]}
    fcount = pd.Series([f.split("__")[0] for f in features]).value_counts().to_dict()
    trows = []
    for t in fs["tables_used"]:
        pt = pit_by.get(t, {})
        trows.append({"Logical table": t, "Depth": pt.get("depth", ""), "Part files": pt.get("n_parts", ""),
                      "Rows in": f"{pt['rows_in']:,}" if pt.get("rows_in") else "-",
                      "Dropped by PIT rule": pct(pt.get("drop_share", 0.0), 1) if pt.get("rows_in") else "-",
                      "Applications with data": f"{pt.get('rows_out', 0):,}", "Final features": fcount.get(t, 0)})
    P(md(pd.DataFrame(trows)))
    P(f"\nSource: `artifacts/pit_audit.json`, `artifacts/feature_meta.json`. {', '.join('`' + m + '`' for m in pit['missing_tables'])} was read but gave no usable column. "
      "`Rows in` is empty for tables without a date column, where the rule does not apply.\n")
    P("**Relative dates.** Absolute calendar values identify when an application happened and separate training from later weeks by construction. "
      "Date columns (suffix `D`) therefore enter only as days before `date_decision`, and the year and month column pairs (`pmts_year_*`, `pmts_month_*`) only as months before the decision month (`relative_expr` in `src/hc_data.py`). "
      "These are aggregated as `rel_mean`, `rel_max`, `rel_min` and `rel_std`. A date column is never used as a categorical, and the index columns `num_group*` are not candidates. "
      f"Quality gate `no_absolute_calendar_features` fails the run if a final feature is built from a calendar column without this transform. {sum(1 for f in features if '__rel_' in f)} of the {n_feat} final features are relative-date aggregates.\n")
    if parts:
        pf = {x["file"]: x for x in parts["files"]}
        tot = parts["split_totals"]
        lrows = []
        for spec in C.TABLE_SPECS:
            pre = spec["prefix"]
            pat = re.compile(rf"^{re.escape(pre)}(_\d+)?$")
            fl = [pf[k] for k in pf if pat.match(k)]
            if not fl:
                continue
            cs = {s: sum(x["cases_" + s] for x in fl) for s in ("train", "valid", "oot")}
            src = cov["by_source"].get(pre)
            lrows.append({"Logical table": pre, "Weeks": f"{min(x['week_min'] for x in fl)}-{max(x['week_max'] for x in fl)}",
                          "TRAIN apps covered": pct(cs["train"] / tot["train"], 1), "VALID apps covered": pct(cs["valid"] / tot["valid"], 1),
                          "OOT apps covered": pct(cs["oot"] / tot["oot"], 1),
                          "Mean non-null share of final features in OOT": pct(src["mean_coverage_oot"], 1) if src else "-"})
        P("**Coverage by split.** `scripts/check_table_parts.py` reads only `case_id` from every train file and joins `WEEK_NUM` from `train_base` (`artifacts/table_parts_check.json`). "
          "The last column is the mean non-null share of the table's final features among OOT applications (`artifacts/feature_coverage.json`). It cannot exceed the share of applications the table covers.\n")
        P(md(pd.DataFrame(lrows)))
        P("\nTwo tables cover only part of the calendar. `train_tax_registry_c_1` ends in week 40 and `train_tax_registry_b_1` starts in week 62. Neither has enough recent TRAIN rows to pass the coverage rule, so neither contributes features.\n")
    P(f"**Coverage gate.** `artifacts/feature_coverage.json` stores the non-null share of every model column in TRAIN, VALID and OOT. "
      f"Candidates need at least {C.MIN_RECENT_COVERAGE:.0%} non-null values among TRAIN rows of the last {C.RECENT_TRAIN_WEEKS} TRAIN weeks, which uses no VALID or OOT information. "
      f"The run fails if a final feature has no value at all in VALID or OOT. Result: {cov['n_null_in_valid']} of {cov['n_final_features']} final features are empty in VALID and {cov['n_null_in_oot']} in OOT.\n")

    # ------------------------------------------------------------------ split
    P("## 3. Time split\n")
    P("`split_bounds` in `src/hc_data.py` divides the consecutive weeks 70 / 15 / 15 so that every week falls in exactly one partition. Source: `artifacts/split.json`.\n")
    srows = [{"Split": k.upper(), "Weeks": f"{int(sp.loc[k, 'week_min'])}-{int(sp.loc[k, 'week_max'])}", "Number of weeks": int(sp.loc[k, "n_weeks"]),
              "From": sp.loc[k, "date_min"], "To": sp.loc[k, "date_max"], "Applications": f"{int(sp.loc[k, 'n']):,}",
              "Share": pct(sp.loc[k, "share"], 1), "Default rate": pct(sp.loc[k, "default_rate"])} for k in ("train", "valid", "oot")]
    P(md(pd.DataFrame(srows)))
    tt = drop["train_to_oot"]
    P(f"\nThe default rate falls from {pct(drop['train_default_rate'])} to {pct(drop['oot_default_rate'])}. An exact decomposition on TRAIN-score deciles (`artifacts/default_rate_drop.json`) splits the change of {tt['total_change']:+.4f} "
      f"into a score-mix effect of {tt['mix_effect']:+.4f} and a within-band rate effect of {tt['rate_effect']:+.4f}. The drop happens inside the same risk bands and is not explained by safer applicants. "
      "The cause was not tested. Payment relief during 2020 is a plausible context but remains an assumption.\n")
    P("![default rate](artifacts/figures/default_rate_drop.png)\n")

    # ------------------------------------------------------------------ features
    P("## 4. Features and screening\n")
    scr = pd.DataFrame(fs["screening"])
    n_q = len(leak["checks"][4]["detail"]["quarantined"])
    n_low = len(cov.get("low_recent_coverage_dropped", []))
    n_proxy = int(proxy["dropped"].sum())
    P(f"The pipeline builds {fs['n_features_built']} model columns ({fs['n_categorical']} categorical codes) from {len(scr)} screened raw columns. "
      f"For each depth 1 and 2 table a column needs at least {C.MIN_COVERAGE:.0%} non-null values on a TRAIN sample, and the top {C.MAX_NUMERIC_COLS_PER_TABLE} numeric and {C.MAX_CATEGORICAL_COLS_PER_TABLE} categorical columns are kept by univariate AUC. "
      "Aggregates per application are the record count, days since the newest record, mean, max, min and std of each numeric column, and the distinct count and largest category code of each categorical column. "
      "Category codes are ranks in the sorted TRAIN levels of the whole logical table, so they are identical across part files.\n")
    P("Selection steps in order, all on TRAIN rows (`feature_screen` in `src/hc_pipeline.py`):\n")
    drop_txt = ", ".join(f"`{short(r.feature)}` (rho {r.rho:.3f})" for r in proxy[proxy["dropped"]].itertuples())
    P(f"1. Coverage rule: {n_low} columns removed because their non-null share was below {C.MIN_RECENT_COVERAGE:.0%} in the last {C.RECENT_TRAIN_WEEKS} TRAIN weeks.\n"
      f"2. Calendar-proxy screen: every relative-date feature is correlated (Spearman) with `WEEK_NUM` on a TRAIN sample and dropped if the absolute correlation is at least {C.CALENDAR_PROXY_RHO}. "
      f"{len(proxy)} features were screened and {n_proxy} dropped: {drop_txt}. Gate `no_calendar_proxy_features` checks that no final feature breaches the threshold. Source: `artifacts/calendar_proxy_screen.csv`.\n"
      f"3. Univariate leakage screen: a column whose own AUC exceeds {C.UNIVARIATE_AUC_SCREEN} is quarantined. Quarantined: {n_q}.\n"
      f"4. Information value on 10 quantile bins: keep IV of at least {C.IV_MIN}. {len(iv)} columns pass, and they are the {n_feat} tree features (`artifacts/iv_table.csv`).\n"
      f"5. Linear design: the top {C.MAX_LINEAR_FEATURES} numeric columns by IV plus one-hot codes of categoricals with at most 8 TRAIN levels, then removal of columns with absolute correlation of at least 0.95. Imputation and scaling are fitted on TRAIN.\n"
      f"6. Scorecard variables: the top {C.MAX_WOE_FEATURES} numeric columns by IV, then the steps in section 5.\n")
    P("Final feature count by source table, with the mean non-null share of those features per split (`artifacts/feature_coverage.json`, `by_source`):\n")
    brows = [{"Source table": k, "Final features": v["n_features"], "TRAIN": pct(v["mean_coverage_train"], 1),
              "VALID": pct(v["mean_coverage_valid"], 1), "OOT": pct(v["mean_coverage_oot"], 1)} for k, v in sorted(cov["by_source"].items())]
    P(md(pd.DataFrame(brows)))
    P("\nThe leakage gate (section 8) checks structure, ordering, single-feature AUC and fitting scope. It cannot prove that no feature carries post-decision information.\n")

    # ------------------------------------------------------------------ models
    P("## 5. Models, tuning and the selection rule\n")
    P("All models see the same split and the same screened columns. Hyper-parameter choices use VALID only (`artifacts/tuning.json`).\n")
    P(f"* **Logistic regression.** L2 penalty, saga solver, C chosen on VALID AUC from {len(tun['logistic_C_grid'])} values ({', '.join(str(r['C']) for r in tun['logistic_C_grid'])}). Chosen C = {tun['logistic_C']}.\n"
      f"* **WoE scorecard.** Monotone optimal binning (optbinning) on a TRAIN sample, IV filter, iterative VIF pruning at {C.VIF_MAX}, statsmodels Logit, then wrong-sign variables are dropped one at a time. "
      f"Sign consistency was {scj['sign_consistent_before_fix']} of {scj['sign_total_before_fix']} before the fix and {scj['sign_consistent']} of {scj['sign_total']} after, with {len(scj['features'])} variables. "
      f"Points use PDO {C.PDO:.0f} and {C.BASE_SCORE:.0f} points at {C.BASE_ODDS:.0f}:1 odds (`artifacts/scorecard_points.csv`).\n"
      f"* **Random forest.** {tun['rf']['n_estimators']} trees, depth {tun['rf']['max_depth']}, minimum leaf {tun['rf']['min_samples_leaf']}, fitted on a {tun['rf']['train_rows']:,}-row TRAIN subsample, so it is handicapped against the boosters.\n"
      f"* **XGBoost.** Random search over {len(tun['xgb_search'])} configurations on a 500,000-row TRAIN subsample scored on VALID AUC, then refit on all TRAIN rows. Chosen: `{json.dumps(tun['xgb_best'])}`. Best iteration {tun['xgb_best_iteration']} of at most 800.\n"
      f"* **Monotone XGBoost.** Same parameters, with {tun['xgb_monotone_constrained_features']} of {n_feat} features constrained to the sign of their TRAIN Spearman correlation.\n"
      f"* **LightGBM.** Fixed parameters, no search, best iteration {tun['lightgbm_best_iter']}.\n"
      "* **Class imbalance** was not handled and not tested. No class weights or resampling were used.\n")
    P(f"**Selection rule, frozen before OOT.** `{champ['rule']}`. It is applied in the freeze cell of the notebook and written to `artifacts/champion.json` at {champ['frozen_at']} together with the calibrator choice and the cut-offs. "
      "OOT is scored in the next cell. The monotone variant and the dummy are not candidates.\n")
    vg = pd.DataFrame([{"Model": LABEL[m], "VALID Gini": v} for m, v in sorted(champ["valid_gini"].items(), key=lambda kv: -kv[1])])
    P(md(vg))
    P(f"\nTied set within 0.002 Gini of the best: {champ['tied']}. Selected model: {LABEL[sel]}.\n")

    # ------------------------------------------------------------------ calibration
    P("## 6. Calibration\n")
    P(f"The calibrator for {LABEL[sel]} is chosen on VALID alone. Each candidate is fitted on VALID weeks below {champ['calibrator_split']['fit_weeks_lt']} and scored on the later VALID weeks, and the lowest log loss wins. "
      f"The winner, **{champ['calibrator']}**, is then refitted on all VALID rows and applied to OOT. The table is the OOT result (`artifacts/calibration.json`). ECE is the expected calibration error over deciles.\n")
    crow = [{"Calibration": r["calibration"], "Mean predicted PD": r["mean_pd"], "Observed rate": r["actual_rate"], "Expected / actual": r["expected_over_actual"],
             "ECE": r["ece"], "Brier": r["brier"], "Log loss": r["logloss"], "Chosen on VALID": "yes" if r["chosen_on_valid"] else "no"} for r in cal["oot"]]
    P(md(pd.DataFrame(crow), default="{:.4f}"))
    raw_r, pl_r = cal["oot"][0], next(r for r in cal["oot"] if r["chosen_on_valid"])
    P(f"\nThe raw score over-predicts by a factor of {raw_r['expected_over_actual']:.2f} on OOT, with ECE {raw_r['ece']:.4f}. After {pl_r['calibration']} scaling the factor is {pl_r['expected_over_actual']:.2f} and ECE is {pl_r['ece']:.4f}. "
      "The cause of the raw over-prediction was not traced, and part of it follows from the lower default rate in OOT.\n")
    P("![calibration](artifacts/figures/calibration.png)\n")

    # ------------------------------------------------------------------ credit policy
    P("## 7. Credit policy\n")
    P("Cut-offs are computed on VALID raw PD and applied unchanged to OOT. An application is approved when its PD is below the cut-off. "
      "All rates are observed OOT outcomes. The data hold no cost of review, margin or exposure for the new loan, so no monetary result is estimated. Source: `artifacts/cutoff_strategies.csv`.\n")
    stt = cut.copy()
    for c_ in ("approval_rate", "bad_rate_approved", "bad_rate_declined", "bad_rate_all", "defaulters_declined_share"):
        stt[c_] = stt[c_].map(lambda v: "-" if pd.isna(v) else pct(v))
    stt = stt[["strategy", "model", "approval_rate", "bad_rate_approved", "bad_rate_declined", "defaulters_declined_share"]]
    stt.columns = ["Strategy", "Model", "OOT approval rate", "Bad rate, approved", "Bad rate, declined", "Defaulters declined"]
    P(md(stt))
    x80, l80, g80 = cs80.loc["XGBoost"], cs80.loc["Logistic regression (L2)"], cs80.loc["LightGBM"]
    P(f"\nThe 80% target is the 80% quantile of VALID raw PD, so the OOT approval rate differs: {pct(x80['approval_rate'])} for XGBoost, {pct(l80['approval_rate'])} for logistic regression and {pct(g80['approval_rate'])} for LightGBM. "
      f"The bad rate of approved loans is {pct(x80['bad_rate_approved'])}, {pct(l80['bad_rate_approved'])} and {pct(g80['bad_rate_approved'])}, against {pct(oo['default_rate'])} when everyone is approved. "
      f"Because the approval rates differ, the models are also compared at exactly 80% (`business.json`, `swap_sets`). The bad rate of approved loans is then {pct(sw['portfolio_bad_rate_A'])} for XGBoost and {pct(sw['portfolio_bad_rate_B'])} for logistic regression, "
      f"with {sw['n_approved_each']:,} loans approved by each. The {sw['swap_in_A_only']['n']:,} loans only XGBoost approves have a bad rate of {pct(sw['swap_in_A_only']['bad_rate'])}, and the {sw['swap_out_B_only']['n']:,} loans only logistic regression approves have {pct(sw['swap_out_B_only']['bad_rate'])}.\n")
    P("**Capture of defaulters.** Share of OOT defaulters inside the riskiest 5, 10 and 20% of OOT applications (`artifacts/business.json`, `capture`). The last row is the expected value for a random ordering.\n")
    cp = cap.reset_index()[["model", "caught_at_5pct", "caught_at_10pct", "caught_at_20pct"]]
    cp.columns = ["Model", "Top 5%", "Top 10%", "Top 20%"]
    for c_ in cp.columns[1:]:
        cp[c_] = cp[c_].map(lambda v: pct(v, 1))
    P(md(cp))
    P("\n**Risk grades.** Ten equal-count grades from the VALID score of the selected model. The grade PD is the VALID default rate. Each grade gets an exact two-sided binomial test on OOT, uncorrected for multiple testing (`artifacts/master_scale.csv`).\n")
    mst = ms[["grade", "n_oot", "grade_pd_valid", "oot_default_rate", "binomial_p", "direction"]].copy()
    mst.columns = ["Grade", "OOT apps", "Grade PD (VALID)", "OOT default rate", "Binomial p", "OOT vs grade PD"]
    mst["Grade"] = mst["Grade"].astype(int).astype(str)
    mst["OOT apps"] = mst["OOT apps"].astype(int).map(lambda v: f"{v:,}")
    mst["Binomial p"] = mst["Binomial p"].map(lambda v: "<0.0001" if v < 0.00005 else f"{v:.4f}")
    P(md(mst, default="{:.4f}"))
    P(f"\nThe OOT default rate is monotone across grades: {biz['master_scale_oot_monotone']}. "
      f"The illustrative expected loss uses an assumed LGD of {el['lgd_ASSUMED']} (Basel foundation-IRB value for senior unsecured claims) and the application credit amount as exposure, median-filled when missing. "
      f"It gives an expected total of {el['expected_loss_total']:,.0f} against {el['realised_loss_total']:,.0f} when the same LGD is applied to observed defaults (ratio {el['ratio']:.3f}). This is not a measured loss.\n")
    P("![credit policy](artifacts/figures/credit_policy.png)\n")

    # ------------------------------------------------------------------ stability and drift
    P("## 8. Stability and drift\n")
    P("**Weekly Gini.** The competition stability metric is `mean(weekly Gini) + 88 * min(0, slope) - 0.5 * std(residuals)`, computed over OOT weeks with enough applications (`artifacts/stability.csv`).\n")
    so = stab[stab["split"] == "oot"].set_index("model")
    P(md(pd.DataFrame([{"Model": LABEL[m], "Mean weekly Gini": so.loc[m, "mean_gini"], "Slope per week": so.loc[m, "slope"], "Residual std": so.loc[m, "std_residual"],
                        "Stability metric": so.loc[m, "stability_metric"]} for m in ORDER if m != "dummy"]), {"Slope per week": "{:.4f}", "Residual std": "{:.4f}"}))
    P(f"\n**Score PSI.** TRAIN to VALID {mon['score_psi_train_to_valid']:.3f}, TRAIN to OOT {sd['psi_train_to_oot']:.3f} (band `{sd['psi_band']}`). "
      "The TRAIN baseline is the model's in-sample score. Month by month against the VALID score distribution (`artifacts/monitoring.json`): " +
      ", ".join(f"{r['month']} {r['psi']:.3f}" for r in mon["psi_by_month"]) + ". Bands: below 0.10 stable, 0.10 to 0.25 watch, above 0.25 alert.\n")
    bands = psi_csv["band"].value_counts().to_dict()
    P(f"**Feature PSI.** TRAIN to OOT for the {len(psi_csv)} model features (`artifacts/psi.csv`): " + ", ".join(f"{k} {v}" for k, v in bands.items()) + ".\n")
    P(f"**Adversarial validation** (`adversarial_validation` in `src/hc_stability.py`). A LightGBM classifier is trained to separate TRAIN from OOT applications on {adv['n_train_sample']:,} sampled rows per side with a 70/30 split. "
      f"The held-out AUC is **{adv['auc']:.3f}**. The five strongest separating features by gain share (`artifacts/drift.json`):\n")
    top_model15 = set(top["feature"])
    arows = [{"Feature": f"`{short(r['feature'])}`", "Source": r["feature"].split("__")[0], "Gain share": r["gain_share"],
              "In model top 15 by gain": "yes" if r["feature"] in top_model15 else "no"} for r in adv["top_drifting"][:5]]
    P(md(pd.DataFrame(arows)))
    n_in = sum(1 for r in adv["top_drifting"][:5] if r["feature"] in top_model15)
    P(f"\nHow to read this. The classifier uses no label, so the figure measures how separable the two periods are and is not target leakage. "
      f"An AUC near 1 means OOT applications are easy to tell apart from TRAIN. The separating features are aggregates of history tables (category codes, payment timing and person-type summaries), with `{short(adv['top_drifting'][0]['feature'])}` taking the largest share. "
      f"A likely reason is a change in product mix and in how much history each source holds in later weeks, but this was not tested. {n_in} of the five are among the model's top 15 features by gain, so the model does not rely on them. "
      f"The score PSI of {sd['psi_train_to_oot']:.3f} and the OOT Gini of {mt.loc[sel, 'oot_gini']:.3f} against {mt.loc[sel, 'valid_gini']:.3f} on VALID show no loss of ranking power. "
      "The OOT result is still an estimate for a different population. The residual separators were not removed, because dropping features by their adversarial importance would tune the feature set to the OOT period.\n")
    P("![psi](artifacts/figures/psi.png)\n")
    plan = mon["monitoring_plan"]
    P(f"**Monitoring plan** (written, not run on a live portfolio). Gini trigger: {plan['gini_trigger']['rule']}. Calibration trigger: {plan['calibration_trigger']}. Data trigger: {plan['data_quality_trigger']}. Minimum weekly volume: {plan['minimum_weekly_volume']} applications.\n")
    P("**Most important XGBoost features** by gain share (`artifacts/top_features.csv`):\n")
    tf = top.head(10).copy()
    tf["Feature"] = tf["plain_name"].map(lambda s: s.replace("most recent category of", "highest category code of"))
    tf["Non-null share in OOT"] = tf["feature"].map(lambda f: pct(cov_rows.loc[f, "coverage_oot"], 1) if f in cov_rows.index else "-")
    P(md(tf[["Feature", "gain_share", "Non-null share in OOT"]].rename(columns={"gain_share": "Gain share"}), {"Gain share": "{:.4f}"}))
    P("")

    # ------------------------------------------------------------------ gates
    P("## 9. Quality gates\n")
    P(f"{n_pass} of {n_gates} gates pass (`artifacts/quality_gates.json`). The leakage, coverage, calendar and proxy checks raise before any model is fitted. The remaining gates are evaluated at the end of the run.\n")
    grows = []
    for g in gates["gates"]:
        txt = json.dumps(g["detail"], default=str)
        grows.append({"Gate": g["gate"], "Status": g["status"], "Detail": txt if len(txt) <= 150 else txt[:147] + "..."})
    P(md(pd.DataFrame(grows)))
    P("")

    # ------------------------------------------------------------------ history
    P("## 10. Run history\n")
    P("Four earlier runs are kept for comparison. Each row is a full run on the same split. Sources: `artifacts_v1_partsbug/`, `kaggle/pull_v3_gpu/`, `kaggle/pull_v4_gpu/` and `artifacts/`.\n")
    hist = [history_row("v1", "Table-parts bug: only some part files were read, each part a separate feature block", ROOT / "artifacts_v1_partsbug"),
            history_row("v3", "All part files read and coverage gate added. Dates still used as categorical codes and raw years", ROOT / "kaggle" / "pull_v3_gpu" / "artifacts"),
            history_row("v4", "Dates and year/month columns converted to values relative to the decision date", ROOT / "kaggle" / "pull_v4_gpu" / "artifacts"),
            history_row("v5", "Calendar-proxy screen drops relative-date features with absolute Spearman rho with the week of at least 0.8", art)]
    P(md(pd.DataFrame([h for h in hist if h])))
    v1 = ROOT / "artifacts_v1_partsbug"
    if (v1 / "metrics_table.csv").exists():
        v1m = pd.read_csv(v1 / "metrics_table.csv").set_index("model")
        P("\nOOT AUC, Gini and KS of each model in v1 and v5:\n")
        P(md(pd.DataFrame([{"Model": LABEL[m], "v1 AUC": v1m.loc[m, "oot_auc"], "v5 AUC": mt.loc[m, "oot_auc"], "v1 Gini": v1m.loc[m, "oot_gini"], "v5 Gini": mt.loc[m, "oot_gini"],
                            "v1 KS": v1m.loc[m, "oot_ks"], "v5 KS": mt.loc[m, "oot_ks"]} for m in ORDER if m != "dummy"])))
    P("\nIn v1, `TABLE_SPECS` named individual part files, so the later weeks of several tables, including credit bureau (a), were never read and many features were empty in OOT. "
      "v3 fixed the loader and added the coverage gate, but it kept date strings as categorical codes and raw payment years, which let a model recognise the calendar period. "
      "v4 replaced them with relative values. One bureau refresh date still behaved like a fixed calendar date, because its days before decision tracked the week. The v5 screen removes it with a rule that does not name any feature. "
      "The change from v1 to v5 combines the repaired inputs and the new coverage and proxy rules, and the contribution of each was not separated. "
      "A CPU copy of the v3 run also executed on Kaggle, and its outputs were not used.\n")

    # ------------------------------------------------------------------ reproduce
    P("## 11. Reproduce\n")
    P("1. Accept the competition rules on Kaggle and attach the competition to a notebook, or place the files under `data/raw/parquet_files/train/`. `data_root` in `src/hc_config.py` also scans `/kaggle/input`.\n"
      "2. Build the notebook with `python build_notebook.py`. All `src/` modules are inlined into `notebooks/HomeCreditStability.ipynb`. `python build_notebook.py --smoke 60000` builds a sample run.\n"
      "3. Smoke test on a sample of cases, no artifacts written: `python scripts/smoke.py 40000`.\n"
      f"4. Push to Kaggle with `python scripts/push_kernel.py --gpu --message \"full run\"`. The kernel is `{KERNEL}` with the competition attached and internet on. `scripts/poll_kernel.py` polls a kernel, and its slug is set at the top of the file.\n"
      "5. Coverage audit: `python scripts/check_table_parts.py`. README: `python build_readme.py`. Tests: `pip install -r requirements-dev.txt`, then `python -m pytest -q tests`. App: `pip install -r requirements.txt`, then `streamlit run app.py`.\n")
    P(f"**Hardware.** The run of record used a Kaggle GPU session. {', '.join(gpu_models)} ran on CUDA (`device=\"cuda\"`, `tree_method=\"hist\"`, same parameters as on CPU) and {', '.join(cpu_models)} ran on CPU (`artifacts/environment.json`, `devices`). "
      f"Runtime was {env['runtime_seconds'] / 60:.0f} minutes. The code falls back to CPU when no GPU is visible. GPU and CPU tree builds can differ slightly, so a CPU rerun may not reproduce the digits exactly.\n")
    P(f"Environment: python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, polars {env['polars']}, scikit-learn {env['sklearn']}, statsmodels {env['statsmodels']}, optbinning {env['optbinning']}, xgboost {env['xgboost']}, lightgbm {env['lightgbm']}. Seed 42 throughout.\n")
    P("The Streamlit app (`app.py`) reads `artifacts/` only and never trains. It has seven pages: Overview, Model comparison, Stability & monitoring, Score a case, Batch scoring, Credit policy and Model card. `tests/test_app.py` drives every page on a synthetic artifact bundle, so the suite needs no Kaggle outputs.\n")

    # ------------------------------------------------------------------ layout
    P("## 12. Repository layout\n")
    P("```\n"
      "app.py                      Streamlit app: Credit Risk Modelling | Loan Default Prediction with Out-of-Time Validation\n"
      "requirements.txt            app runtime only (xgboost-cpu, streamlit, altair)\n"
      "requirements-dev.txt        test suite (full xgboost, polars, optbinning, statsmodels, pytest)\n"
      ".github/workflows/ci.yml    pytest and a seven-page Streamlit smoke test\n"
      "build_notebook.py           builds the Kaggle notebook from src/\n"
      "build_readme.py             renders this README from artifacts/\n"
      "src/hc_config.py            constants: split, thresholds, scaling, LGD assumption, table list\n"
      "src/hc_data.py             part-file union, PIT filter, relative dates, screening, aggregation, model matrix\n"
      "src/hc_features.py          IV, VIF, correlation pruning, linear pipeline\n"
      "src/hc_model.py             tree factories, WoE binning, scorecard, sign check, points table\n"
      "src/hc_ladder.py            tuning, selection rule, bootstrap, calibration, policy layer, grades\n"
      "src/hc_metrics.py           AUC, Gini, KS, PSI, stability metric\n"
      "src/hc_stability.py         adversarial validation, PSI tables, monitoring plan\n"
      "src/hc_pipeline.py          dataset build, leakage, coverage, calendar and proxy gates\n"
      "scripts/                    Kaggle push and poll, smoke test, provenance, coverage audit\n"
      "tests/                      pytest suites\n"
      "notebooks/                  generated and executed notebooks\n"
      "artifacts/                  every number in this README (figures/, models/)\n"
      "artifacts_v1_partsbug/      artifacts of the v1 run, kept for comparison\n"
      "kaggle/pull_v3_gpu, pull_v4_gpu, pull_v5_gpu   raw pulls of the GPU kernel versions\n"
      "data/PROVENANCE.md          source, retrieval and hashes (data/raw is not committed)\n"
      "CONTRACT.md                 build contract\n"
      "```\n")

    # ------------------------------------------------------------------ limitations
    P("## 13. Limitations\n")
    P(f"* **Class imbalance** was not handled and not tested. The TRAIN default rate is {pct(tr['default_rate'])}, and all models are fitted on the raw data.\n"
      "* **One split, one seed, one run.** Variation across other week cut-offs and seeds was not measured. The cross-validation code in `src/hc_pipeline.py` was not run for these results.\n"
      f"* **Drift.** The adversarial AUC is {adv['auc']:.3f}. OOT metrics describe a population that differs from TRAIN in ways that were not decomposed.\n"
      f"* **Tuning is small.** The logistic C grid has {len(tun['logistic_C_grid'])} points and the XGBoost search {len(tun['xgb_search'])} configurations on a 500,000-row subsample. LightGBM and the random forest have no search, and the random forest uses {tun['rf']['train_rows']:,} of {int(tr['n']):,} TRAIN rows.\n"
      "* **VALID is used for several decisions** (early stopping, XGBoost search, C, calibrator, cut-offs, grades), so VALID metrics are not independent evidence.\n"
      "* **Coverage rule.** Sources that do not cover the end of the training period, such as tax registry b and c, cannot contribute. Their value for later weeks was not tested.\n"
      "* **Point-in-time rule** drops whole records with any date after the decision. Its effect on accuracy was not measured.\n"
      "* **Bootstrap** resamples applications independently and ignores week-level dependence. The risk-grade binomial tests are not corrected for multiple testing.\n"
      "* **Scorecard.** The gap to logistic regression mixes variable count, binning, VIF and sign pruning. No ablation was run.\n"
      "* **Categorical encoding.** Trees receive integer codes as numbers. The largest category code is an arbitrary ordinal summary and not a most-recent value.\n"
      "* **Cut-off tables** cover logistic regression, XGBoost and LightGBM. The scorecard cut-offs were frozen but not evaluated on OOT.\n"
      "* **No fairness or subgroup analysis** was run. SHAP values were not computed, and the importances shown are XGBoost gain shares.\n"
      "* **Monetary results** are not measured. The expected-loss table depends on an assumed LGD and on the application amount as exposure.\n"
      "* **Leaderboard.** The Kaggle metric uses a different, hidden period and protocol, so these Gini values are not comparable with leaderboard scores.\n"
      "* **Time scope.** The OOT window is the last 14 weeks of the labelled train file.\n")

    # ------------------------------------------------------------------ appendix
    P("## Appendix: figures quoted elsewhere\n")
    P("Values copied from the artifacts named in the last column, for checking other text against this run.\n")
    q = []
    for m, lab in (("penalised_lr", "Logistic regression"), ("random_forest", "Random forest"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
        q.append({"Statement": f"{lab}: OOT AUC / Gini / KS", "Value": f"{mt.loc[m, 'oot_auc']:.3f} / {mt.loc[m, 'oot_gini']:.3f} / {mt.loc[m, 'oot_ks']:.3f}", "Source": "`metrics_table.csv`"})
    q.append({"Statement": "Final features and source tables", "Value": f"{n_feat} features: " + ", ".join(f"{k.replace('train_', '')} {v}" for k, v in sorted(fcount.items())), "Source": "`feature_meta.json`"})
    q.append({"Statement": "Applications: train / valid / OOT", "Value": f"{int(tr['n']):,} / {int(va['n']):,} / {int(oo['n']):,}", "Source": "`split.json`"})
    q.append({"Statement": "Date ranges: train / OOT", "Value": f"{tr['date_min']} to {tr['date_max']} / {oo['date_min']} to {oo['date_max']}", "Source": "`split.json`"})
    q.append({"Statement": "80% target, OOT approval rate: XGBoost / logistic regression / LightGBM", "Value": f"{pct(x80['approval_rate'])} / {pct(l80['approval_rate'])} / {pct(g80['approval_rate'])}", "Source": "`cutoff_strategies.csv`"})
    q.append({"Statement": "Bad rate of approved loans: approve all / XGBoost / logistic regression / LightGBM", "Value": f"{pct(oo['default_rate'])} / {pct(x80['bad_rate_approved'])} / {pct(l80['bad_rate_approved'])} / {pct(g80['bad_rate_approved'])}", "Source": "`cutoff_strategies.csv`"})
    q.append({"Statement": "Defaulters in the riskiest 10%: logistic regression / XGBoost / LightGBM", "Value": f"{pct(cap.loc['Logistic regression (L2)', 'caught_at_10pct'], 1)} / {pct(cap.loc['XGBoost', 'caught_at_10pct'], 1)} / {pct(cap.loc['LightGBM', 'caught_at_10pct'], 1)}", "Source": "`business.json`"})
    P(md(pd.DataFrame(q)))
    P("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--out", type=Path, default=ROOT / "README.md")
    args = ap.parse_args()
    args.out.write_text(render(args.artifacts), encoding="utf-8", newline="\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
