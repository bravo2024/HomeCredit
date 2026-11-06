"""Orchestration: dataset, screening, roster fitting, stability, decision layer.

The notebook is thin on purpose - it calls these functions and interprets the
artifacts.  Anything that could change a reported number lives here or in the
modules this file imports, so it can be unit-tested without a notebook kernel.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hc_config import (
    CV_ROW_CAP,
    ID_COL,
    MAX_LINEAR_FEATURES,
    MAX_TREE_FEATURES,
    MAX_WOE_FEATURES,
    SEED,
    SPLIT_FRACTIONS,
    TARGET,
    UNIVARIATE_AUC_SCREEN,
    VIF_MAX,
    WEEK_COL,
    WOE_FIT_SAMPLE,
    DATE_COL,
    IV_MIN,
    MIN_RECENT_COVERAGE,
    RECENT_TRAIN_WEEKS,
    CALENDAR_PROXY_RHO,
)
from hc_data import (
    is_absolute_calendar_feature,
    build_diagnostics,
    build_feature_frame,
    feature_columns,
    read_base,
    split_bounds,
    to_model_matrix,
    univariate_screen,
)
from hc_features import drop_high_correlation, iv_table, vif_prune, quantile_woe_iv
from hc_metrics import (
    auc,
    psi,
    psi_band,
    brier,
    gini,
    logloss,
    pr_auc,
    stability_metric,
    weekly_gini,
)
from hc_model import (
    ConstantModel,
    sanitize_design,
    IsotonicCalibrator,
    cap_rows,
    expanding_folds,
    fit_penalised_logistic,
    fit_scorecard,
    fit_single_penalised,
    fit_tree,
    fit_woe_binning,
    points_table,
    sign_check,
    woe_design,
)
from hc_stability import adversarial_validation, csi_table, psi_table, score_drift


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def build_dataset(
    sample_cases: int | None = None,
    max_tables: int | None = None,
    verbose: bool = True,
) -> dict:
    """Base table + every history table aggregated to one row per application.

    Returns a dict with the polars frame, the time split, the model matrix and
    the audit trail (PIT drop counts, screening tables).
    """
    t0 = time.time()
    base = read_base(sample_cases=sample_cases)
    weeks = sorted(base[WEEK_COL].unique().to_list())
    bounds = split_bounds(weeks, SPLIT_FRACTIONS)

    feats, audit = build_feature_frame(base, max_tables=max_tables, verbose=verbose)
    diagnostics = build_diagnostics(base)
    cols = feature_columns(feats)

    week = feats[WEEK_COL].to_numpy()
    y = feats[TARGET].to_numpy().astype(int)
    dates = feats[DATE_COL].to_numpy()
    case_ids = feats[ID_COL].to_numpy()
    split = np.where(
        week <= bounds["train_weeks"][-1],
        "train",
        np.where(week <= bounds["valid_weeks"][-1], "valid", "oot"),
    )
    train_mask = split == "train"

    X, enc = to_model_matrix(feats, cols, train_mask)
    cat_cols = list(enc["categorical_vocab_sizes"].keys())
    # The model matrix is the only copy of the features the pipeline needs after
    # this point; dropping the polars frame keeps the peak footprint inside the
    # CPU-kernel memory budget on the full run.
    feats = None if sample_cases is None else feats

    return {
        "base": base,
        "feats": feats,
        "X": X,
        "y": y,
        "case_ids": case_ids,
        "week": week,
        "date": dates,
        "split": split,
        "train_mask": train_mask,
        "valid_mask": split == "valid",
        "oot_mask": split == "oot",
        "bounds": bounds,
        "audit": audit,
        "diagnostics": diagnostics,
        "feature_columns": cols,
        "cat_cols": cat_cols,
        "encoding": enc,
        "elapsed_s": time.time() - t0,
    }


def split_profile(ds: dict) -> dict:
    """Row count, default rate, week range and date range per split."""
    out = {}
    for name in ("train", "valid", "oot"):
        mask = ds["split"] == name
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        dates = pd.to_datetime(pd.Series(ds["date"][mask]))
        out[name] = {
            "n": int(mask.sum()),
            "default_rate": float(ds["y"][mask].mean()),
            "weeks": [int(ds["week"][mask].min()), int(ds["week"][mask].max())],
            "n_weeks": int(len(np.unique(ds["week"][mask]))),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
        }
    total = sum(v.get("n", 0) for v in out.values())
    for v in out.values():
        v["share"] = float(v.get("n", 0) / total) if total else 0.0
    return out


# --------------------------------------------------------------------------
# Feature screening
# --------------------------------------------------------------------------
def feature_screen(ds: dict) -> dict:
    """Rank candidate features by IV on training rows and select the designs.

    Selection rules:
    * IV is computed on a training-row sample (supervised screen -> train only);
    * features with a univariate AUC above the leakage threshold are quarantined
      before selection, and the quarantine list is written to the artifact;
    * the tree design keeps the top ``MAX_TREE_FEATURES`` features by IV;
    * the linear/scorecard design keeps the top numeric features by IV.
    """
    X = ds["X"]
    y = ds["y"]
    train_mask = ds["train_mask"]
    rng = np.random.default_rng(SEED)
    n_train = int(train_mask.sum())
    take = min(n_train, 300_000)
    idx = np.sort(rng.choice(np.where(train_mask)[0], take, replace=False))

    Xs = X.iloc[idx]
    ys = y[idx]
    screen = univariate_screen(Xs, ys, UNIVARIATE_AUC_SCREEN)
    quarantined = {r["feature"] for r in screen["flagged"]}

    # Coverage rule (v2): TRAIN rows only, last RECENT_TRAIN_WEEKS train weeks. A feature that is
    # (almost) empty at the end of training cannot be learned from or applied later.
    recent_weeks = sorted(np.unique(ds["week"][train_mask]))[-RECENT_TRAIN_WEEKS:]
    recent_mask = train_mask & np.isin(ds["week"], recent_weeks)
    recent_cov = X.loc[recent_mask].notna().mean() if recent_mask.any() else X.notna().mean()
    low_recent = sorted(c for c in X.columns if recent_cov[c] < MIN_RECENT_COVERAGE)

    # Calendar-proxy screen (v5, TRAIN rows only, before selection): drop any relative-date feature
    # that tracks WEEK_NUM almost monotonically. No feature is hand-picked.
    proxy = calendar_proxy_screen(Xs, ds["week"][idx])
    proxy_dropped = set(proxy.loc[proxy["dropped"], "feature"])

    candidates = [c for c in X.columns if c not in quarantined and c not in set(low_recent) and c not in proxy_dropped]
    iv = iv_table(Xs[candidates], ys, n_bins=10)
    iv = iv[iv["iv"] >= IV_MIN]
    ordered = [f for f in iv["feature"].tolist() if f in X.columns]

    tree_features = ordered[:MAX_TREE_FEATURES]
    numeric_iv = [f for f in ordered if f not in ds["cat_cols"]]
    linear_numeric = numeric_iv[:MAX_LINEAR_FEATURES]
    woe_features = [f for f in numeric_iv[:MAX_WOE_FEATURES]]

    return {
        "iv_table": iv,
        "quarantined": screen["flagged"],
        "leakage_screen": screen,
        "tree_features": tree_features,
        "linear_numeric": linear_numeric,
        "woe_features": woe_features,
        "n_candidates": len(candidates),
        "low_recent_coverage": low_recent,
        "calendar_proxy": proxy,
        "recent_train_weeks": [int(w) for w in recent_weeks],
    }


def build_linear_design(
    ds: dict,
    numeric_features: list[str],
    max_levels: int = 8,
    max_onehot_cols: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """Standardised design: selected numeric features + one-hot low-cardinality cats.

    Only categorical columns with at most ``max_levels`` training levels are
    one-hot encoded.  A 4000-level district column one-hot would add noise and
    hundreds of columns; those levels are left to the tree models, which can use
    the raw codes.  The level set is taken from training rows only.
    """
    X = ds["X"]
    train_mask = ds["train_mask"]
    parts = [X[numeric_features].copy()]
    used_cats = []
    onehot_budget = max_onehot_cols
    for c in ds["cat_cols"]:
        if onehot_budget <= 0:
            break
        train_values = X.loc[train_mask, c]
        levels = sorted(train_values.dropna().unique().tolist())
        if not levels or len(levels) > max_levels:
            continue
        used_cats.append(c)
        onehot_budget -= max(len(levels) - 1, 1)
        # Drop one level as the reference: keeping every level makes the dummies
        # sum to one and the design exactly collinear with the intercept, which
        # leaves the coefficient vector unidentified and makes saga crawl to the
        # iteration cap.
        for lvl in levels[:-1]:
            parts.append(
                pd.Series(
                    (X[c] == lvl).astype("float32").to_numpy(),
                    index=X.index,
                    name=f"{c}__is_{int(lvl)}" if float(lvl).is_integer() else f"{c}__is_{lvl}",
                )
            )
        if train_values.isna().any():
            parts.append(
                pd.Series(
                    X[c].isna().astype("float32").to_numpy(),
                    index=X.index,
                    name=f"{c}__is_missing",
                )
            )
    design = pd.concat(parts, axis=1)
    keep, _dropped = drop_high_correlation(design.loc[train_mask])
    return design[keep], used_cats


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------
def build_scorecard(
    ds: dict,
    woe_features: list[str],
    train_idx: np.ndarray | None = None,
) -> dict:
    """Fit monotone WoE bins, screen by IV/VIF and estimate the Logit.

    Fitting order matters and is enforced here: bins on a *training* sample,
    then IV filter, then VIF pruning on the training WoE design, then the
    statsmodels fit.  VALID is only ever transformed, never fitted.

    ``train_idx`` lets the cross-validation reuse this exact routine on a fold's
    training rows, so the scorecard in the CV panel is the same estimator as the
    final one rather than a simplified stand-in.
    """
    X = ds["X"]
    y = ds["y"]
    train_mask = ds["train_mask"]
    rng = np.random.default_rng(SEED)
    if train_idx is None:
        train_idx = np.where(train_mask)[0]
    take = train_idx[np.sort(rng.choice(len(train_idx), min(len(train_idx), WOE_FIT_SAMPLE), replace=False))]

    Xtr_fit = X.iloc[take][woe_features]
    ytr_fit = y[take]

    # optbinning cannot pre-bin a variable with no observed values, and a fold
    # that is short on history will contain such variables. Require a minimum
    # observed share in the fitting sample instead of letting the run die on a
    # single empty column.
    coverage = Xtr_fit.notna().mean()
    usable = [c for c in woe_features if coverage.get(c, 0.0) >= 0.01]
    if len(usable) < 5:
        usable = [c for c in woe_features if coverage.get(c, 0.0) > 0.0]
    if not usable:
        raise ValueError("no scorecard variable has observed values in the fitting sample")
    Xtr_fit = Xtr_fit[usable]
    woe_features = usable

    process = fit_woe_binning(Xtr_fit, ytr_fit, categorical=[])
    summary = process.summary()
    summary = summary.rename(columns={c: c.lower().replace(" ", "_") for c in summary.columns})
    iv_series = dict(zip(summary["name"], summary["iv"]))
    kept = [f for f in woe_features if iv_series.get(f, 0.0) >= IV_MIN]

    # The binning object was fitted on every candidate variable, so it must be
    # applied to the full candidate frame and only then subset by IV - passing a
    # subset to ``transform`` is rejected by optbinning.
    woe_all = woe_design(process, X.iloc[train_idx][woe_features])
    woe_train = woe_all[[c for c in kept if c in woe_all.columns]]
    pruned, vif_log = vif_prune(woe_train, threshold=VIF_MAX)
    pruned = sanitize_design(woe_train[pruned])
    woe_train = woe_train[pruned]

    sc = fit_scorecard(
        woe_train,
        y[train_idx],
        woe_design(process, X.iloc[train_idx[:1]][woe_features], columns=pruned),
    )
    coef = sc["coef_table"]
    signs = sign_check(coef, process, X.iloc[train_idx[:5000]][pruned], y[train_idx[:5000]])
    pts, base_points = points_table(process, coef, pdo=20.0, base_score=600.0, base_odds=20.0)

    return {
        "process": process,
        "process_vars": list(woe_features),
        "features": pruned,
        "vif_log": vif_log,
        "coef_table": coef,
        "sign_check": signs,
        "summary": summary,
        "points": pts,
        "base_points": base_points,
        "woe_train": woe_train,
        "result": sc["result"],
        "train_rows": int(len(train_idx)),
        "woe_fit_rows": int(len(take)),
        "woe_fit_fraction": float(len(take) / max(1, len(train_idx))),
    }


def scorecard_predict(sc: dict, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted WoE design + Logit coefficients to raw columns."""
    design = woe_design(sc["process"], X[sc["process_vars"]], columns=sc["features"])
    return np.asarray(sc["result"].predict(np.c_[np.ones(len(design)), design.to_numpy(dtype=float)]))


# --------------------------------------------------------------------------
# Coverage gate (v2)
# --------------------------------------------------------------------------
def feature_coverage(ds: dict, selected: dict[str, list[str]]) -> dict:
    """Non-null share of every model-matrix column in TRAIN, VALID and OOT.

    ``selected`` maps a design name (tree / linear / woe) to its feature list; the
    gate applies to the union of the final model features.  Selection itself uses only
    TRAIN rows (see ``feature_screen``); VALID and OOT are measured here for audit.
    """
    X = ds["X"]
    cov = {}
    for name in ("train", "valid", "oot"):
        m = ds["split"] == name
        cov[name] = X.loc[m].notna().mean() if m.any() else pd.Series(np.nan, index=X.columns)
    final = {f for fs in selected.values() for f in fs}
    rows = []
    for c in X.columns:
        rows.append({
            "feature": c,
            "source": c.split("__")[0] if "__" in c else "base",
            "coverage_train": float(cov["train"][c]),
            "coverage_valid": float(cov["valid"][c]),
            "coverage_oot": float(cov["oot"][c]),
            "final_model_feature": c in final,
        })
    fin = [r for r in rows if r["final_model_feature"]]
    null_oot = [r["feature"] for r in fin if r["coverage_oot"] == 0.0]
    null_valid = [r["feature"] for r in fin if r["coverage_valid"] == 0.0]
    by_source: dict[str, dict] = {}
    for r in fin:
        s = by_source.setdefault(r["source"], {"n_features": 0, "_tr": 0.0, "_va": 0.0, "_oo": 0.0})
        s["n_features"] += 1
        s["_tr"] += r["coverage_train"]
        s["_va"] += r["coverage_valid"]
        s["_oo"] += r["coverage_oot"]
    for s in by_source.values():
        n = s["n_features"]
        s["mean_coverage_train"] = s.pop("_tr") / n
        s["mean_coverage_valid"] = s.pop("_va") / n
        s["mean_coverage_oot"] = s.pop("_oo") / n
    return {
        "n_matrix_columns": len(rows),
        "n_final_features": len(fin),
        "n_null_in_oot": len(null_oot),
        "n_null_in_valid": len(null_valid),
        "null_in_oot": null_oot,
        "null_in_valid": null_valid,
        "passed": not null_oot and not null_valid,
        "rule": "FAIL if any final model feature has zero non-null rows in VALID or OOT",
        "by_source": by_source,
        "features": rows,
    }


def calendar_proxy_screen(Xs: pd.DataFrame, weeks: np.ndarray, threshold: float = CALENDAR_PROXY_RHO) -> pd.DataFrame:
    """Spearman correlation of every ``__rel_*`` feature with WEEK_NUM on a TRAIN sample."""
    w = pd.Series(np.asarray(weeks, dtype=float), index=Xs.index)
    rows = []
    for c in Xs.columns:
        if "__rel_" not in c:
            continue
        x = Xs[c]
        ok = x.notna()
        rho = float("nan")
        if ok.sum() >= 1000 and x[ok].nunique() > 1:
            rho = float(x[ok].corr(w[ok], method="spearman"))
        rows.append({"feature": c, "rho": rho, "coverage": float(ok.mean()),
                     "dropped": bool(np.isfinite(rho) and abs(rho) >= threshold)})
    return pd.DataFrame(rows, columns=["feature", "rho", "coverage", "dropped"])


def calendar_proxy_gate(proxy: pd.DataFrame, features: list[str], threshold: float = CALENDAR_PROXY_RHO,
                        raise_on_fail: bool = False) -> dict:
    """FAIL if a final feature is a relative-date feature with |rho| >= threshold."""
    rho = dict(zip(proxy["feature"], proxy["rho"])) if len(proxy) else {}
    bad = sorted(f for f in features if f in rho and np.isfinite(rho[f]) and abs(rho[f]) >= threshold)
    row = {"gate": "no_calendar_proxy_features", "status": "PASS" if not bad else "FAIL",
           "detail": {"threshold": threshold, "n_screened": int(len(proxy)), "n_dropped": int(proxy["dropped"].sum()) if len(proxy) else 0,
                      "offending_final_features": bad[:20]}}
    if bad and raise_on_fail:
        raise RuntimeError(f"CALENDAR PROXY GATE FAILED: {bad[:10]}")
    return row


def calendar_gate(features: list[str], raise_on_fail: bool = False) -> dict:
    """FAIL if a final feature is built from an absolute calendar value.

    Date columns (suffix D) and year/month columns (pmts_year_*, pmts_month_*) may only enter
    as values relative to ``date_decision`` (feature names containing ``__rel_``).
    """
    bad = sorted(f for f in features if is_absolute_calendar_feature(f))
    row = {"gate": "no_absolute_calendar_features", "status": "PASS" if not bad else "FAIL",
           "detail": {"n_features_checked": len(features), "absolute_calendar_features": bad[:20],
                      "n_relative_features": sum(1 for f in features if "__rel_" in f)}}
    if bad and raise_on_fail:
        raise RuntimeError(f"CALENDAR GATE FAILED: {bad[:10]}")
    return row


def coverage_gate(cov: dict, raise_on_fail: bool = False) -> dict:
    """Quality-gate row for ``quality_gates.json``; optionally raises on failure."""
    row = {
        "gate": "coverage_no_final_feature_empty_in_valid_or_oot",
        "status": "PASS" if cov["passed"] else "FAIL",
        "detail": {"n_final_features": cov["n_final_features"], "null_in_oot": cov["null_in_oot"][:20],
                   "null_in_valid": cov["null_in_valid"][:20]},
    }
    if raise_on_fail and not cov["passed"]:
        raise RuntimeError(f"COVERAGE GATE FAILED: {row['detail']}")
    return row


# --------------------------------------------------------------------------
# Model roster
# --------------------------------------------------------------------------
def catboost_cat_indices(columns: list[str], cat_cols: list[str]) -> list[int]:
    """Column positions of the native categorical features for CatBoost."""
    return [i for i, c in enumerate(columns) if c in cat_cols]


def fit_roster(
    ds: dict,
    screen: dict,
    linear_design: pd.DataFrame,
    sc: dict | None,
    lr_config: dict,
    include_trees: bool = True,
) -> dict:
    """Fit every model on the training window and predict VALID + OOT.

    All models see the identical split; the scorecard sees the WoE design, the
    linear models the standardised design, the trees the raw matrix with native
    categoricals where the library supports them.
    """
    X = ds["X"]
    y = ds["y"]
    train_mask = ds["train_mask"]
    valid_mask = ds["valid_mask"]
    oot_mask = ds["oot_mask"]

    tree_features = [f for f in screen["tree_features"] if f in X.columns]
    cat_idx_tree = catboost_cat_indices(tree_features, ds["cat_cols"])
    X_tree = X[tree_features]

    out: dict[str, dict] = {}

    # --- baseline ---------------------------------------------------------
    const = ConstantModel().fit(X_tree[train_mask], y[train_mask])
    out["dummy"] = {
        "p_valid": const.predict_proba(X_tree[valid_mask]),
        "p_oot": const.predict_proba(X_tree[oot_mask]),
        "meta": {"family": "baseline", "features": 0, "train_rate": const.rate},
    }

    # --- scorecard --------------------------------------------------------
    if sc is not None:
        out["woe_scorecard"] = {
            "p_valid": scorecard_predict(sc, X[valid_mask]),
            "p_oot": scorecard_predict(sc, X[oot_mask]),
            "meta": {"family": "scorecard", "features": len(sc["features"])},
        }

    # --- penalised logistic ----------------------------------------------
    from hc_features import linear_pipeline

    pipe = linear_pipeline()
    Ztr = pipe.fit_transform(linear_design[train_mask], y[train_mask])
    Zva = pipe.transform(linear_design[valid_mask])
    Zoo = pipe.transform(linear_design[oot_mask])

    lr_fit = fit_single_penalised(Ztr, y[train_mask], lr_config)
    out["penalised_lr"] = {
        "p_valid": lr_fit["model"].predict_proba(Zva)[:, 1],
        "p_oot": lr_fit["model"].predict_proba(Zoo)[:, 1],
        "model": lr_fit["model"],
        "meta": {
            "family": "linear",
            "features": int(linear_design.shape[1]),
            "config": lr_fit["config"],
            "converged": lr_fit["converged"],
            "n_iter": lr_fit["n_iter"],
            "n_nonzero": lr_fit["n_nonzero"],
        },
    }

    # --- trees -------------------------------------------------------------
    if include_trees:
        for kind, name in (("rf", "random_forest"), ("xgb", "xgboost"), ("lgbm", "lightgbm"), ("cat", "catboost")):
            try:
                fit = fit_tree(
                    kind,
                    X_tree[train_mask],
                    y[train_mask],
                    X_tree[valid_mask],
                    y[valid_mask],
                    cat_idx=cat_idx_tree if kind == "cat" else None,
                )
            except Exception as exc:  # noqa: BLE001 - a library failure must not
                # destroy a multi-hour run; it is recorded and the gate reports the
                # missing family instead of hiding it behind a crashed kernel.
                print(f"  {name} FAILED: {type(exc).__name__}: {exc}")
                out[name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            model = fit["model"]
            pva = model.predict_proba(
                _cat_matrix(X_tree[valid_mask], cat_idx_tree) if kind == "cat" else X_tree[valid_mask]
            )[:, 1]
            poo = model.predict_proba(
                _cat_matrix(X_tree[oot_mask], cat_idx_tree) if kind == "cat" else X_tree[oot_mask]
            )[:, 1]
            out[name] = {
                "p_valid": pva,
                "p_oot": poo,
                "meta": {"family": "tree", "features": len(tree_features), "best_iter": fit["best_iter"]},
                "model": model,
            }
    return out


def _cat_matrix(X: pd.DataFrame, cat_idx: list[int]) -> pd.DataFrame:
    """CatBoost copy with integer-string categorical columns."""
    out = X.copy()
    for j in cat_idx or []:
        col = out.columns[j]
        out[col] = out[col].fillna(-2).astype("int64").astype(str)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def weekly_min_cases(n_rows: int, n_weeks: int, cap: int = 150) -> int:
    """Minimum weekly volume for a weekly Gini to be treated as informative.

    A fixed threshold would either drop every week on a small sample or let a
    30-application week drive the slope penalty on a big one.  Using a third of
    the median weekly volume (capped at the monitoring minimum) keeps the
    stability fit meaningful at both scales, and the number used is reported.
    """
    if n_weeks <= 0:
        return 20
    median = n_rows / n_weeks
    return int(max(20, min(cap, median / 3)))


def evaluate_model(
    name: str,
    p_valid: np.ndarray,
    p_oot: np.ndarray,
    ds: dict,
    min_cases: int | None = None,
) -> dict:
    """Full metric block for one model on VALID and OOT.

    Calibration is fitted on VALID and applied to OOT; the *uncalibrated*
    probabilities are also reported because a scorecard is used through its
    points table, not through a softmax.
    """
    y, week, split = ds["y"], ds["week"], ds["split"]
    valid_mask = split == "valid"
    oot_mask = split == "oot"

    calibrator = IsotonicCalibrator().fit(p_valid, y[valid_mask])
    p_oot_cal = calibrator.predict(p_oot)
    p_valid_cal = calibrator.predict(p_valid)

    mv = weekly_min_cases(int(valid_mask.sum()), len(np.unique(week[valid_mask])))
    mo = weekly_min_cases(int(oot_mask.sum()), len(np.unique(week[oot_mask])))
    if min_cases is not None:
        mv = mo = min_cases
    wv = weekly_gini(y[valid_mask], p_valid, week[valid_mask], min_cases=mv)
    wo = weekly_gini(y[oot_mask], p_oot, week[oot_mask], min_cases=mo)

    from hc_metrics import calibration_summary, topk_metrics, ks_stat

    return {
        "model": name,
        "valid": {
            "auc": auc(y[valid_mask], p_valid),
            "gini": gini(y[valid_mask], p_valid),
            "pr_auc": pr_auc(y[valid_mask], p_valid),
            "brier": brier(y[valid_mask], p_valid_cal),
            "logloss": logloss(y[valid_mask], p_valid_cal),
            "ks": ks_stat(y[valid_mask], p_valid),
            "stability": stability_metric(wv),
            "top5": topk_metrics(y[valid_mask], p_valid, 0.05),
            "top10": topk_metrics(y[valid_mask], p_valid, 0.10),
        },
        "oot": {
            "auc": auc(y[oot_mask], p_oot),
            "gini": gini(y[oot_mask], p_oot),
            "pr_auc": pr_auc(y[oot_mask], p_oot),
            "brier": brier(y[oot_mask], p_oot_cal),
            "logloss": logloss(y[oot_mask], p_oot_cal),
            "ks": ks_stat(y[oot_mask], p_oot),
            "stability": stability_metric(wo),
            "calibration": calibration_summary(y[oot_mask], p_oot_cal),
            "top5": topk_metrics(y[oot_mask], p_oot, 0.05),
            "top10": topk_metrics(y[oot_mask], p_oot, 0.10),
        },
        "weekly": {"valid": wv.to_dict(orient="records"), "oot": wo.to_dict(orient="records")},
        "weekly_min_cases": {"valid": mv, "oot": mo},
    }


def choose_champion(results: dict, metric_key: str = "valid") -> str:
    """Champion = highest competition stability metric on the VALID weeks.

    Deliberately *not* pooled AUC: a model that wins AUC by being unstable loses
    money in the weeks after development, which is the whole premise of the
    competition's metric.
    """
    scored = {
        name: r[metric_key]["stability"]["stability_metric"]
        for name, r in results.items()
        if name != "dummy" and np.isfinite(r[metric_key]["stability"]["stability_metric"])
    }
    if not scored:
        # Degenerate sample (too few rows per week): fall back to pooled AUC and
        # say so, rather than crashing or silently picking the first model.
        return max(
            (n for n in results if n != "dummy"),
            key=lambda n: results[n][metric_key]["auc"],
        )
    return max(scored, key=scored.get)


# --------------------------------------------------------------------------
# Cross-validation inside the training window
# --------------------------------------------------------------------------
def cv_roster(
    ds: dict,
    screen: dict,
    linear_design: pd.DataFrame,
    n_folds: int = 4,
    row_cap: int = 300_000,
    lr_cap: int = 200_000,
    include_trees: bool = True,
    verbose: bool = True,
) -> dict:
    """Expanding-window CV over the training weeks for the whole roster.

    The folds never touch VALID or OOT weeks.  Each fold refits every step -
    WoE bins, imputer, scaler, penalty path, trees - on its own training rows,
    so the CV number is an honest estimate of the refit procedure rather than a
    score of a model that has already seen the fold.

    Row caps are the documented compute budget (7 families x 4 folds has to fit
    in one kernel); the cap applies to the *fit* rows only, never to the rows
    the fold is scored on.
    """
    from hc_features import linear_pipeline
    from hc_model import PENALTY_GRID

    X, y, week = ds["X"], ds["y"], ds["week"]
    train_pos = np.where(ds["train_mask"])[0]
    sub = pd.DataFrame({"week": week[train_pos]})
    folds = expanding_folds(sub, n_folds)

    tree_features = [f for f in screen["tree_features"] if f in X.columns]
    cat_idx = catboost_cat_indices(tree_features, ds["cat_cols"])
    Xt = X[tree_features]

    records: list[dict] = []
    lr_records: list[dict] = []
    fold_meta = []

    for f in folds:
        t0 = time.time()
        tr = train_pos[f["train_idx"]]
        va = train_pos[f["valid_idx"]]
        tr_cap = cap_rows(tr, row_cap)
        fold_meta.append(
            {
                "fold": f["fold"],
                "train_weeks": f["train_weeks"],
                "valid_weeks": f["valid_weeks"],
                "n_train_full": int(len(tr)),
                "n_train_used": int(len(tr_cap)),
                "n_valid": int(len(va)),
                "default_rate_train": float(y[tr_cap].mean()),
                "default_rate_valid": float(y[va].mean()),
            }
        )
        y_va = y[va]
        w_va = week[va]

        def record(name: str, p: np.ndarray, t_start: float) -> None:
            wg = weekly_gini(y_va, p, w_va, min_cases=50)
            st = stability_metric(wg)
            records.append(
                {
                    "fold": f["fold"],
                    "model": name,
                    "auc": auc(y_va, p),
                    "gini": gini(y_va, p),
                    "mean_weekly_gini": st["mean_gini"],
                    "slope": st["slope"],
                    "std_residual": st["std_residual"],
                    "stability_metric": st["stability_metric"],
                    "seconds": time.time() - t_start,
                }
            )

        # baseline
        rate = float(y[tr_cap].mean())
        record("dummy", np.full(len(va), rate), t0)
        if verbose:
            print(f"  fold {f['fold']}: dummy done")

        # scorecard
        t1 = time.time()
        sc_fold = build_scorecard(ds, screen["woe_features"], train_idx=tr_cap)
        record("woe_scorecard", scorecard_predict(sc_fold, X.iloc[va]), t1)
        if verbose:
            print(f"  fold {f['fold']}: scorecard {time.time() - t1:.0f}s")

        # penalised logistic grid
        t2 = time.time()
        pipe = linear_pipeline()
        cap_lr = cap_rows(tr, lr_cap)
        Ztr = pipe.fit_transform(linear_design.iloc[cap_lr], y[cap_lr])
        Zva = pipe.transform(linear_design.iloc[va])
        best_p, best_auc, best_cfg = None, -np.inf, None
        for cfg in PENALTY_GRID:
            fit = fit_single_penalised(Ztr, y[cap_lr], cfg)
            p = fit["model"].predict_proba(Zva)[:, 1]
            a = auc(y_va, p)
            lr_records.append(
                {
                    "fold": f["fold"],
                    "config": fit["config"],
                    "converged": fit["converged"],
                    "n_iter": fit["n_iter"],
                    "n_nonzero": fit["n_nonzero"],
                    "auc": a,
                }
            )
            if fit["converged"] and a > best_auc:
                best_auc, best_p, best_cfg = a, p, fit["config"]
        if best_p is None:  # every fit failed to converge: keep the L2 path
            fit = fit_single_penalised(Ztr, y[cap_lr], {"penalty": "l2", "C": 0.2, "l1_ratio": None})
            best_p = fit["model"].predict_proba(Zva)[:, 1]
            best_cfg = fit["config"]
        record("penalised_lr", best_p, t2)
        if verbose:
            print(f"  fold {f['fold']}: penalised LR grid {time.time() - t2:.0f}s ({best_cfg})")

        if include_trees:
            for kind, name in (("rf", "random_forest"), ("xgb", "xgboost"), ("lgbm", "lightgbm"), ("cat", "catboost")):
                t3 = time.time()
                Xtr = Xt.iloc[tr_cap]
                Xva = Xt.iloc[va]
                if kind == "cat":
                    Xtr = _cat_matrix(Xtr, cat_idx)
                    Xva = _cat_matrix(Xva, cat_idx)
                fit = fit_tree(kind, Xtr, y[tr_cap], Xva, y_va, cat_idx=cat_idx if kind == "cat" else None)
                record(name, fit["model"].predict_proba(Xva)[:, 1], t3)
                if verbose:
                    print(f"  fold {f['fold']}: {name} {time.time() - t3:.0f}s")

    # aggregate the LR grid across folds (train-only evidence for the penalty)
    agg: dict[str, dict] = {}
    for rec in lr_records:
        key = json.dumps(rec["config"], sort_keys=True)
        slot = agg.setdefault(key, {"config": rec["config"], "aucs": [], "converged": [], "n_nonzero": []})
        slot["aucs"].append(rec["auc"])
        slot["converged"].append(rec["converged"])
        slot["n_nonzero"].append(rec["n_nonzero"])
    lr_summary = []
    for key, slot in agg.items():
        lr_summary.append(
            {
                "config": slot["config"],
                "mean_auc": float(np.nanmean(slot["aucs"])),
                "std_auc": float(np.nanstd(slot["aucs"])),
                "all_converged": bool(all(slot["converged"])),
                "mean_nonzero": float(np.mean(slot["n_nonzero"])),
            }
        )
    lr_summary.sort(key=lambda r: -r["mean_auc"])
    eligible = [r for r in lr_summary if r["all_converged"]]
    best_lr = eligible[0]["config"] if eligible else lr_summary[0]["config"]

    return {
        "folds": fold_meta,
        "records": records,
        "lr_grid": lr_summary,
        "best_lr_config": best_lr,
    }


# --------------------------------------------------------------------------
# Leakage gate
# --------------------------------------------------------------------------
def leakage_gate(
    ds: dict,
    screen: dict,
    design_columns: list[str],
    sc: dict | None = None,
    linear_pipe=None,
    linear_design: pd.DataFrame | None = None,
) -> dict:
    """Assert the five leakage invariants; raise if one is violated.

    This is deliberately a *runtime assertion*, not a report: a violated gate
    must stop the run before any artifact is written, otherwise a leaked model
    would be published with a good-looking AUC and a green summary table.
    """
    from hc_config import FORBIDDEN_EXACT, FORBIDDEN_TOKENS

    checks: list[dict] = []

    def add(name: str, passed: bool, detail, critical: bool = True) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail, "critical": critical})

    # 1. forbidden columns
    from hc_data import is_identifier_feature

    bad_exact = [c for c in ds["feature_columns"] if c in FORBIDDEN_EXACT]
    bad_token = [c for c in ds["feature_columns"] if is_identifier_feature(c)]
    add("no_id_or_outcome_columns", not bad_exact and not bad_token,
        {"exact": bad_exact[:10], "token": bad_token[:10]})
    add("target_not_in_design", TARGET not in design_columns, {"columns_checked": len(design_columns)})

    # 2. strict time ordering of the splits
    d = pd.to_datetime(pd.Series(ds["date"]))
    tr_max = d[ds["split"] == "train"].max()
    va_min, va_max = d[ds["split"] == "valid"].min(), d[ds["split"] == "valid"].max()
    oo_min = d[ds["split"] == "oot"].min()
    ordered = bool(tr_max < va_min and va_max < oo_min)
    add("time_split_ordering", ordered,
        {"train_max": str(tr_max.date()), "valid_min": str(va_min.date()),
         "valid_max": str(va_max.date()), "oot_min": str(oo_min.date())})

    # 3. entity disjointness
    ids = ds["case_ids"]
    tr_ids, va_ids, oo_ids = set(ids[ds["split"] == "train"]), set(ids[ds["split"] == "valid"]), set(ids[ds["split"] == "oot"])
    overlap = (tr_ids & oo_ids) | (tr_ids & va_ids) | (va_ids & oo_ids)
    add("no_case_overlap", len(overlap) == 0, {"overlapping_cases": len(overlap)})

    # 4. univariate AUC screen
    flagged = {r["feature"] for r in screen["quarantined"]}
    still_in = sorted(flagged & set(design_columns))
    add("univariate_auc_screen", not still_in, {"quarantined": sorted(flagged)[:10], "still_in_design": still_in})

    # 5. preprocessing fitted on training rows only
    if linear_pipe is not None and linear_design is not None:
        # Compare against the *imputed* design: median imputation shifts the
        # column means, so a raw nanmean comparison would flag a correctly
        # fitted scaler.  The check has teeth because fitting on all rows would
        # make the full-design delta exactly zero and the train delta positive.
        scaler = linear_pipe.named_steps["scale"]
        imputer = linear_pipe.named_steps["impute"]
        imputed_train = imputer.transform(linear_design.loc[ds["train_mask"]])
        imputed_full = imputer.transform(linear_design)
        scaled_mean = float(np.mean(scaler.mean_))
        train_mean = float(imputed_train.mean(axis=0).mean())
        full_mean = float(imputed_full.mean(axis=0).mean())
        diff_train = abs(scaled_mean - train_mean)
        diff_full = abs(scaled_mean - full_mean)
        add("scaler_fitted_on_train_rows", diff_train <= diff_full + 1e-9,
            {"scaled_mean": scaled_mean, "train_mean": train_mean,
             "full_mean": full_mean, "delta_train": diff_train, "delta_full": diff_full})

    if sc is not None:
        # every bin's count must sum to the size of the sample the bins were fitted on
        var = sc["features"][0]
        tab = sc["process"].get_binned_variable(var).binning_table.build()
        # the final row is the summary ("Totals"); counting it would double the sum
        detail = tab[pd.to_numeric(tab["Count"], errors="coerce").notna()].copy()
        detail = detail[detail["Bin"].astype(str).str.strip().str.lower() != "totals"]
        detail = detail[detail["Bin"].notna() & (detail["Bin"].astype(str).str.strip() != "")]
        counts = float(pd.to_numeric(detail["Count"], errors="coerce").sum())
        # The bins are fitted on a declared random subsample of the training
        # rows (WOE_FIT_SAMPLE), not on all of them, so the bin counts must
        # reconcile with the rows that were actually fitted. Comparing against
        # the full train count would fail on a legitimate subsample and, worse,
        # would pass silently if a future refactor fitted on extra rows held
        # inside the same declared sample size.
        fit_rows = float(sc["woe_fit_rows"])
        add("woe_binned_on_train_rows", abs(counts - fit_rows) <= max(10.0, 0.01 * fit_rows),
            {"variable": var, "bin_count_sum": counts, "woe_fit_rows": fit_rows,
             "train_rows": float(sc["train_rows"]), "fit_fraction": sc.get("woe_fit_fraction")})
        add("woe_fit_sample_within_train", fit_rows <= float(sc["train_rows"]) + 1e-9,
            {"woe_fit_rows": fit_rows, "train_rows": float(sc["train_rows"])})

    failed = [c for c in checks if c["status"] == "FAIL" and c["critical"]]
    if failed:
        raise RuntimeError(
            "LEAKAGE GATE FAILED: " + "; ".join(f"{c['check']} -> {c['detail']}" for c in failed)
        )
    return {"checks": checks, "passed": len(failed) == 0}


# --------------------------------------------------------------------------
# Interpretation helpers
# --------------------------------------------------------------------------
def tree_shap_summary(
    model,
    X_sample: pd.DataFrame,
    kind: str,
    top_n: int = 20,
    cat_idx: list[int] | None = None,
) -> dict:
    """Mean absolute SHAP contribution per feature using each library's native path.

    Native ``pred_contrib`` / ``ShapValues`` avoids a hard dependency on the
    ``shap`` package at prediction time and uses exactly the model that is
    shipped.
    """
    feats = list(X_sample.columns)
    if kind == "lgbm":
        contrib = model.predict(X_sample, pred_contrib=True)
        values = np.asarray(contrib)[:, :-1]
    elif kind == "xgb":
        import xgboost as xgb

        booster = model.get_booster()
        dm = xgb.DMatrix(X_sample, feature_names=feats)
        values = np.asarray(booster.predict(dm, pred_contribs=True))[:, :-1]
    elif kind == "cat":
        from catboost import Pool

        # the Pool must declare the same categorical columns the model was
        # trained with, otherwise CatBoost refuses to compute contributions
        pool = Pool(X_sample, cat_features=cat_idx or [])
        values = np.asarray(model.get_feature_importance(type="ShapValues", data=pool))[:, :-1]
    else:  # pragma: no cover - only boosting champions are interpreted
        return {"error": f"SHAP not implemented for {kind}"}

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return {
        "n_rows": int(len(X_sample)),
        "top_features": [
            {"feature": feats[i], "mean_abs_shap": float(mean_abs[i])} for i in order[:top_n]
        ],
        "sample_contributions": [
            {feats[i]: float(values[row, i]) for i in order[:10]}
            for row in range(min(20, values.shape[0]))
        ],
    }


def drift_removal_experiment(
    ds: dict,
    screen: dict,
    drop_features: list[str],
    kind: str = "lgbm",
) -> dict:
    """Refit the champion family without the most drifting features.

    Answers the trade-off question directly: how much pooled AUC and how much
    weekly stability does a model lose when the features that identify the OOT
    cohort are removed?
    """
    X, y, week = ds["X"], ds["y"], ds["week"]
    train_mask, oot_mask, valid_mask = ds["train_mask"], ds["oot_mask"], ds["valid_mask"]
    full = [f for f in screen["tree_features"] if f in X.columns]
    reduced = [f for f in full if f not in set(drop_features)]
    if len(reduced) < 20:
        return {"skipped": "fewer than 20 features would remain"}

    out = {"n_full": len(full), "n_reduced": len(reduced), "dropped": drop_features}
    for name, cols in (("full", full), ("reduced", reduced)):
        fit = fit_tree(kind, X[cols][train_mask], y[train_mask], X[cols][valid_mask], y[valid_mask])
        p_oot = fit["model"].predict_proba(X[cols][oot_mask])[:, 1]
        wg = weekly_gini(y[oot_mask], p_oot, week[oot_mask], min_cases=50)
        st = stability_metric(wg)
        out[name] = {
            "auc_oot": auc(y[oot_mask], p_oot),
            "stability_metric_oot": st["stability_metric"],
            "mean_weekly_gini_oot": st["mean_gini"],
            "slope": st["slope"],
            "std_residual": st["std_residual"],
        }
    out["delta"] = {
        "auc_oot": out["reduced"]["auc_oot"] - out["full"]["auc_oot"],
        "stability_metric_oot": out["reduced"]["stability_metric_oot"] - out["full"]["stability_metric_oot"],
    }
    return out


def build_monitoring(
    p_valid: np.ndarray,
    p_oot: np.ndarray,
    ds: dict,
    model_features: list[str],
    min_cases: int = 150,
) -> dict:
    """PSI/CSI table + score drift + the written monitoring plan.

    The *feature* PSI baseline is the training window (that is what the model was
    fitted on).  The *score* PSI baseline is the VALID cohort, because a
    deployed scorecard is monitored against the last cohort whose realised
    defaults were observed - using train scores would compare against a
    distribution that has already been superseded.
    """
    from hc_config import GINI_DROP_TRIGGER, PSI_ALERT, PSI_WATCH
    from hc_stability import monitoring_plan

    X = ds["X"]
    train_mask, oot_mask, valid_mask = ds["train_mask"], ds["oot_mask"], ds["valid_mask"]
    psi_tab = psi_table(X.loc[train_mask, model_features], X.loc[oot_mask, model_features], model_features)
    drift = score_drift(p_valid, p_oot, ds["week"][oot_mask], ds["y"][oot_mask], min_cases=min_cases)

    # Monthly score PSI: the monitoring cadence a credit-risk team actually runs.
    # reset_index is load-bearing: the group labels must be positions inside the
    # OOT slice, not labels from the full frame, or p_oot[pos] indexes out of range.
    dates = pd.to_datetime(pd.Series(ds["date"]))
    oot_months = dates[oot_mask].reset_index(drop=True).dt.to_period("M")
    monthly = []
    for month, idx in oot_months.groupby(oot_months).groups.items():
        pos = np.array(idx)
        value = psi(p_valid, p_oot[pos])
        monthly.append({"month": str(month), "n": int(len(pos)), "psi": value, "band": psi_band(value)})

    # CSI on the encoded categorical levels (the raw strings are not retained in
    # the model matrix, so the level codes are the monitored characteristic).
    cat_feats = [f for f in model_features if f in ds["cat_cols"]][:15]
    csi_tab = csi_table(X.loc[train_mask], X.loc[oot_mask], cat_feats) if cat_feats else pd.DataFrame()
    wg_tr = weekly_gini(ds["y"][valid_mask], p_valid, ds["week"][valid_mask], min_cases=50)
    wg_oo = weekly_gini(ds["y"][oot_mask], p_oot, ds["week"][oot_mask], min_cases=50)
    st_tr, st_oo = stability_metric(wg_tr), stability_metric(wg_oo)
    plan = monitoring_plan(
        PSI_WATCH, PSI_ALERT, GINI_DROP_TRIGGER, min_cases,
        st_tr["mean_gini"], st_oo["mean_gini"],
    )
    return {
        "psi_table": psi_tab.to_dict(orient="records"),
        "psi_by_month": monthly,
        "csi_table": csi_tab.to_dict(orient="records") if not csi_tab.empty else [],
        "n_alert": int((psi_tab["band"] == "alert").sum()),
        "n_watch": int((psi_tab["band"] == "watch").sum()),
        "score_drift": drift,
        "monitoring_plan": plan,
        "reference_split": "valid",
        "reference_mean_gini": st_tr["mean_gini"],
        "oot_mean_gini": st_oo["mean_gini"],
    }


# --------------------------------------------------------------------------
# Artifact helpers
# --------------------------------------------------------------------------
def to_jsonable(obj):
    """Recursively convert numpy/pandas objects into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat() if not isinstance(obj, pd.Timestamp) else str(obj.date())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict) -> Path:
    """Write an artifact with sorted keys and no NaN (JSON has no NaN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path
