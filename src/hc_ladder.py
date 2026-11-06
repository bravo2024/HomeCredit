"""Model ladder helpers: metrics, sign-fixed scorecard, tuning, paired bootstrap,
calibration, credit-policy layer and plain-English feature names.

Everything here is fitted on TRAIN rows (or, where stated, on VALID rows) and
only *applied* to OOT rows.  Nothing in this module reads OOT labels before the
champion, calibrator and cut-off have been frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from hc_metrics import auc, brier, gini, ks_stat, logloss, pr_auc, stability_metric, weekly_gini
from hc_model import fit_scorecard, points_table, sign_check


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metric_block(y, p) -> dict:
    """AUC, Gini, KS, PR-AUC, Brier and log loss on raw probabilities."""
    return {
        "auc": auc(y, p), "gini": gini(y, p), "ks": ks_stat(y, p),
        "pr_auc": pr_auc(y, p), "brier": brier(y, p), "logloss": logloss(y, p),
    }


def eval_split(y, p, week) -> dict:
    """Metric block + weekly Gini + competition stability metric for one split."""
    n_weeks = max(len(np.unique(week)), 1)
    min_cases = int(max(20, min(150, (len(y) / n_weeks) / 3)))
    wk = weekly_gini(y, p, week, min_cases=min_cases)
    out = metric_block(y, p)
    out["stability"] = stability_metric(wk)
    out["weekly"] = wk.to_dict(orient="records")
    out["weekly_min_cases"] = min_cases
    return out


# --------------------------------------------------------------------------
# Scorecard: iterative removal of wrong-sign variables
# --------------------------------------------------------------------------
def fix_scorecard_signs(ds: dict, sc: dict, max_rounds: int = 80) -> tuple[dict, list[dict], dict]:
    """Drop the wrong-sign variable with the largest p-value, refit, repeat.

    A WoE-scorecard coefficient must have the sign implied by its own monotone
    bins.  A wrong sign is a collinearity symptom.  Rounds continue until ALL
    signs are consistent.  Returns the updated scorecard, the drop log and the
    sign count before the first drop.
    """
    y = ds["y"]
    train_idx = np.where(ds["train_mask"])[0]
    woe = sc["woe_train"]
    feats = list(sc["features"])
    log: list[dict] = []
    first = None
    fit = None
    for rnd in range(max_rounds + 1):
        fit = fit_scorecard(woe[feats], y[train_idx], woe[feats].iloc[:1])
        coef = fit["coef_table"]
        signs = sign_check(coef, sc["process"], None, None)
        n_ok = sum(1 for r in signs if r["consistent"])
        if first is None:
            first = {"consistent": n_ok, "total": len(signs)}
        wrong = [r["feature"] for r in signs if not r["consistent"]]
        if not wrong:
            break
        pv = coef.set_index("feature")["p_value"]
        worst = max(wrong, key=lambda f: float(pv.get(f, 0.0)))
        log.append({"round": rnd + 1, "dropped": worst, "p_value": float(pv.get(worst, np.nan)),
                    "n_wrong_before_drop": len(wrong), "n_variables_before_drop": len(feats)})
        feats.remove(worst)
    pts, base_points = points_table(sc["process"], coef, pdo=20.0, base_score=600.0, base_odds=20.0)
    sc = dict(sc)
    sc.update({"features": feats, "woe_train": woe[feats], "coef_table": coef, "result": fit["result"],
               "sign_check": signs, "points": pts, "base_points": base_points})
    return sc, log, first


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
def tune_lr(Ztr, ytr, Zva, yva, Cs=(0.001, 0.01, 0.1, 1.0), sub: int = 400_000, seed: int = 42):
    """L2 logistic regression: choose C on VALID AUC (fit on a train subsample), refit on all train."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ytr), min(sub, len(ytr)), replace=False))
    rows = []
    for C in Cs:
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=300)
        m.fit(Ztr[idx], ytr[idx])
        rows.append({"C": C, "valid_auc": auc(yva, m.predict_proba(Zva)[:, 1]), "n_iter": int(m.n_iter_[0])})
    tbl = pd.DataFrame(rows)
    best = float(tbl.loc[tbl["valid_auc"].idxmax(), "C"])
    final = LogisticRegression(penalty="l2", C=best, solver="lbfgs", max_iter=500)
    final.fit(Ztr, ytr)
    return final, best, tbl, int(final.n_iter_[0])


_XGB_DEVICE = None          # "cuda" or "cpu", decided once per process
DEVICE_LOG: dict = {}       # model family -> device actually used (written to environment.json)


def xgb_device() -> str:
    """"cuda" when a GPU is visible and XGBoost can train on it, else "cpu".

    Only XGBoost may use the GPU; LightGBM, the random forest, the logistic regression and the
    scorecard always run on the CPU. The parameters are identical on both devices
    (tree_method="hist"); only ``device`` differs.
    """
    global _XGB_DEVICE
    if _XGB_DEVICE is None:
        dev = "cpu"
        try:
            import subprocess

            if subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0:
                from xgboost import XGBClassifier

                XGBClassifier(n_estimators=2, tree_method="hist", device="cuda").fit(
                    np.random.rand(64, 3), np.arange(64) % 2)
                dev = "cuda"
        except Exception:
            dev = "cpu"
        _XGB_DEVICE = dev
    return _XGB_DEVICE


def fit_xgb(params: dict, Xtr, ytr, Xva, yva, monotone=None):
    global _XGB_DEVICE
    from xgboost import XGBClassifier

    kw = dict(n_estimators=800, learning_rate=0.05, tree_method="hist", eval_metric="auc",
              early_stopping_rounds=50, random_state=42, n_jobs=-1)
    kw.update(params)
    if monotone is not None:
        kw["monotone_constraints"] = tuple(int(v) for v in monotone)
    kw["device"] = xgb_device()
    try:
        m = XGBClassifier(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except Exception:
        if kw["device"] == "cpu":
            raise
        _XGB_DEVICE = "cpu"   # GPU failed mid-run: fall back to the CPU for this and later fits
        kw["device"] = "cpu"
        m = XGBClassifier(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    DEVICE_LOG["xgboost"] = kw["device"]
    return m


def xgb_random_search(Xtr, ytr, Xva, yva, n_iter: int = 6, sub: int = 500_000, seed: int = 42):
    """Small random search on a train subsample, scored on VALID AUC."""
    rng = np.random.default_rng(seed)
    space = {
        "max_depth": [4, 5, 6, 7],
        "min_child_weight": [20, 50, 100],
        "colsample_bytree": [0.3, 0.5],
        "subsample": [0.7, 0.9],
        "reg_lambda": [1.0, 5.0, 20.0],
    }
    idx = np.sort(rng.choice(len(ytr), min(sub, len(ytr)), replace=False))
    Xs, ys = Xtr.iloc[idx], ytr[idx]
    rows = []
    for i in range(n_iter):
        params = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        m = fit_xgb(params, Xs, ys, Xva, yva)
        rows.append({**params, "valid_auc": auc(yva, m.predict_proba(Xva)[:, 1]),
                     "best_iteration": int(getattr(m, "best_iteration", 0) or 0)})
    tbl = pd.DataFrame(rows)
    best_row = tbl.loc[tbl["valid_auc"].idxmax()]
    best = {k: (int(best_row[k]) if k in ("max_depth", "min_child_weight") else float(best_row[k]))
            for k in space}
    return best, tbl


def monotone_signs(X: pd.DataFrame, y: np.ndarray, sub: int = 200_000, min_abs: float = 0.02, seed: int = 42):
    """Sign of the train Spearman correlation of every feature with the target (0 if weak)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), min(sub, len(y)), replace=False))
    Xs = X.iloc[idx]
    ys = pd.Series(y[idx], index=Xs.index, dtype="float32")
    out = []
    for c in X.columns:
        r = Xs[c].corr(ys, method="spearman")
        out.append(0 if (not np.isfinite(r) or abs(r) < min_abs) else int(np.sign(r)))
    return out


def fit_rf(Xtr, ytr, sub: int = 500_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ytr), min(sub, len(ytr)), replace=False))
    m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=50, max_features=0.3,
                               n_jobs=-1, random_state=seed)
    m.fit(Xtr.iloc[idx], ytr[idx])
    return m, int(len(idx))


# --------------------------------------------------------------------------
# Champion rule (frozen on VALID)
# --------------------------------------------------------------------------
def pick_champion(valid_gini: dict, main: str = "xgboost", tol: float = 0.002) -> dict:
    """Best VALID Gini; anything within ``tol`` of the best is a tie won by the pre-declared main model."""
    best = max(valid_gini, key=valid_gini.get)
    tied = [m for m, g in valid_gini.items() if valid_gini[best] - g <= tol]
    champion = main if main in tied else best
    return {"rule": f"max VALID Gini; models within {tol} Gini of the best are tied and the tie goes to '{main}'",
            "valid_gini": valid_gini, "best_by_valid_gini": best, "tied": tied, "champion": champion}


# --------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------
def paired_bootstrap(y, preds: dict, ref: str, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired application-level bootstrap of Gini(ref) - Gini(other); same resample for every model."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    names = [k for k in preds if k != ref]
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        g_ref = gini(yb, preds[ref][idx])
        for k in names:
            draws[k].append(g_ref - gini(yb, preds[k][idx]))
    out = {}
    for k in names:
        d = np.asarray(draws[k])
        out[f"{ref}_minus_{k}"] = {
            "point_gini_diff": float(gini(y, preds[ref]) - gini(y, preds[k])),
            "ci95_lo": float(np.quantile(d, 0.025)), "ci95_hi": float(np.quantile(d, 0.975)),
            "n_boot": int(len(d)), "share_positive": float((d > 0).mean()),
        }
    return out


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Platt:
    """Platt scaling: logistic regression on logit(p), fitted on VALID."""

    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5).fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self.m.predict(p), 1e-5, 1 - 1e-5)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(n=("y", "size"), p=("p", "mean"), o=("y", "mean"))
    return float((g["n"] * (g["p"] - g["o"]).abs()).sum() / g["n"].sum())


def choose_calibrator(p_valid, y_valid, week_valid) -> dict:
    """Pick none / Platt / isotonic on VALID: fit on early VALID weeks, score log loss on late VALID weeks."""
    weeks = np.sort(np.unique(week_valid))
    cut = weeks[len(weeks) // 2]
    early, late = week_valid < cut, week_valid >= cut
    rows = []
    cands = {"none": None, "platt": Platt, "isotonic": Iso}
    for name, cls in cands.items():
        if cls is None:
            pl_ = p_valid[late]
        else:
            pl_ = cls().fit(p_valid[early], y_valid[early]).predict(p_valid[late])
        rows.append({"method": name, "late_valid_logloss": logloss(y_valid[late], pl_),
                     "late_valid_brier": brier(y_valid[late], pl_), "late_valid_ece": ece(y_valid[late], pl_)})
    tbl = pd.DataFrame(rows)
    best = str(tbl.loc[tbl["late_valid_logloss"].idxmin(), "method"])
    fitted = {n: (cls().fit(p_valid, y_valid) if cls else None) for n, cls in cands.items()}
    return {"choice": best, "selection_table": tbl, "calibrators": fitted,
            "split": {"fit_weeks_lt": int(cut), "score_weeks_ge": int(cut)}}


def ea_deciles(y, p, bins: int = 10) -> pd.DataFrame:
    """Expected-vs-actual by decile of predicted PD."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    g = df.groupby("decile").agg(n=("y", "size"), expected=("p", "mean"), actual=("y", "mean")).reset_index()
    g["expected_over_actual"] = g["expected"] / g["actual"].replace(0, np.nan)
    return g


# --------------------------------------------------------------------------
# Business layer (ALL economics are assumptions)
# --------------------------------------------------------------------------
def capture_table(y, preds: dict, fracs=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Share of defaulters caught when the riskiest ``frac`` of applications is reviewed."""
    y = np.asarray(y)
    total = max(int(y.sum()), 1)
    rows = []
    for name, p in preds.items():
        row = {"model": name}
        for f in fracs:
            m = max(int(round(f * len(y))), 1)
            top = np.argsort(-np.asarray(p), kind="stable")[:m]
            row[f"caught_at_{int(f * 100)}pct"] = float(y[top].sum() / total)
        rows.append(row)
    rows.append({"model": "random (reference)", **{f"caught_at_{int(f * 100)}pct": float(f) for f in fracs}})
    return pd.DataFrame(rows)


def cutoff_for_bad_rate(score_valid, y_valid, target_bad_rate: float, min_share: float = 0.05) -> float:
    """Largest approval set (lowest scores first) on VALID whose bad rate stays at or below the target.

    Returns the score cut-off: approve if score < cut-off.  Applied unchanged to OOT.
    """
    order = np.argsort(score_valid, kind="stable")
    ys = np.asarray(y_valid)[order]
    cum = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(cum <= target_bad_rate)[0]
    ok = ok[ok >= int(min_share * len(ys))]
    k = int(ok.max()) + 1 if len(ok) else int(min_share * len(ys))
    return float(np.sort(score_valid)[min(k, len(ys) - 1)])


def cutoff_for_approval_rate(score_valid, approval_rate: float) -> float:
    return float(np.quantile(score_valid, approval_rate))


def policy_stats(y, approve: np.ndarray) -> dict:
    """Observed outcomes of a cut-off on a set of loans (all rates on observed defaults)."""
    y = np.asarray(y)
    return {
        "n_applications": int(len(y)), "approval_rate": float(approve.mean()),
        "bad_rate_all": float(y.mean()),
        "bad_rate_approved": float(y[approve].mean()) if approve.any() else float("nan"),
        "bad_rate_declined": float(y[~approve].mean()) if (~approve).any() else float("nan"),
        "defaulters_declined_share": float(y[~approve].sum() / max(y.sum(), 1)),
    }


def swap_sets(y, score_a, score_b, approval_rate: float) -> dict:
    """Same approval rate for both models (lowest-risk share by each model's own OOT ranking).

    Swap-in  = approved by A, declined by B.   Swap-out = declined by A, approved by B.
    """
    y = np.asarray(y)
    k = int(round(approval_rate * len(y)))
    ap_a = np.zeros(len(y), dtype=bool); ap_a[np.argsort(score_a, kind="stable")[:k]] = True
    ap_b = np.zeros(len(y), dtype=bool); ap_b[np.argsort(score_b, kind="stable")[:k]] = True
    def blk(m):
        return {"n": int(m.sum()), "bad_rate": float(y[m].mean()) if m.any() else float("nan"), "defaults": int(y[m].sum())}
    return {"approval_rate": approval_rate, "n_approved_each": k,
            "both_approve": blk(ap_a & ap_b), "both_decline": blk(~ap_a & ~ap_b),
            "swap_in_A_only": blk(ap_a & ~ap_b), "swap_out_B_only": blk(~ap_a & ap_b),
            "portfolio_bad_rate_A": float(y[ap_a].mean()), "portfolio_bad_rate_B": float(y[ap_b].mean())}


def master_scale(score_valid, y_valid, score_oot, y_oot, n_grades: int = 10, ead_oot=None, lgd: float = 0.45) -> pd.DataFrame:
    """Risk grades from VALID quantiles of the score; grade PD = VALID observed default rate.

    OOT: observed default rate per grade and an exact two-sided binomial test of that rate against the grade PD.
    Expected loss per grade (illustrative) = PD x LGD x EAD, against realised loss = default x LGD x EAD.
    """
    from scipy.stats import binomtest

    edges = np.quantile(score_valid, np.linspace(0, 1, n_grades + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    gv = np.digitize(score_valid, edges[1:-1], right=True)
    go = np.digitize(score_oot, edges[1:-1], right=True)
    yv, yo = np.asarray(y_valid), np.asarray(y_oot)
    ead = np.ones(len(yo)) if ead_oot is None else np.asarray(ead_oot, dtype=float)
    rows = []
    for g in range(len(edges) - 1):
        mv, mo = gv == g, go == g
        pd_g = float(yv[mv].mean())
        n_o, k_o = int(mo.sum()), int(yo[mo].sum())
        pval = float(binomtest(k_o, n_o, min(max(pd_g, 1e-6), 1 - 1e-6)).pvalue) if n_o else float("nan")
        rows.append({"grade": g + 1, "n_valid": int(mv.sum()), "grade_pd_valid": pd_g, "n_oot": n_o, "defaults_oot": k_o,
                     "oot_default_rate": k_o / max(n_o, 1), "binomial_p": pval,
                     "direction": "lower than PD" if k_o / max(n_o, 1) < pd_g else "higher than PD",
                     "expected_loss": float((pd_g * lgd * ead[mo]).sum()),
                     "realised_loss": float((yo[mo] * lgd * ead[mo]).sum())})
    t = pd.DataFrame(rows)
    t["el_over_realised"] = t["expected_loss"] / t["realised_loss"].replace(0, np.nan)
    return t


# --------------------------------------------------------------------------
# Default-rate drop: mix vs rate decomposition
# --------------------------------------------------------------------------
def decile_decomposition(s_train, y_train, s_other, y_other, bins: int = 10) -> dict:
    """Change in default rate train -> other split, split into score-mix and within-band rate effects.

    Bands are deciles of the TRAIN score.  total = mix + rate holds exactly:
    mix  = sum_d (w_other_d - w_train_d) * r_train_d
    rate = sum_d  w_other_d * (r_other_d - r_train_d)
    """
    edges = np.quantile(s_train, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    bt = np.digitize(s_train, edges[1:-1], right=True)
    bo = np.digitize(s_other, edges[1:-1], right=True)
    k = len(edges) - 1
    rows = []
    for d in range(k):
        mt, mo = bt == d, bo == d
        rows.append({"band": d + 1, "w_train": float(mt.mean()), "w_other": float(mo.mean()),
                     "rate_train": float(y_train[mt].mean()), "rate_other": float(y_other[mo].mean()) if mo.any() else float("nan")})
    t = pd.DataFrame(rows)
    mix = float(((t["w_other"] - t["w_train"]) * t["rate_train"]).sum())
    rate = float((t["w_other"] * (t["rate_other"] - t["rate_train"])).sum())
    return {"table": t, "total_change": float(y_other.mean() - y_train.mean()), "mix_effect": mix, "rate_effect": rate}


# --------------------------------------------------------------------------
# Plain-English feature names
# --------------------------------------------------------------------------
_TABLE_LABEL = (
    ("static_cb", "credit bureau summary"), ("static_0", "application"), ("applprev", "previous applications"),
    ("credit_bureau_a", "credit bureau (a)"), ("credit_bureau_b", "credit bureau (b)"),
    ("person", "applicant / related person"), ("tax_registry", "tax registry"), ("deposit", "deposit account"),
    ("other", "other obligations"), ("debitcard", "debit card"),
)
_AGG_TEXT = {"mean": "average", "max": "maximum", "min": "minimum", "std": "variability (std) of",
             "nunique": "number of distinct values of", "maxcode": "most recent category of",
             "n_records": "number of records", "recency_days": "days since latest record",
             "rel_mean": "average time before decision of", "rel_max": "maximum time before decision of",
             "rel_min": "minimum time before decision of", "rel_std": "variability (std) of time before decision of"}


def plain_name(feat: str, defs: dict) -> str:
    parts = feat.split("__")
    prefix = parts[0].replace("train_", "")
    label = next((v for k, v in _TABLE_LABEL if k in prefix), prefix)
    if len(parts) == 2 and parts[1] in _AGG_TEXT:
        return f"{_AGG_TEXT[parts[1]]} [{label}]"
    col = parts[1] if len(parts) > 1 else feat
    desc = defs.get(col, col)
    desc = str(desc).split(". ")[0].rstrip(".")
    if len(desc) > 80:
        desc = desc[:77].rstrip() + "..."
    if len(parts) >= 3 and parts[2] in _AGG_TEXT:
        return f"{_AGG_TEXT[parts[2]]} {desc} [{label}]"
    return f"{desc} [{label}]"
