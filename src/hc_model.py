"""Model roster: dummy, WoE scorecard, penalised logistic, forests, boosting.

Every model exposes the same interface - ``fit`` then ``predict_proba`` on a raw
feature frame - so the comparison, the weekly-Gini panel and the champion
selection all run through one code path.  That sameness is the point: a
different preprocessing route per family would make the stability comparison
meaningless.

Notes that matter for interpreting the results:

* the scorecard is deliberately restricted to a monotone, VIF-pruned WoE design
  because that is what a regulated scorecard has to be, not because it scores
  best;
* the penalised logistic runs on the *standardised raw* design, so it can use
  information the monotone binning throws away;
* the gradient-boosted models see the same columns with NaN left in place -
  their whole advantage is modelling missingness and interactions natively.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from hc_metrics import auc


# --------------------------------------------------------------------------
# Baselines and calibration
# --------------------------------------------------------------------------
class ConstantModel:
    """Majority-class / base-rate baseline.

    Predicts the training prevalence for everyone.  Any model that cannot beat
    this is not measuring risk at all, so the baseline is a *gate*, not a
    formality.
    """

    def __init__(self) -> None:
        self.rate = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConstantModel":
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate)


class IsotonicCalibrator:
    """Isotonic recalibration fitted on the VALID weeks only.

    Fitting on VALID (never on train, never on OOT) means the calibrator cannot
    learn anything about the evaluation cohort, and the OOT score distribution
    therefore measures genuine drift.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.iso = IsotonicRegression(out_of_bounds=out_of_bounds, y_min=1e-5, y_max=1 - 1e-5)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(np.clip(p, 1e-6, 1 - 1e-6), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(np.clip(p, 1e-6, 1 - 1e-6)), 1e-5, 1 - 1e-5)


# --------------------------------------------------------------------------
# Penalised logistic
# --------------------------------------------------------------------------
PENALTY_GRID: tuple[dict, ...] = (
    {"penalty": "l2", "C": 0.05, "l1_ratio": None},
    {"penalty": "l2", "C": 0.2, "l1_ratio": None},
    {"penalty": "l2", "C": 1.0, "l1_ratio": None},
    {"penalty": "l1", "C": 0.05, "l1_ratio": None},
    {"penalty": "l1", "C": 0.2, "l1_ratio": None},
    {"penalty": "elasticnet", "C": 0.05, "l1_ratio": 0.25},
    {"penalty": "elasticnet", "C": 0.2, "l1_ratio": 0.25},
    {"penalty": "elasticnet", "C": 0.2, "l1_ratio": 0.50},
    {"penalty": "elasticnet", "C": 0.2, "l1_ratio": 0.75},
    {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.50},
)


def make_penalised(cfg: dict, max_iter: int = 3000, tol: float = 1e-3) -> LogisticRegression:
    """Build one saga logistic-regression estimator from a grid config."""
    return LogisticRegression(
        penalty=cfg["penalty"],
        C=cfg["C"],
        l1_ratio=cfg["l1_ratio"],
        solver="saga",
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        n_jobs=-1,
    )


def fit_single_penalised(
    design_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    max_iter: int = 3000,
    tol: float = 1e-3,
) -> dict:
    """Fit one penalised configuration and report whether saga converged.

    A run that hits ``max_iter`` is *not* a fit: the coefficients are a snapshot
    of an unfinished optimisation, so the caller must exclude it rather than
    report an AUC that will not reproduce.
    """
    model = make_penalised(cfg, max_iter=max_iter, tol=tol)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(design_train, y_train)
    converged = int(model.n_iter_[0]) < max_iter and not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    return {
        "model": model,
        "config": dict(cfg),
        "converged": bool(converged),
        "n_iter": int(model.n_iter_[0]),
        "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-8)),
    }


def fit_penalised_logistic(
    design_train: np.ndarray,
    y_train: np.ndarray,
    design_eval: np.ndarray,
    y_eval: np.ndarray,
    max_iter: int = 3000,
) -> list[dict]:
    """Fit the L2 / L1 / ElasticNet grid with saga.

    Returns one record per configuration including the ``converged`` flag; the
    caller is expected to exclude non-converged fits rather than silently
    reporting their coefficients, because a saga run that stops at the
    iteration cap has not found the penalty path it advertises.
    """
    results = []
    for cfg in PENALTY_GRID:
        model = LogisticRegression(
            penalty=cfg["penalty"],
            C=cfg["C"],
            l1_ratio=cfg["l1_ratio"],
            solver="saga",
            max_iter=max_iter,
            tol=1e-4,
            random_state=42,
            n_jobs=-1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(design_train, y_train)
        converged = int(model.n_iter_[0]) < max_iter and not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        p = model.predict_proba(design_eval)[:, 1]
        results.append(
            {
                "model": model,
                "config": {k: v for k, v in cfg.items()},
                "converged": bool(converged),
                "n_iter": int(model.n_iter_[0]),
                "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-8)),
                "eval_auc": float(auc(y_eval, p)),
            }
        )
    return results


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------
def tree_model(kind: str, cat_features: list[int] | None = None):
    """Factory with fixed, documented hyper-parameters.

    The parameters are set a priori from the published defaults for tabular
    credit data (shallow trees, strong row/column subsampling, early stopping on
    VALID) rather than tuned on OOT, which would turn the out-of-time estimate
    into a validation score.
    """
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=50,
            max_features=0.3,
            n_jobs=-1,
            random_state=42,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.4,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.4,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if kind == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=50,
            cat_features=cat_features or [],
            random_seed=42,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        )
    raise ValueError(f"unknown tree kind {kind}")


def fit_tree(
    kind: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    cat_idx: list[int] | None = None,
) -> dict:
    """Fit one boosting/forest model with early stopping on the evaluation fold."""
    Xtr = X_train
    Xev = X_eval
    if kind == "cat":
        # CatBoost needs integer categories: NaN would be read as a float level.
        Xtr = Xtr.copy()
        Xev = Xev.copy()
        for j in cat_idx or []:
            col = Xtr.columns[j]
            Xtr[col] = Xtr[col].fillna(-2).astype("int64").astype(str)
            Xev[col] = Xev[col].fillna(-2).astype("int64").astype(str)

    model = tree_model(kind, cat_features=cat_idx)
    if kind == "rf":
        model.fit(Xtr, y_train)
        best_iter = model.n_estimators
    elif kind == "cat":
        model.fit(Xtr, y_train, eval_set=(Xev, y_eval), use_best_model=True)
        best_iter = int(model.get_best_iteration() or 0)
    elif kind == "lgbm":
        import lightgbm as lgb

        # LightGBM declares categorical columns at fit time and wants integer
        # codes; -1 is its documented "missing" level for a categorical.
        cat_names = []
        for j in cat_idx or []:
            col = Xtr.columns[j]
            Xtr[col] = Xtr[col].fillna(-1).astype("int32")
            Xev[col] = Xev[col].fillna(-1).astype("int32")
            cat_names.append(col)
        model.fit(
            Xtr,
            y_train,
            eval_set=[(Xev, y_eval)],
            categorical_feature=cat_names,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
    else:
        model.fit(Xtr, y_train, eval_set=[(Xev, y_eval)], verbose=False)
        best_iter = int(getattr(model, "best_iteration", 0) or 0)
    return {"model": model, "best_iter": best_iter}


# --------------------------------------------------------------------------
# WoE scorecard
# --------------------------------------------------------------------------
def fit_woe_binning(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical: list[str],
    max_n_bins: int = 8,
    min_bin_size: float = 0.02,
) -> "BinningProcess":  # noqa: F821 - optbinning type, imported lazily
    """Fit supervised monotone WoE binning on training rows only.

    ``monotonic_trend='auto'`` lets optbinning choose the direction per variable
    but forces the bins to be monotone in the event rate; that constraint is
    what makes a points table explainable to a credit committee.
    """
    from optbinning import BinningProcess

    process = BinningProcess(
        variable_names=list(X.columns),
        categorical_variables=categorical,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        max_n_prebins=20,
        min_prebin_size=0.05,
        # optbinning takes the monotonicity constraint per variable; "auto"
        # picks the direction from the data but still forces a monotone event
        # rate within the variable, which is the property a points table needs.
        binning_fit_params={c: {"monotonic_trend": "auto"} for c in X.columns},
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process.fit(X, y)
    return process


def woe_design(
    process,
    X: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Transform raw columns into WoE columns, imputing any gap with the mean.

    ``process.transform`` requires *all* fitted variables to be present, so the
    caller passes the fitted frame and selects ``columns`` afterwards; this
    keeps the binning object reusable for scoring without refitting.
    """
    out = process.transform(X, metric="woe")
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=list(process.get_support(names=True)))
    out = out.reset_index(drop=True)
    out.index = X.index
    out = out.fillna(out.mean(numeric_only=True)).fillna(0.0)
    if columns is not None:
        out = out[[c for c in columns if c in out.columns]]
    return out


def bin_monotonicity(process, variable: str) -> dict:
    """Check that the fitted bins are monotone in the event rate."""
    tab = process.get_binned_variable(variable).binning_table.build()
    rate = pd.to_numeric(tab["Event rate"], errors="coerce").dropna().to_numpy(dtype=float)
    diffs = np.diff(rate)
    return {
        "variable": variable,
        "n_bins": int(len(rate)),
        "monotone_increasing": bool(np.all(diffs >= -1e-12)),
        "monotone_decreasing": bool(np.all(diffs <= 1e-12)),
        "monotone": bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)),
    }


def sanitize_design(df: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Drop zero-variance and exactly-duplicated WoE columns.

    Binning can map two different raw variables onto the same monotone WoE
    pattern, which makes the Hessian singular. VIF pruning normally catches it,
    but a short CV fold can still produce a duplicate; dropping it here keeps
    the Logit well posed and the drop is visible in the returned column list.
    """
    keep = [c for c in df.columns if float(df[c].std()) > tol]
    if len(keep) > 1:
        corr = df[keep].corr().abs()
        drop: set[str] = set()
        for i, a in enumerate(keep):
            if a in drop:
                continue
            for b in keep[i + 1:]:
                if b in drop:
                    continue
                if corr.loc[a, b] > 0.999:
                    drop.add(b)
        keep = [c for c in keep if c not in drop]
    return keep


def fit_scorecard(
    design_train: pd.DataFrame,
    y_train: np.ndarray,
    design_valid: pd.DataFrame,
    method: str = "newton",
) -> dict:
    """statsmodels Logit on the WoE design, with the full coefficient table.

    Newton-Raphson is the default because it converges in a handful of
    iterations on a well-posed design. If the Hessian is singular anyway (a
    short fold, a variable that collapses to one bin) the fit falls back to the
    GLM/IRLS path, which uses a pseudo-inverse and still returns standard
    errors - better than shipping a scorecard whose covariance matrix was never
    computed.
    """
    import statsmodels.api as sm

    X = sm.add_constant(design_train.to_numpy(dtype=float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(y_train, X).fit(disp=0, maxiter=300, method=method)
        except Exception:
            res = sm.GLM(y_train, X, family=sm.families.Binomial()).fit()
    coef = pd.DataFrame(
        {
            "feature": ["const"] + list(design_train.columns),
            "coef": res.params,
            "std_err": res.bse,
            "z": res.tvalues,
            "p_value": res.pvalues,
            "ci_low": res.conf_int()[:, 0],
            "ci_high": res.conf_int()[:, 1],
        }
    )
    p_valid = res.predict(sm.add_constant(design_valid.to_numpy(dtype=float), has_constant="add"))
    return {"result": res, "coef_table": coef, "p_valid": np.asarray(p_valid)}


def sign_check(
    coef_table: pd.DataFrame,
    process,
    X_reference: pd.DataFrame,
    y_reference: np.ndarray,
) -> list[dict]:
    """Verify every coefficient has the sign implied by its own bin pattern.

    The expected sign is derived from the data, not assumed: for each variable
    the Spearman correlation between bin WoE and bin event rate is computed on
    training rows, and the fitted coefficient must share that sign.  A flipped
    coefficient is the classic symptom of collinearity that VIF pruning missed.
    """
    from scipy.stats import spearmanr

    rows = []
    for _, r in coef_table.iterrows():
        if r["feature"] == "const":
            continue
        var = r["feature"]
        try:
            tab = process.get_binned_variable(var).binning_table.build()
            # the summary row carries empty strings, so coerce before the test
            rate = pd.to_numeric(tab["Event rate"], errors="coerce").to_numpy(dtype=float)
            woe = pd.to_numeric(tab["WoE"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(rate) & np.isfinite(woe)
            rho = spearmanr(woe[mask], rate[mask]).statistic
        except Exception:  # pragma: no cover - binning table unavailable
            rho = np.nan
        expected = np.sign(rho) if np.isfinite(rho) else 0
        rows.append(
            {
                "feature": var,
                "coef": float(r["coef"]),
                "expected_sign": float(expected),
                "woe_event_rate_spearman": float(rho) if np.isfinite(rho) else None,
                "consistent": bool(expected != 0 and np.sign(r["coef"]) == expected),
            }
        )
    return rows


def points_table(
    process,
    coef_table: pd.DataFrame,
    pdo: float,
    base_score: float,
    base_odds: float,
) -> tuple[pd.DataFrame, float]:
    """PDO points for every bin of every scorecard variable.

    ``score = offset - factor * (b0 + sum_j b_j * woe_j)``, so the points a bin
    contributes are ``-factor * b_j * woe_bin`` and the intercept supplies the
    base points.  Summing the base points and the selected bins reproduces the
    score exactly - the notebook verifies that reconstruction.
    """
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    beta = dict(zip(coef_table["feature"], coef_table["coef"]))
    base_points = offset - factor * beta["const"]

    rows = []
    for var in [f for f in beta if f != "const"]:
        tab = process.get_binned_variable(var).binning_table.build()
        for _, r in tab.iterrows():
            label = r.get("Bin", "")
            # the summary row is emitted with an empty bin label in optbinning 1.x
            if pd.isna(label) or str(label).strip() == "" or str(label).strip().lower() == "totals":
                continue
            rows.append(
                {
                    "feature": var,
                    "bin": str(r.get("Bin", "")),
                    "count": float(r.get("Count", np.nan)),
                    "event_rate": float(r.get("Event rate", np.nan)),
                    "woe": float(r.get("WoE", np.nan)),
                    "points": float(-factor * beta[var] * float(r.get("WoE", 0.0))),
                }
            )
    return pd.DataFrame(rows), float(base_points)


def reason_codes(
    design_row: pd.Series,
    coef_table: pd.DataFrame,
    reference_mean: pd.Series,
    top_k: int = 3,
) -> list[dict]:
    """Largest risk-increasing scorecard contributions for one application.

    Contributions are expressed as points *relative to the portfolio average*
    for that variable, so the listed reasons are the ones that make this
    applicant look worse than the average applicant - exactly what a decline
    letter or an underwriter's note needs.
    """
    beta = dict(zip(coef_table["feature"], coef_table["coef"]))
    rows = []
    for var in design_row.index:
        if var not in beta:
            continue
        dev = float(design_row[var] - reference_mean.get(var, 0.0))
        rows.append({"feature": var, "woe": float(design_row[var]), "contribution": float(beta[var] * dev)})
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows[:top_k]


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def expanding_folds(train: pd.DataFrame, n_folds: int = 4) -> list[dict]:
    """Expanding-window folds over the training weeks.

    Fold *k* trains on weeks up to block *k* and validates on block *k+1*, so
    every validation block is strictly later than its training rows - the same
    temporal discipline as the OOT split, just applied inside train.
    """
    weeks = np.sort(train["week"].unique())
    blocks = np.array_split(weeks, n_folds + 1)
    folds = []
    for k in range(n_folds):
        tr_weeks = np.concatenate(blocks[: k + 1])
        va_weeks = blocks[k + 1]
        tr_idx = np.where(train["week"].isin(tr_weeks))[0]
        va_idx = np.where(train["week"].isin(va_weeks))[0]
        folds.append(
            {
                "fold": k + 1,
                "train_weeks": [int(tr_weeks.min()), int(tr_weeks.max())],
                "valid_weeks": [int(va_weeks.min()), int(va_weeks.max())],
                "train_idx": tr_idx,
                "valid_idx": va_idx,
            }
        )
    return folds


def cap_rows(idx: np.ndarray, cap: int, seed: int = 42) -> np.ndarray:
    """Random subsample of fold rows down to ``cap`` (documented compute budget)."""
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=cap, replace=False))
