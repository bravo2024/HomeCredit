"""Metrics, the competition stability statistic, PSI and scorecard scaling.

The module is deliberately dependency-light (numpy, pandas, sklearn) and has no
project imports so that it can be inlined verbatim into the notebook.

Why a dedicated module: the stability metric is the *selection criterion* of
this project, so it must be implemented once, tested against a hand-computed
example, and reused unchanged for every model and every split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

EPS = 1e-6


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------
def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC.  Undefined (nan) when a slice has a single class."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def gini(y: np.ndarray, p: np.ndarray) -> float:
    """Gini = 2 * AUC - 1, the convention used by the competition."""
    a = auc(y, p)
    return float("nan") if np.isnan(a) else 2.0 * a - 1.0


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score for probabilities in [0, 1]."""
    return float(brier_score_loss(np.asarray(y), np.clip(p, 0.0, 1.0 - EPS)))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary log loss."""
    return float(log_loss(np.asarray(y), np.clip(p, EPS, 1.0 - EPS), labels=[0, 1]))


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between the score distributions."""
    y = np.asarray(y)
    order = np.argsort(p)
    y = y[order]
    n_bad = max(int(y.sum()), 1)
    n_good = max(int(len(y) - y.sum()), 1)
    cdf_bad = np.cumsum(y) / n_bad
    cdf_good = np.cumsum(1 - y) / n_good
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def topk_metrics(y: np.ndarray, p: np.ndarray, k: float) -> dict:
    """Recall and precision when the top ``k`` share of scores is reviewed."""
    y = np.asarray(y)
    n = len(y)
    m = max(int(round(k * n)), 1)
    idx = np.argsort(-p)[:m]
    hits = int(y[idx].sum())
    return {
        "capacity": k,
        "reviewed": m,
        "recall": hits / max(int(y.sum()), 1),
        "precision": hits / m,
    }


# --------------------------------------------------------------------------
# Weekly Gini and the official stability metric
# --------------------------------------------------------------------------
def weekly_gini(
    y: np.ndarray,
    p: np.ndarray,
    weeks: np.ndarray,
    min_cases: int = 50,
) -> pd.DataFrame:
    """Per-week Gini table.

    Weeks with fewer than ``min_cases`` rows or a single class are returned with
    ``gini = NaN`` and excluded from the stability fit, because a Gini computed
    on 20 applications is noise that would distort the slope penalty.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "week": np.asarray(weeks)})
    out = []
    for week, g in df.groupby("week", sort=True):
        n = len(g)
        value = gini(g["y"].to_numpy(), g["p"].to_numpy()) if n >= min_cases else float("nan")
        out.append(
            {
                "week": int(week),
                "n": int(n),
                "n_defaults": int(g["y"].sum()),
                "gini": value,
            }
        )
    return pd.DataFrame(out)


def stability_metric(
    weekly: pd.DataFrame,
    slope_penalty: float = 88.0,
    volatility_penalty: float = 0.5,
) -> dict:
    """Competition stability metric.

    ``metric = mean(gini) + slope_penalty * min(0, slope) - volatility_penalty * std(residuals)``

    A linear regression ``a*x + b`` is fitted through the weekly Ginis with
    ``x = WEEK_NUM``.  Because the evaluated weeks are consecutive integers the
    slope is numerically identical to the one obtained with ``x = 0..k-1``; the
    residual standard deviation is invariant to that shift by construction.
    Weeks with NaN Gini (too few cases, or one class only) are dropped and the
    number dropped is reported so a reader can judge how much of the series was
    actually fitted.
    """
    valid = weekly.dropna(subset=["gini"]).sort_values("week")
    if len(valid) < 3:
        return {
            "weeks_used": int(len(valid)),
            "mean_gini": float("nan"),
            "slope": float("nan"),
            "std_residual": float("nan"),
            "stability_metric": float("nan"),
        }
    x = valid["week"].to_numpy(dtype=float)
    g = valid["gini"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, g, 1)
    fitted = slope * x + intercept
    resid = g - fitted
    mean_gini = float(g.mean())
    return {
        "weeks_used": int(len(valid)),
        "weeks_total": int(len(weekly)),
        "mean_gini": mean_gini,
        "slope": float(slope),
        "std_residual": float(resid.std(ddof=1)) if len(resid) > 1 else 0.0,
        "min_zero_slope_term": float(slope_penalty * min(0.0, slope)),
        "volatility_term": float(
            -volatility_penalty * (resid.std(ddof=1) if len(resid) > 1 else 0.0)
        ),
        "stability_metric": float(
            mean_gini
            + slope_penalty * min(0.0, slope)
            - volatility_penalty * (resid.std(ddof=1) if len(resid) > 1 else 0.0)
        ),
    }


# --------------------------------------------------------------------------
# Distribution shift
# --------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from ``expected``.

    Bands used in the monitoring plan: < 0.10 stable, 0.10-0.25 watch,
    > 0.25 alert.  Empty bins are floored at 1e-4 so a hard-zero bin does not
    produce an infinite index.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(value: float) -> str:
    """Monitoring band label for a PSI/CSI value."""
    if not np.isfinite(value):
        return "unknown"
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "watch"
    return "alert"


def csi(expected: pd.Series, actual: pd.Series) -> float:
    """Characteristic Stability Index for a categorical feature."""
    e = expected.astype("object").fillna("__missing__").value_counts(normalize=True)
    a = actual.astype("object").fillna("__missing__").value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx).fillna(1e-4).clip(lower=1e-4)
    a = a.reindex(idx).fillna(1e-4).clip(lower=1e-4)
    return float(np.sum((a - e) * np.log(a / e)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count deciles of predicted PD with observed default rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    g["gap"] = g["observed"] - g["mean_pd"]
    return g


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict:
    """Expected/actual ratio and the calibration intercept-free slope.

    ``slope`` is the OLS coefficient of ``y`` on ``logit(p)`` restricted to
    [1e-6, 1-1e-6]; 1.0 means the spread of the scores matches the outcome.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    logit = np.log(p / (1 - p))
    if np.unique(logit).size < 2:
        return {
            "mean_predicted": float(p.mean()),
            "observed_rate": float(y.mean()),
            "expected_actual_ratio": float(p.mean() / max(y.mean(), EPS)),
            "calibration_slope": 0.0,
            "calibration_intercept": float(y.mean()),
        }
    slope, intercept = np.polyfit(logit, y, 1)
    return {
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "expected_actual_ratio": float(p.mean() / max(y.mean(), EPS)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str = "auc",
    n_boot: int = 300,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for a point metric.

    Resampling is at the application level, which is the unit of analysis, so
    the interval reflects application-level uncertainty only.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    fn = {"auc": auc, "gini": gini, "pr_auc": pr_auc, "brier": brier, "logloss": logloss}[metric]
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    vals = np.asarray(vals)
    point = fn(y, p)
    return {
        "metric": metric,
        "point": float(point),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(len(vals)),
    }


# --------------------------------------------------------------------------
# Scorecard scaling
# --------------------------------------------------------------------------
def pd_to_score(pd_: np.ndarray, pdo: float, base_score: float, base_odds: float) -> np.ndarray:
    """Map PD to a score where higher = safer.

    ``score = offset - factor * logit(PD)`` with ``factor = PDO / ln 2`` and
    ``offset = base_score - factor * ln(base_odds)``.  ``base_odds`` is
    good:bad, so a 20:1 base-odds score of 600 corresponds to PD = 1/21.
    """
    p = np.clip(np.asarray(pd_, dtype=float), EPS, 1 - EPS)
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset - factor * np.log(p / (1 - p))


def score_to_pd(score: np.ndarray, pdo: float, base_score: float, base_odds: float) -> np.ndarray:
    """Inverse of :func:`pd_to_score`."""
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    logit = (offset - np.asarray(score, dtype=float)) / factor
    return 1.0 / (1.0 + np.exp(-logit))
