"""Feature screening for the interpretable track: IV, VIF, monotone WoE, scaling.

The scorecard needs a *small, stable, non-collinear* design.  This module holds
the three statistical screens that get it there, each justified:

* **IV** (information value) - ranks raw variables by how much of the default
  signal they carry; the classic Siddiqi filter (IV < 0.02 = useless).
* **VIF** - a logistic scorecard with collinear variables has unstable
  coefficients, and unstable coefficients are what makes a scorecard's points
  table drift between vintages.  Pruning stops at VIF <= 5.
* **Standardisation** - L1/L2/ElasticNet penalties are only comparable across
  variables when the design is on a common scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


# --------------------------------------------------------------------------
# Information value
# --------------------------------------------------------------------------
def quantile_woe_iv(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """IV and the bin table for one numeric variable.

    Bins are quantiles of the observed values, with an explicit missing bin so
    that "no record" is treated as its own level rather than silently imputed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    missing = ~np.isfinite(x)
    xv = x[~missing]
    yv = y[~missing]
    if len(np.unique(xv)) < 2:
        return 0.0, pd.DataFrame()

    edges = np.unique(np.quantile(xv, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0, pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(xv, edges[1:-1], right=True)
    df = pd.DataFrame({"bin": b, "y": yv})
    rows = []
    if missing.any():
        df = pd.concat([df, pd.DataFrame({"bin": -1, "y": y[missing]})], ignore_index=True)

    total_bad = max(y.sum(), 1)
    total_good = max(len(y) - y.sum(), 1)
    iv = 0.0
    for bin_id, g in df.groupby("bin"):
        bad = int(g["y"].sum())
        good = int(len(g) - bad)
        dist_bad = (bad + 0.5) / (total_bad + 0.5)
        dist_good = (good + 0.5) / (total_good + 0.5)
        woe = np.log(dist_bad / dist_good)
        contribution = (dist_bad - dist_good) * woe
        iv += contribution
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(g)),
                "n_bad": bad,
                "bad_rate": bad / max(len(g), 1),
                "woe": float(woe),
                "iv_contribution": float(contribution),
            }
        )
    return float(iv), pd.DataFrame(rows)


def iv_table(
    X: pd.DataFrame,
    y: np.ndarray,
    max_features: int | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """IV for every column of a design frame, sorted descending.

    ``max_features`` caps the work on very wide frames; because the columns are
    already ordered by the table-level univariate screen, the cap is a prefix
    cut and not a random subset.
    """
    cols = list(X.columns)[: max_features or X.shape[1]]
    rows = []
    for c in cols:
        iv, _ = quantile_woe_iv(X[c].to_numpy(), y, n_bins=n_bins)
        rows.append({"feature": c, "iv": iv})
    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    out["iv_band"] = pd.cut(
        out["iv"],
        bins=[-0.01, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["useless", "weak", "medium", "strong", "suspicious"],
    ).astype(str)
    return out


# --------------------------------------------------------------------------
# Multicollinearity
# --------------------------------------------------------------------------
def vif_from_corr(df: pd.DataFrame, max_rows: int = 100_000) -> pd.Series:
    """VIF per column via the auxiliary-regression definition.

    ``VIF_j = 1 / (1 - R2_j)`` where ``R2_j`` comes from regressing column *j*
    on all the others.  The matrix form ``diag(R^-1)`` is equivalent for a
    well-conditioned design but breaks down exactly when it matters: with two
    near-duplicate columns the correlation matrix is singular and the
    pseudo-inverse returns a *small* value instead of the infinite inflation the
    analyst needs to see.  The regression form degrades to a huge number, which
    is the behaviour VIF pruning depends on.

    Rows are capped because VIF only needs the correlation structure; the cap
    keeps iterative pruning inside a few seconds per round.
    """
    x = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0)
    n, p = x.shape
    if p <= 1:
        return pd.Series([1.0] * p, index=df.columns)
    if n > max_rows:
        x = x[np.linspace(0, n - 1, max_rows).astype(int)]
        n = x.shape[0]
    std = x.std(axis=0)
    keep = std > 0
    x = x[:, keep]
    if x.shape[1] == 0:
        return pd.Series([1.0] * p, index=df.columns)
    x = x - x.mean(axis=0)

    out = np.full(x.shape[1], np.inf)
    for j in range(x.shape[1]):
        others = np.delete(x, j, axis=1)
        y = x[:, j]
        ss_tot = float(y @ y)
        if ss_tot <= 0:
            out[j] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        r2 = 1.0 - float(resid @ resid) / ss_tot
        out[j] = 1.0 / max(1e-9, 1.0 - min(r2, 1.0 - 1e-12))

    full = pd.Series(1.0, index=df.columns)
    full.iloc[np.where(keep)[0]] = out
    return full


def vif_prune(
    df: pd.DataFrame,
    threshold: float = 5.0,
    max_iter: int = 60,
) -> tuple[list[str], list[dict]]:
    """Iteratively drop the worst VIF until every variable is below threshold."""
    kept = list(df.columns)
    drops = []
    for _ in range(max_iter):
        if len(kept) <= 1:
            break
        vif = vif_from_corr(df[kept])
        worst = vif.idxmax()
        if vif.max() <= threshold:
            break
        drops.append({"dropped": worst, "vif": float(vif.max()), "round": len(drops) + 1})
        kept.remove(worst)
    return kept, drops


def condition_number(df: pd.DataFrame) -> float:
    """Condition number of the standardised design (30+ = unstable)."""
    x = df.to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    std = x.std(axis=0)
    x = x[:, std > 0]
    if x.shape[1] == 0:
        return float("nan")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return float(np.linalg.cond(x))


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------
def linear_pipeline() -> Pipeline:
    """Median impute + standardise.

    Both steps live in a Pipeline so they are fitted inside the training fold
    and never on validation or OOT rows.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )


def drop_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
    max_rows: int = 50_000,
) -> tuple[list[str], list[dict]]:
    """Greedy removal of near-duplicate columns (|r| above ``threshold``).

    A penalised logistic fitted on hundreds of highly correlated aggregates
    converges slowly and its coefficients are not identified: saga then spends
    its whole iteration budget on a flat objective. Removing near-duplicates
    before the fit keeps the design well conditioned and the coefficients
    interpretable, and it costs almost nothing in accuracy because the pair
    carries one signal twice.
    """
    sample = df.iloc[:max_rows] if len(df) > max_rows else df
    corr = sample.corr().abs()
    keep: list[str] = []
    dropped: list[dict] = []
    for col in df.columns:
        if not keep:
            keep.append(col)
            continue
        worst = max((corr.loc[col, k] for k in keep), default=0.0)
        if worst >= threshold:
            dropped.append({"column": col, "max_abs_r": float(worst)})
        else:
            keep.append(col)
    return keep, dropped


def correlation_clusters(df: pd.DataFrame, threshold: float = 0.85) -> list[list[str]]:
    """Greedy groups of variables correlated above ``threshold``.

    Used only for reporting: VIF pruning already removes the redundant
    variables, but the cluster table tells a reader *which* economic quantity
    was being measured twice.
    """
    corr = df.corr().abs()
    remaining = list(df.columns)
    clusters = []
    while remaining:
        head = remaining.pop(0)
        group = [head]
        for other in list(remaining):
            if corr.loc[head, other] >= threshold:
                group.append(other)
                remaining.remove(other)
        clusters.append(group)
    return [c for c in clusters if len(c) > 1]
