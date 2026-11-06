"""Stability diagnostics: adversarial validation, PSI/CSI, monitoring triggers.

The premise of this project is that a credit model is a *time series product*:
the coefficients are re-estimated rarely, so the population a scorecard meets
two years after development is not the population it was fitted on.  These
diagnostics answer three questions in order:

1. can a model tell train rows from OOT rows at all (adversarial AUC)?
2. which features moved (PSI/CSI), and does removing them cost accuracy?
3. what score-level drift should trigger a redevelopment (monitoring plan)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hc_metrics import calibration_table, csi, psi, psi_band, weekly_gini


def adversarial_validation(
    X_train: pd.DataFrame,
    X_oot: pd.DataFrame,
    features: list[str],
    sample: int = 150_000,
    seed: int = 42,
) -> dict:
    """Train-vs-OOT classifier.

    AUC near 0.5 means the two cohorts are indistinguishable on the features -
    the scorecard is being tested on the same kind of applicant it was built
    for.  AUC well above 0.5 means the OOT metric is measuring a different
    population, and the drift table explains why.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    feats = [f for f in features if f in X_train.columns and f in X_oot.columns]
    take_tr = X_train[feats]
    take_oo = X_oot[feats]
    if len(take_tr) > sample:
        take_tr = take_tr.iloc[np.sort(rng.choice(len(take_tr), sample, replace=False))]
    if len(take_oo) > sample:
        take_oo = take_oo.iloc[np.sort(rng.choice(len(take_oo), sample, replace=False))]

    X = pd.concat([take_tr, take_oo], ignore_index=True)
    y = np.r_[np.zeros(len(take_tr)), np.ones(len(take_oo))]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, p))

    imp = (
        pd.DataFrame({"feature": feats, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["gain_share"] = imp["gain"] / max(imp["gain"].sum(), 1e-9)
    return {
        "auc": auc,
        "n_train_sample": int(len(take_tr)),
        "n_oot_sample": int(len(take_oo)),
        "top_drifting": imp.head(25).to_dict(orient="records"),
        "interpretation": (
            "cohorts indistinguishable" if auc < 0.60
            else "moderate drift - check the PSI table" if auc < 0.70
            else "strong drift - OOT metrics are a different-population estimate"
        ),
    }


def psi_table(
    X_train: pd.DataFrame,
    X_oot: pd.DataFrame,
    features: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per feature, train -> OOT, with the monitoring band."""
    rows = []
    for f in features:
        if f not in X_train.columns or f not in X_oot.columns:
            continue
        value = psi(X_train[f].to_numpy(), X_oot[f].to_numpy(), bins=bins)
        rows.append({"feature": f, "psi": value, "band": psi_band(value)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


def csi_table(
    train: pd.DataFrame,
    oot: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """CSI for raw categorical columns (before any numeric encoding)."""
    rows = []
    for c in columns:
        if c not in train.columns or c not in oot.columns:
            continue
        value = csi(train[c], oot[c])
        rows.append({"feature": c, "csi": value, "band": psi_band(value)})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


def score_drift(
    p_train: np.ndarray,
    p_oot: np.ndarray,
    week_oot: np.ndarray,
    y_oot: np.ndarray,
    bins: int = 10,
    min_cases: int = 150,
) -> dict:
    """Score-level PSI and per-week calibration drift on OOT.

    Two separate questions: has the *distribution* of PDs moved (PSI), and is the
    PD still the right number for the population that arrived (mean predicted vs
    observed by week)?
    """
    overall = psi(p_train, p_oot, bins=bins)
    weekly = []
    for w in np.unique(week_oot):
        mask = week_oot == w
        if mask.sum() < min_cases:
            continue
        weekly.append(
            {
                "week": int(w),
                "n": int(mask.sum()),
                "mean_pd": float(np.mean(p_oot[mask])),
                "observed": float(np.mean(y_oot[mask])),
                "gap": float(np.mean(p_oot[mask]) - np.mean(y_oot[mask])),
            }
        )
    return {
        "psi_train_to_oot": float(overall),
        "psi_band": psi_band(overall),
        "weekly_calibration": weekly,
    }


def monitoring_plan(
    psi_watch: float,
    psi_alert: float,
    gini_drop: float,
    min_cases: int,
    valid_stability: float,
    oot_stability: float,
) -> dict:
    """Human-readable trigger table for the model owner.

    Bands are the conventional Siddiqi PSI thresholds; the Gini trigger is the
    drop from the VALID stability figure that would have flagged the declines
    seen in this project's own OOT weeks.
    """
    return {
        "psi_bands": {
            "stable": f"PSI < {psi_watch:.2f} - no action, keep monitoring",
            "watch": f"{psi_watch:.2f} <= PSI < {psi_alert:.2f} - investigate drivers, no rebuild",
            "alert": f"PSI >= {psi_alert:.2f} - redevelop or recalibrate",
        },
        "gini_trigger": {
            "rule": f"weekly Gini more than {gini_drop:.2f} below the VALID mean for 2 consecutive weeks",
            "valid_mean_gini": float("nan") if valid_stability is None else valid_stability,
            "oot_mean_gini": float("nan") if oot_stability is None else oot_stability,
        },
        "minimum_weekly_volume": int(min_cases),
        "calibration_trigger": "mean predicted PD / observed default rate outside [0.8, 1.25] for a month",
        "data_quality_trigger": "PSI of any feature in the model above the alert band, or a new category above 5% share",
    }


def stability_panel(
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
    weeks: np.ndarray,
    min_cases: int = 50,
) -> dict:
    """Weekly Gini table for every model on one split."""
    return {name: weekly_gini(y, p, weeks, min_cases=min_cases) for name, p in predictions.items()}


def decile_calibration_by_period(
    y: np.ndarray,
    p: np.ndarray,
    weeks: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration deciles per OOT week, to see the level drift, not just the shape."""
    frames = []
    for w in np.unique(weeks):
        mask = weeks == w
        if mask.sum() < 300:
            continue
        tab = calibration_table(y[mask], p[mask], n_bins=n_bins)
        tab["week"] = int(w)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
