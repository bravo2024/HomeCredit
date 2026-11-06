"""Headless Streamlit tests on *synthetic* artifacts.

The app is contract-tested against a tiny generated artifact bundle so the tests
never need the Kaggle outputs: if a page, the scorer or the CSV schema check
breaks, this fails before a deployment does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
FEATS = [f"f{i}" for i in range(6)]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("artifacts")
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, len(FEATS))), columns=FEATS)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * X["f0"]))).astype(int)

    import lightgbm as lgb

    model = lgb.LGBMClassifier(n_estimators=25, verbose=-1).fit(X, y)
    (d / "models").mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(d / "models" / "champion_lgbm.txt"))

    stability = {"mean_gini": 0.41, "slope": -0.001, "std_residual": 0.02, "stability_metric": 0.37}
    _write(
        d / "metrics.json",
        {
            "champion": "lightgbm",
            "champion_rule": "max stability_metric on VALID weeks",
            "models": {
                "lightgbm": {
                    "valid": {"auc": 0.72, "gini": 0.44, "pr_auc": 0.2, "brier": 0.03,
                              "logloss": 0.1, "ks": 0.3, "stability": stability},
                    "oot": {"auc": 0.71, "gini": 0.42, "pr_auc": 0.19, "brier": 0.032,
                            "logloss": 0.11, "ks": 0.29, "stability": stability},
                }
            },
        },
    )
    _write(d / "split.json", {"table": [
        {"split": "train", "n": 1000, "default_rate": 0.031, "week_min": 0, "week_max": 63},
        {"split": "valid", "n": 200, "default_rate": 0.030, "week_min": 64, "week_max": 77},
        {"split": "oot", "n": 200, "default_rate": 0.028, "week_min": 78, "week_max": 91},
    ]})
    _write(d / "feature_meta.json", {
        "champion": "lightgbm",
        "kind": "lgbm",
        "features": FEATS,
        "cat_features": [],
        "model_file": "models/champion_lgbm.txt",
        "scorecard_file": None,
        "scorecard_features": [],
        "typical_values": {f: {"p05": -1.5, "p50": 0.0, "p95": 1.5, "missing": 0.0} for f in FEATS},
    })
    _write(d / "champion.json", {"champion": "lightgbm", "cutoffs_valid_raw_pd": {"xgboost": {"approval_80": 0.05}}})
    _write(d / "business.json", {
        "cutoff_strategies_oot": [{"strategy": "approve all", "model": "-", "approval_rate": 1.0, "bad_rate_approved": 0.02}],
        "capture": [{"model": "xgboost", "caught_at_5pct": 0.2}],
        "swap_sets": {"approval_rate": 0.8, "both_approve": {"n": 10, "bad_rate": 0.01}},
        "master_scale": [{"grade": 1, "grade_pd_valid": 0.01, "oot_default_rate": 0.009}],
        "expected_loss_illustrative": {"lgd_ASSUMED": 0.45, "expected_loss_total": 100.0, "realised_loss_total": 90.0},
    })
    _write(d / "evaluation.json", {
        "calibration": [{"bin": 0, "n": 30, "mean_pd": 0.01, "observed": 0.012, "gap": 0.002}],
        "shap": {"top_features": [{"feature": "f0", "mean_abs_shap": 0.4}]},
    })
    _write(d / "monitoring.json", {
        "monitoring_plan": {
            "psi_bands": {"stable": "PSI < 0.10", "watch": "0.10-0.25", "alert": ">= 0.25"},
            "gini_trigger": {"rule": "two consecutive weeks below the reference", "valid_mean_gini": 0.41},
            "minimum_weekly_volume": 150,
        },
        "score_drift": {"psi_train_to_oot": 0.03, "psi_band": "stable"},
    })
    _write(d / "quality_gates.json", {"gates": [{"gate": "leakage", "status": "PASS", "detail": {}}]})
    _write(d / "environment.json", {"sklearn": "1.6.1", "lightgbm": "4.6.0"})
    _write(d / "curves.json", {
        "roc": {"lightgbm": [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 0.1, "tpr": 0.5}, {"fpr": 1.0, "tpr": 1.0}]},
        "pr": {"lightgbm": [{"recall": 0.0, "precision": 0.2}, {"recall": 1.0, "precision": 0.03}]},
    })
    _write(d / "weekly_gini.json", {"lightgbm": [
        {"week": 78 + i, "n": 200, "gini": 0.40 - 0.001 * i} for i in range(14)
    ]})

    pd.DataFrame({
        "model": ["lightgbm", "woe_scorecard"],
        "valid_auc": [0.72, 0.68], "valid_pr_auc": [0.20, 0.18], "valid_brier": [0.03, 0.031],
        "valid_stability": [0.37, 0.31], "oot_auc": [0.71, 0.67], "oot_gini": [0.42, 0.34],
        "oot_pr_auc": [0.19, 0.17], "oot_brier": [0.032, 0.033], "oot_logloss": [0.11, 0.115],
        "oot_stability": [0.36, 0.30],
    }).to_csv(d / "metrics_table.csv", index=False)
    pd.DataFrame({
        "model": ["lightgbm", "lightgbm"], "split": ["valid", "oot"],
        "mean_gini": [0.41, 0.40], "slope": [-0.001, -0.002],
        "std_residual": [0.02, 0.03], "stability_metric": [0.37, 0.36],
    }).to_csv(d / "stability.csv", index=False)
    pd.DataFrame({"feature": ["f0", "f1"], "psi": [0.03, 0.40], "band": ["stable", "alert"]}).to_csv(
        d / "psi.csv", index=False
    )
    return d


@pytest.fixture()
def app(artifact_dir, monkeypatch):
    monkeypatch.setenv("HC_ARTIFACT_DIR", str(artifact_dir))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def test_overview_renders(app):
    assert not app.exception
    assert any("Champion" in m.label for m in app.metric)


@pytest.mark.parametrize(
    "page",
    ["Model comparison", "Stability & monitoring", "Score a case", "Batch scoring",
     "Credit policy", "Model card"],
)
def test_every_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"page {page} raised"


def test_scorer_returns_a_pd(app):
    app.sidebar.radio[0].set_value("Score a case").run()
    assert not app.exception
    assert app.button, "the score form should expose a submit button"
    app.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "PD" in labels and "Score" in labels and "Cut-off band" in labels


def test_batch_scoring_accepts_a_small_csv(app):
    app.sidebar.radio[0].set_value("Batch scoring").run()
    assert not app.exception
    header = "case_id," + ",".join(FEATS)
    row = "1," + ",".join("0.5" for _ in FEATS)
    app.text_area[0].set_value(header + "\n" + row + "\n2," + ",".join("-0.2" for _ in FEATS))
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) >= 1
    assert "pd" in app.dataframe[0].value.columns


def test_batch_scoring_rejects_an_unknown_column(app):
    app.sidebar.radio[0].set_value("Batch scoring").run()
    app.text_area[0].set_value("case_id,unexpected_column\n1,5")
    app.button[0].click().run()
    assert not app.exception
    assert app.error, "a schema violation must surface as an error, not a silent score"


def test_missing_artifacts_do_not_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HC_ARTIFACT_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.warning, "an empty artifact directory must produce a visible warning"
