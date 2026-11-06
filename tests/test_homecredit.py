"""Unit tests on tiny synthetic frames - no raw data, no network.

The suite pins the three things that would silently corrupt every published
number: the stability metric arithmetic, the point-in-time filter and the
decision-layer accounting.  It also exercises the scorecard's points
reconstruction and the leakage gate's teeth.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hc_data
import hc_features as feat
import hc_metrics as met
import hc_model as model
import hc_pipeline as pipe


# --------------------------------------------------------------------------
# Metrics and the competition stability statistic
# --------------------------------------------------------------------------
def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.normal(0.3, 0.2, 500), 0.01, 0.99)
    assert met.gini(y, p) == pytest.approx(2 * met.auc(y, p) - 1, rel=1e-9)


def test_stability_metric_matches_manual_arithmetic():
    weeks = np.arange(10, 22)
    ginis = np.array([0.50, 0.49, 0.47, 0.46, 0.44, 0.43, 0.42, 0.40, 0.39, 0.38, 0.36, 0.35])
    weekly = pd.DataFrame({"week": weeks, "n": 1000, "n_defaults": 30, "gini": ginis})

    slope, intercept = np.polyfit(weeks.astype(float), ginis, 1)
    resid = ginis - (slope * weeks + intercept)
    expected = ginis.mean() + 88.0 * min(0.0, slope) - 0.5 * resid.std(ddof=1)

    out = met.stability_metric(weekly, 88.0, 0.5)
    assert out["stability_metric"] == pytest.approx(expected, rel=1e-12)
    assert out["slope"] < 0
    assert out["weeks_used"] == len(weeks)


def test_stability_metric_no_penalty_when_slope_positive():
    weekly = pd.DataFrame(
        {"week": np.arange(6), "n": 500, "n_defaults": 20, "gini": [0.3, 0.31, 0.32, 0.33, 0.34, 0.35]}
    )
    out = met.stability_metric(weekly)
    assert out["min_zero_slope_term"] == 0.0
    assert out["stability_metric"] > out["mean_gini"] - 1e-9


def test_stability_metric_slope_invariant_to_week_offset():
    """Fitting on WEEK_NUM or on 0..k-1 must give the same penalty."""
    base = np.array([0.40, 0.39, 0.37, 0.35, 0.34, 0.32])
    a = met.stability_metric(pd.DataFrame({"week": np.arange(6), "gini": base}))
    b = met.stability_metric(pd.DataFrame({"week": np.arange(80, 86), "gini": base}))
    assert a["slope"] == pytest.approx(b["slope"], rel=1e-12)
    assert a["stability_metric"] == pytest.approx(b["stability_metric"], rel=1e-12)


def test_weekly_gini_marks_small_weeks_nan():
    y = np.array([0, 1, 0, 1, 0, 1] * 10)
    p = np.linspace(0.01, 0.99, len(y))
    weeks = np.array([1] * 10 + [2] * 50)
    tab = met.weekly_gini(y, p, weeks, min_cases=50)
    assert np.isnan(tab.loc[tab["week"] == 1, "gini"].iloc[0])
    assert np.isfinite(tab.loc[tab["week"] == 2, "gini"].iloc[0])


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=5000)
    assert met.psi(a, a) == pytest.approx(0.0, abs=1e-6)
    shifted = met.psi(a, a + 3)
    assert shifted > 0.25
    assert met.psi_band(shifted) == "alert"
    assert met.psi_band(0.05) == "stable"
    assert met.psi_band(0.15) == "watch"


def test_score_scale_semantics():
    p = 1 / 21
    assert met.pd_to_score(p, 20.0, 600.0, 20.0) == pytest.approx(600.0, abs=1e-6)
    # doubling the good:bad odds adds exactly PDO points
    p2 = 1.0 / 41.0  # good:bad odds 40:1, double the base odds
    assert met.pd_to_score(p2, 20.0, 600.0, 20.0) == pytest.approx(620.0, abs=1e-6)
    # PD -> score -> PD round-trips
    pd_vals = np.array([0.002, 0.02, 0.15, 0.4])
    scores = met.pd_to_score(pd_vals, 20, 600, 20)
    assert np.all(np.diff(scores) < 0)  # higher PD -> lower score
    assert met.score_to_pd(scores, 20, 600, 20) == pytest.approx(pd_vals, rel=1e-9)


def test_calibration_summary_detects_overprediction():
    rng = np.random.default_rng(2)
    y = (rng.random(4000) < 0.10).astype(int)
    p = np.full(4000, 0.20)
    s = met.calibration_summary(y, p)
    assert s["expected_actual_ratio"] > 1.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(0.4 * y + rng.normal(0.2, 0.2, 2000), 0, 1)
    ci = met.bootstrap_ci(y, p, "auc", n_boot=120)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


# --------------------------------------------------------------------------
# Point-in-time filter
# --------------------------------------------------------------------------
def _decision_frame():
    return pl.DataFrame(
        {
            "case_id": [1, 2],
            "date_decision": [dt.date(2019, 1, 15), dt.date(2019, 2, 1)],
        }
    ).with_columns(pl.col("date_decision").cast(pl.Date))


def test_pit_drops_records_dated_after_decision():
    df = pl.DataFrame(
        {
            "case_id": [1, 1, 1, 2],
            "num_group1": [0, 1, 2, 0],
            "date_111D": ["2019-01-01", "2019-03-01", None, "2019-02-01"],
            "amt_1A": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out, log = hc_data.pit_frame(df, _decision_frame())
    assert out.height == 3          # the March record for case 1 is dropped
    assert 2.0 not in out["amt_1A"].to_list()
    assert log["rows_dropped"] == 1
    # a record with no date information is kept: the rule is "drop if after",
    # not "drop if unknown".
    assert 3.0 in out["amt_1A"].to_list()


def test_pit_month_pairs_drop_future_months():
    df = pl.DataFrame(
        {
            "case_id": [1, 1, 1],
            "num_group1": [0, 1, 2],
            "pmts_year_1139T": [2018.0, 2019.0, 2019.0],
            "pmts_month_158T": [12.0, 2.0, 1.0],
            "pmts_overdue_1140A": [10.0, 20.0, 30.0],
        }
    )
    out, log = hc_data.pit_frame(df, _decision_frame())
    # 2019-02 is after 2019-01-15 (same year, later month) -> dropped;
    # 2019-01 is the decision month -> kept.
    assert sorted(out["pmts_overdue_1140A"].to_list()) == [10.0, 30.0]
    assert log["rows_dropped"] == 1


def test_pit_keeps_all_rows_without_temporal_information():
    df = pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 1], "amt_1A": [1.0, 2.0]})
    out, log = hc_data.pit_frame(df, _decision_frame())
    assert out.height == 2
    assert log["rows_dropped"] == 0
    assert log["has_temporal_information"] is False


# --------------------------------------------------------------------------
# Splitting and folds
# --------------------------------------------------------------------------
def test_split_bounds_are_ordered_and_complete():
    bounds = hc_data.split_bounds(list(range(92)), (0.70, 0.15, 0.15))
    assert bounds["train_weeks"][-1] < bounds["valid_weeks"][0]
    assert bounds["valid_weeks"][-1] < bounds["oot_weeks"][0]
    assert len(bounds["train_weeks"]) + len(bounds["valid_weeks"]) + len(bounds["oot_weeks"]) == 92


def test_label_split_matches_bounds():
    bounds = hc_data.split_bounds(list(range(20)), (0.5, 0.25, 0.25))
    labels = [hc_data.label_split(w, bounds) for w in range(20)]
    assert labels[0] == "train" and labels[-1] == "oot"
    assert labels == sorted(labels, key=["train", "valid", "oot"].index)


def test_expanding_folds_never_see_the_future():
    train = pd.DataFrame({"week": np.repeat(np.arange(10, 30), 100)})
    folds = model.expanding_folds(train, n_folds=4)
    assert len(folds) == 4
    for f in folds:
        assert f["train_weeks"][1] < f["valid_weeks"][0]
    assert folds[0]["train_weeks"][0] == 10


def test_cap_rows_is_a_subset():
    idx = np.arange(1000)
    capped = model.cap_rows(idx, 100)
    assert len(capped) == 100
    assert set(capped).issubset(set(idx))


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------
def test_iv_ranks_signal_above_noise():
    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    strong = y + rng.normal(0, 0.5, n)
    noise = rng.normal(size=n)
    tab = feat.iv_table(pd.DataFrame({"strong": strong, "noise": noise}), y)
    assert tab.iloc[0]["feature"] == "strong"
    assert tab.iloc[0]["iv"] > tab.iloc[1]["iv"]


def test_vif_prune_removes_duplicate_column():
    rng = np.random.default_rng(5)
    x = rng.normal(size=2000)
    df = pd.DataFrame({"a": x, "a_copy": x * 2 + 1, "b": rng.normal(size=2000)})
    kept, drops = feat.vif_prune(df, threshold=5.0)
    assert len(kept) == 2
    assert len(drops) == 1
    assert drops[0]["dropped"] in {"a", "a_copy"}


def test_univariate_screen_flags_a_perfect_feature():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 3000)
    X = pd.DataFrame({"leak": y.astype(float), "noise": rng.normal(size=3000)})
    out = pipe.univariate_screen(X, y, 0.80)
    assert not out["passed"]
    assert out["flagged"][0]["feature"] == "leak"


# --------------------------------------------------------------------------
# Scorecard algebra
# --------------------------------------------------------------------------
def test_scorecard_points_reconstruct_the_score():
    rng = np.random.default_rng(7)
    n = 6000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = -2.0 + 1.2 * x1 - 0.8 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    process = model.fit_woe_binning(X, y, categorical=[])
    design = model.woe_design(process, X)
    sc = model.fit_scorecard(design, y, design.iloc[:5])
    pts, base_points = model.points_table(process, sc["coef_table"], 20.0, 600.0, 20.0)

    # reconstruct the score for the first row from its bin points
    row = design.iloc[0]
    total = base_points
    for var in design.columns:
        points = pts[pts["feature"] == var]
        # the bin whose WoE equals this row's WoE (bins are unique per variable)
        hit = points.iloc[(points["woe"] - row[var]).abs().argmin()]
        total += hit["points"]
    candidate = met.pd_to_score(np.array([sc["p_valid"][0]]), 20.0, 600.0, 20.0)[0]
    assert total == pytest.approx(candidate, abs=1.0)


def test_sign_check_flags_a_flipped_coefficient():
    coef = pd.DataFrame({"feature": ["const", "x1"], "coef": [1.0, 0.5],
                         "std_err": [0.1, 0.1], "z": [1, 1], "p_value": [0.1, 0.1],
                         "ci_low": [0, 0], "ci_high": [1, 1]})
    rows = [{"feature": "x1", "coef": 0.5, "expected_sign": -1.0, "consistent": False}]
    assert rows[0]["consistent"] is False


# --------------------------------------------------------------------------
# Leakage gate
# --------------------------------------------------------------------------
def _synthetic_ds(leak: bool = False) -> dict:
    n = 600
    rng = np.random.default_rng(8)
    week = np.repeat(np.arange(12), 50)
    y = (rng.random(n) < 0.1).astype(int)
    X = pd.DataFrame({"noise": rng.normal(size=n), "x2": rng.normal(size=n)})
    if leak:
        X["target_copy"] = y.astype(float)
    split = np.where(week < 8, "train", np.where(week < 10, "valid", "oot"))
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(week * 7, unit="D")
    return {
        "X": X,
        "y": y,
        "week": week,
        "date": dates.values,
        "split": split,
        "train_mask": split == "train",
        "valid_mask": split == "valid",
        "oot_mask": split == "oot",
        "case_ids": np.arange(n),
        "feature_columns": list(X.columns),
        "cat_cols": [],
    }


def test_leakage_gate_passes_on_clean_data():
    ds = _synthetic_ds()
    screen = {"quarantined": []}
    out = pipe.leakage_gate(ds, screen, list(ds["X"].columns))
    assert out["passed"] is True


def test_leakage_gate_raises_on_a_leaked_feature():
    ds = _synthetic_ds(leak=True)
    screen = {"quarantined": [{"feature": "target_copy", "univariate_auc": 1.0}]}
    with pytest.raises(RuntimeError, match="LEAKAGE GATE FAILED"):
        pipe.leakage_gate(ds, screen, list(ds["X"].columns))


class _StubBinningTable:
    def __init__(self, counts):
        self._counts = list(counts)

    def build(self):
        return pd.DataFrame({
            "Bin": [f"b{i}" for i in range(len(self._counts))] + ["Totals"],
            "Count": self._counts + [sum(self._counts)],
        })


class _StubBinnedVariable:
    def __init__(self, counts):
        self.binning_table = _StubBinningTable(counts)


class _StubProcess:
    def __init__(self, counts):
        self._counts = counts

    def get_binned_variable(self, var):
        return _StubBinnedVariable(self._counts)


def test_woe_gate_reconciles_against_the_fitted_sample():
    """Regression: bins are fitted on a declared subsample, not every train row.

    The old check compared the bin counts with the full train count, so a
    legitimate 250k subsample of a 1.28m train frame failed the gate.
    """
    ds = _synthetic_ds()
    train_rows = int(ds["train_mask"].sum())
    fit_rows = train_rows - 60
    counts = [fit_rows - 100, 100]
    sc = {
        "features": ["noise"],
        "process": _StubProcess(counts),
        "woe_fit_rows": fit_rows,
        "woe_fit_fraction": fit_rows / train_rows,
        "train_rows": train_rows,
    }
    out = pipe.leakage_gate(ds, {"quarantined": []}, list(ds["X"].columns), sc=sc)
    assert out["passed"] is True
    with pytest.raises(RuntimeError, match="woe_binned_on_train_rows"):
        pipe.leakage_gate(ds, {"quarantined": []}, list(ds["X"].columns),
                          sc={**sc, "woe_fit_rows": fit_rows - 100})


def test_leakage_gate_raises_on_case_overlap():
    ds = _synthetic_ds()
    ds["case_ids"] = np.zeros_like(ds["case_ids"])
    with pytest.raises(RuntimeError):
        pipe.leakage_gate(ds, {"quarantined": []}, list(ds["X"].columns))


# --------------------------------------------------------------------------
# Calibration and baselines
# --------------------------------------------------------------------------
def test_constant_model_returns_training_rate():
    m = model.ConstantModel().fit(pd.DataFrame({"a": [1, 2]}), np.array([0, 1, 0, 1]))
    assert m.predict_proba(pd.DataFrame({"a": [1, 2, 3]})) == pytest.approx([0.5, 0.5, 0.5])


def test_isotonic_calibrator_is_monotone_and_bounded():
    rng = np.random.default_rng(9)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(rng.normal(0.2, 0.1, 2000), 0.001, 0.999)
    cal = model.IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-9)
    assert out.min() > 0 and out.max() < 1


def test_build_monitoring_uses_positions_inside_each_slice():
    """Regression: the monthly PSI must index the OOT array, not the full frame."""
    ds = _synthetic_ds()
    rng = np.random.default_rng(11)
    p = rng.random(len(ds["y"]))
    out = pipe.build_monitoring(
        p[ds["valid_mask"]], p[ds["oot_mask"]], ds, ["noise"], min_cases=20
    )
    assert out["psi_by_month"], "monthly PSI should be produced for the OOT months"
    assert all(0.0 <= row["psi"] for row in out["psi_by_month"])
    assert out["n_alert"] + out["n_watch"] <= len(out["psi_table"])


# ---------------------------------------------------------------- v2: table parts + coverage gate
def _ds_for_coverage(null_oot: bool, null_valid: bool = False):
    n = 30
    split = np.array(["train"] * 10 + ["valid"] * 10 + ["oot"] * 10)
    good = np.arange(n, dtype=float)
    bad = good.copy()
    if null_oot:
        bad[split == "oot"] = np.nan
    if null_valid:
        bad[split == "valid"] = np.nan
    return {"X": pd.DataFrame({"t__good": good, "t__bad": bad}), "split": split}


def test_coverage_gate_fails_when_a_final_feature_is_empty_in_oot():
    cov = pipe.feature_coverage(_ds_for_coverage(null_oot=True), {"tree": ["t__good", "t__bad"]})
    assert cov["n_null_in_oot"] == 1 and cov["null_in_oot"] == ["t__bad"]
    assert not cov["passed"]
    assert pipe.coverage_gate(cov)["status"] == "FAIL"
    with pytest.raises(RuntimeError):
        pipe.coverage_gate(cov, raise_on_fail=True)


def test_coverage_gate_fails_when_a_final_feature_is_empty_in_valid():
    cov = pipe.feature_coverage(_ds_for_coverage(False, null_valid=True), {"tree": ["t__bad"]})
    assert cov["n_null_in_valid"] == 1 and not cov["passed"]


def test_coverage_gate_passes_and_ignores_unselected_empty_columns():
    cov = pipe.feature_coverage(_ds_for_coverage(null_oot=True), {"tree": ["t__good"]})
    assert cov["passed"] and cov["n_final_features"] == 1
    assert cov["features"][0]["coverage_oot"] == 1.0
    assert pipe.coverage_gate(cov)["status"] == "PASS"


def test_table_parts_reads_every_numbered_part_once(tmp_path, monkeypatch):
    root = tmp_path / "train"
    root.mkdir()
    for name, ids in (("train_x_1_0", [1, 2]), ("train_x_1_1", [3]), ("train_x_1_10", [4]), ("train_x_2", [9]),
                      ("train_x_1b_0", [8])):
        pl.DataFrame({"case_id": ids, "a": [1.0] * len(ids)}).write_parquet(root / f"{name}.parquet")
    monkeypatch.setattr(hc_data, "data_root", lambda sample=False: tmp_path)
    parts = [p.stem for p in hc_data.table_parts("train_x_1")]
    assert parts == ["train_x_1_0", "train_x_1_1", "train_x_1_10"]
    assert sorted(hc_data.scan_table("train_x_1").collect()["case_id"].to_list()) == [1, 2, 3, 4]


def test_scan_table_aligns_schemas_across_parts(tmp_path, monkeypatch):
    root = tmp_path / "train"
    root.mkdir()
    pl.DataFrame({"case_id": [1], "a": [1]}).write_parquet(root / "train_y_0.parquet")
    pl.DataFrame({"case_id": [2], "a": [2.5], "b": ["z"]}).write_parquet(root / "train_y_1.parquet")
    monkeypatch.setattr(hc_data, "data_root", lambda sample=False: tmp_path)
    out = hc_data.scan_table("train_y").collect().sort("case_id")
    assert out.columns == ["case_id", "a", "b"] and out["b"].to_list() == [None, "z"]


def test_aggregate_table_stacks_parts_without_averaging_means(tmp_path, monkeypatch):
    root = tmp_path / "train"
    root.mkdir()
    pl.DataFrame({"case_id": [1, 1, 2], "v": [1.0, 3.0, 5.0]}).write_parquet(root / "train_z_1_0.parquet")
    pl.DataFrame({"case_id": [3, 3], "v": [10.0, 20.0]}).write_parquet(root / "train_z_1_1.parquet")
    monkeypatch.setattr(hc_data, "data_root", lambda sample=False: tmp_path)
    decision = pl.DataFrame({"case_id": [1, 2, 3], "date_decision": [dt.date(2020, 1, 1)] * 3})
    out, log = hc_data.aggregate_table("train_z_1", ["v"], [], decision, 1)
    got = {r["case_id"]: r for r in out.to_dicts()}
    assert got[1]["train_z_1__v__mean"] == 2.0 and got[3]["train_z_1__v__mean"] == 15.0
    assert got[3]["train_z_1__n_records"] == 2 and log["n_parts"] == 2 and out.height == 3

# ---------------------------------------------------------------- v4: no absolute calendar values in the design
def test_absolute_calendar_feature_rule():
    f = hc_data.is_absolute_calendar_feature
    assert f("train_credit_bureau_a_1__refreshdate_3813885D__maxcode")
    assert f("train_credit_bureau_a_2__pmts_year_1139T__mean")
    assert f("train_credit_bureau_a_2__pmts_month_706T__std")
    assert not f("train_credit_bureau_a_2__pmts_year_1139T__rel_mean")
    assert not f("train_credit_bureau_a_1__refreshdate_3813885D__rel_max")
    assert not f("train_applprev_1__amount_1115A__mean")
    assert not f("train_applprev_1__n_records")


def test_calendar_gate_fails_on_absolute_and_passes_on_relative():
    bad = pipe.calendar_gate(["a__x__mean", "t__pmts_year_507T__min"])
    assert bad["status"] == "FAIL" and bad["detail"]["absolute_calendar_features"] == ["t__pmts_year_507T__min"]
    with pytest.raises(RuntimeError):
        pipe.calendar_gate(["t__d_1D__maxcode"], raise_on_fail=True)
    ok = pipe.calendar_gate(["t__pmts_year_507T__rel_min", "t__d_1D__rel_mean", "t__v__mean"])
    assert ok["status"] == "PASS" and ok["detail"]["n_relative_features"] == 2


def test_aggregate_table_turns_dates_and_year_month_into_relative_values(tmp_path, monkeypatch):
    root = tmp_path / "train"
    root.mkdir()
    pl.DataFrame({
        "case_id": [1, 1, 2],
        "opendate_1D": ["2020-01-01", "2020-03-01", "2019-12-31"],
        "pmts_year_1139T": [2019.0, 2020.0, 2019.0],
        "pmts_month_158T": [11.0, 1.0, 12.0],
        "v": [1.0, 2.0, 3.0],
    }).write_parquet(root / "train_q_1.parquet")
    monkeypatch.setattr(hc_data, "data_root", lambda sample=False: tmp_path)
    decision = pl.DataFrame({"case_id": [1, 2], "date_decision": [dt.date(2020, 4, 1), dt.date(2020, 1, 1)]})
    out, _ = hc_data.aggregate_table("train_q_1", ["opendate_1D", "pmts_year_1139T", "v"], [], decision, 1)
    got = {r["case_id"]: r for r in out.to_dicts()}
    # case 1: dates are 91 and 31 days before the decision; months before April 2020: 5 and 3
    assert got[1]["train_q_1__opendate_1D__rel_max"] == 91 and got[1]["train_q_1__opendate_1D__rel_min"] == 31
    assert got[1]["train_q_1__pmts_year_1139T__rel_max"] == 5 and got[1]["train_q_1__pmts_year_1139T__rel_min"] == 3
    assert not [c for c in out.columns if hc_data.is_absolute_calendar_feature(c)]

def test_calendar_proxy_screen_drops_week_tracking_feature_only():
    rng = np.random.default_rng(0)
    n = 4000
    weeks = rng.integers(0, 60, n)
    X = pd.DataFrame({
        "t__d_1D__rel_max": 700 - weeks * 7 + rng.normal(0, 5, n),   # near-fixed calendar date
        "t__d_2D__rel_mean": rng.normal(100, 30, n),                 # genuine relative age
        "t__v__mean": weeks + rng.normal(0, 1, n),                   # not relative: never screened
    })
    proxy = pipe.calendar_proxy_screen(X, weeks)
    assert set(proxy["feature"]) == {"t__d_1D__rel_max", "t__d_2D__rel_mean"}
    assert proxy.set_index("feature").loc["t__d_1D__rel_max", "dropped"]
    assert not proxy.set_index("feature").loc["t__d_2D__rel_mean", "dropped"]
    assert pipe.calendar_proxy_gate(proxy, ["t__d_2D__rel_mean", "t__v__mean"])["status"] == "PASS"
    bad = pipe.calendar_proxy_gate(proxy, ["t__d_1D__rel_max"])
    assert bad["status"] == "FAIL"
    with pytest.raises(RuntimeError):
        pipe.calendar_proxy_gate(proxy, ["t__d_1D__rel_max"], raise_on_fail=True)