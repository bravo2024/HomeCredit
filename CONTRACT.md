# CONTRACT — HomeCreditStability

This file is the build contract: what the pipeline promises, where every number comes from, and what must
hold before the project is considered done. It is written before the model results exist so the results
cannot redefine the rules.

## 1. Problem contract

| Item | Definition |
|---|---|
| Unit of analysis | `case_id` — one credit application |
| Prediction time | `date_decision` (the day the approval decision was made) |
| Target | competition `target` (default on that credit case, observed after a certain period) |
| Feature window | any record dated on or before `date_decision` |
| Split | by `WEEK_NUM`, ~70% / 15% / 15% of consecutive weeks, cut-offs computed from the data |
| Tuning | expanding-window folds **inside** train; VALID is used only to pick the champion and the policy |
| Final evaluation | OOT weeks, evaluated once |
| Selection rule | competition stability metric on VALID: `mean(weekly Gini) + 88·min(0, slope) − 0.5·std(residuals)` |

## 2. Data contract

* Source: Kaggle `home-credit-credit-risk-model-stability`, parquet files, train split only (the test split
  is unlabelled). Retrieval path and hashes: `data/PROVENANCE.md`, `artifacts/data_provenance.json`.
* `data/raw` is read-only. Nothing derived is written back into it.
* Point-in-time rule: a history record is dropped when its latest attributable date (column suffix `D`) or
  `(year, month)` pair (suffix `T`) is after the application's `date_decision`. Records with no date are
  kept. Every table reports `rows_in / rows_kept / rows_dropped`.
* No table is loaded into pandas before aggregation; everything is a polars lazy scan.

## 2b. Table-parts contract (v2)

Every logical table is read once from **all** its numbered part files (`table_parts()` in `src/hc_data.py`):
`train_static_0_*`, `train_applprev_1_*`, `train_credit_bureau_a_1_*`, `train_credit_bureau_a_2_*`, plus the
single-file tables. v1 listed only some parts, which left the later weeks empty. Parts are aggregated one at a time
and stacked (a `case_id` lives in exactly one part; the loader raises if not). Schemas are aligned (missing columns
become null, dtypes relaxed).

**Coverage gate (hard failure):** the run stops if any final model feature has zero non-null rows in VALID or OOT.
Candidate selection uses TRAIN rows only: a feature needs >= 1% non-null in the last 8 TRAIN weeks.
`artifacts/feature_coverage.json` holds the non-null share per feature and split.

## 3. Leakage gate (hard failure)

The run stops (`RuntimeError`) if any of these fail:

1. no identifier / outcome column in the feature list (`case_id`, `num_group*`, `target`, week/month,
   dates, and raw zip/pincode/name/url columns; aggregates over those columns are allowed);
2. `target` never appears in a design matrix;
3. strict time ordering — `max(train date) < min(valid date)` and `max(valid date) < min(oot date)`;
4. no `case_id` appears in more than one split;
5. every single feature's own univariate AUC is below 0.80, or the feature is quarantined before selection;
6. the linear design's scaler is fitted on training rows only (its means match the imputed train means
   better than the full-data means);
7. the WoE bins were fitted on exactly the training sample (bin counts sum to the fit-sample size).

## 4. Artifact contract (`artifacts/`)

| File | Written by | Content |
|---|---|---|
| `environment.json` | notebook | python + library versions used to fit the saved models |
| `data_provenance.json` | notebook | path, bytes, sha256 of every parquet consumed |
| `split.json` | notebook | week cut-offs, per-split n / default rate / weeks / date range |
| `pit_audit.json` | notebook | rows in / kept / dropped per table and why |
| `feature_screening.json` | notebook | per-table column screening (coverage, univariate AUC) |
| `iv_table.csv` | notebook | IV per selected feature |
| `leakage_gate.json` | notebook | every gate check with PASS/FAIL and the offending values |
| `scorecard.json`, `scorecard_coefficients.csv`, `scorecard_points.csv`, `vif_pruning.csv`, `sign_check.csv` | notebook | WoE scorecard: coefficients with SE/z/p/CI, points table, VIF path, sign audit |
| `cv.json` | not produced in the final run | the expanding-window CV (`cv_roster`) exists in `src/hc_pipeline.py` but the final kernel tunes on VALID (`tuning.json`) instead |
| `metrics.json`, `metrics_table.csv` | notebook | every model's VALID/OOT metrics, weekly series, champion |
| `curves.json` | notebook | ROC/PR curve points for the comparison page |
| `weekly_gini.json`, `stability.csv` | notebook | weekly Gini series and stability metric per split |
| `drift.json`, `psi.csv`, `drift_experiment.json` | notebook | adversarial validation, PSI/CSI, drift-removal trade-off |
| `evaluation.json`, `subgroups.csv` | notebook | calibration, bootstrap CI, subgroups, SHAP, reason codes |
| `decision.json`, `cutoff_strategies.csv`, `business.json` | notebook | cut-offs frozen on VALID, observed OOT approval rate / bad rate, swap sets, master scale (expected loss is illustrative: LGD is an assumption) |
| `monitoring.json` | notebook | PSI bands, Gini trigger, calibration trigger |
| `quality_gates.json` | notebook | every gate with the number that decided it |
| `feature_coverage.json` | notebook | non-null share of every model-matrix column in TRAIN / VALID / OOT; coverage gate |
| `feature_meta.json` | notebook | champion features, kinds, typical values (app contract) |
| `scored_oot_sample.parquet` | notebook | ≤ 3000 OOT applications with predictions (app demo) |
| `models/*` | notebook | champion in its native format + scorecard + linear bundle |
| `MANIFEST.json` | notebook | file, bytes, sha256, created_at, kernel type |

`README.md` is rendered from these files by `build_readme.py`. No metric is ever typed by hand.

## 5. Modelling contract

* Same split, same folds, same metrics for every family: dummy, WoE logistic, penalised logistic
  (L2/L1/EN × C, saga, `max_iter=3000`, non-converged fits excluded), random forest, XGBoost, LightGBM,
  CatBoost (CatBoost and the L1/elastic-net grid are not part of the final kernel run; the final roster is dummy, L2 logistic regression, WoE scorecard, random forest, XGBoost, monotone XGBoost, LightGBM).
* Scorecard: optbinning on train only, monotone bins, IV filter, iterative VIF ≤ 5, statsmodels table,
  sign check against the bin direction, PDO points table, reasons codes.
* Trees: hyper-parameters fixed a priori (documented in `src/hc_model.py`), early stopping on VALID.
* Calibration: Platt and isotonic are both fitted on VALID only; the choice is made on VALID (early VALID weeks fit, late VALID weeks score) and the result is evaluated on OOT. Isotonic is not the default.
* Champion = highest stability metric on VALID; ties or a degenerate sample fall back to pooled AUC and
  the artifact says which rule fired.

## 6. Decision contract

Cut-offs are chosen on VALID only (raw PD below the cut-off is approved) for three stated targets: bad rate of
approved loans <= 1.5%, <= 2.0%, and an 80% approval rate. They are applied once to OOT and the *observed* approval
rate and bad rate are reported. There is no cost model, no money-gain figure and no review band in the final run:
the data has no exposure, margin or term for the current application. The expected-loss table
(PD x LGD x EAD) is **illustrative**: LGD = 0.45 is an assumption (Basel foundation-IRB senior unsecured) and EAD is
the application credit amount.

## 7. App contract

`app.py` reads `artifacts/` only, never trains, and every page shows "artifact missing — run the Kaggle
notebook" when a file is absent. Pages are listed in `app.py`. `tests/test_app.py` drives every page, one scored row and one
CSV through `streamlit.testing.v1.AppTest` against a synthetic artifact bundle.

## 8. Definition of done

1. Unit tests green (`pytest tests`), including PIT, stability metric arithmetic and the leakage gate.
2. Kaggle kernel completes with zero error cells; the executed notebook is stored as
   `notebooks/HomeCreditStability_executed.ipynb`.
3. `artifacts/MANIFEST.json` exists and lists every artifact with a hash.
4. `quality_gates.json` reports each gate PASS/FAIL with the deciding number.
5. README rendered from artifacts; FINAL_REPORT states what ran, what passed and what did not.
6. Plausible-range check: OOT ROC-AUC inside 0.74–0.86; outside the range means a leakage hunt (above) or a
   bug hunt (below) before anything is accepted.
