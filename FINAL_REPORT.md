# Final report

Kernel run: `Batch`, runtime 52 min, python 3.13.16, xgboost 3.4.1, lightgbm 4.6.0, scikit-learn 1.6.1, sample_cases=None.

## Data and split
* train: 1,282,082 applications, default rate 0.0332, weeks 0-63
* valid: 92,870 applications, default rate 0.0243, weeks 64-77
* oot: 151,707 applications, default rate 0.0212, weeks 78-91

## Champion
`xgboost`. Rule: max VALID Gini; models within 0.002 Gini of the best are tied and the tie goes to 'xgboost'. Frozen at 2026-11-06T05:41:06 before OOT was scored (`artifacts/champion.json`).

## Scorecard sign check
Before the fix 39/46; after 37/37 (9 variables dropped).

## Paired bootstrap, OOT Gini difference
* xgboost_minus_lightgbm: +0.0040 (95% CI +0.0010 to +0.0072, 1000 resamples)
* xgboost_minus_penalised_lr: +0.1218 (95% CI +0.1105 to +0.1328, 1000 resamples)
* xgboost_minus_random_forest: +0.0391 (95% CI +0.0322 to +0.0463, 1000 resamples)
* xgboost_minus_woe_scorecard: +0.0772 (95% CI +0.0689 to +0.0862, 1000 resamples)

## Quality gates
* PASS: leakage:no_id_or_outcome_columns
* PASS: leakage:target_not_in_design
* PASS: leakage:time_split_ordering
* PASS: leakage:no_case_overlap
* PASS: leakage:univariate_auc_screen
* PASS: leakage:scaler_fitted_on_train_rows
* PASS: leakage:woe_binned_on_train_rows
* PASS: leakage:woe_fit_sample_within_train
* PASS: dummy_baseline_beaten
* PASS: plausible_oot_range
* PASS: valid_oot_gap_under_0.05
* PASS: scorecard_all_signs_consistent
* PASS: calibration_expected_actual_after
* PASS: score_psi_train_to_oot
* PASS: xgb_minus_lr_ci_excludes_zero
* PASS: coverage_no_final_feature_empty_in_valid_or_oot
* PASS: no_absolute_calendar_features
* PASS: no_calendar_proxy_features
* PASS: adversarial_auc_and_top_drivers_reported
* PASS: champion_frozen_before_oot

20 pass, 0 warn, 0 fail.

## Not done / limits
* Random forest is fitted on a 500k-row TRAIN subsample (compute cap).
* Expected loss uses an assumed LGD (0.45) and is illustrative.
* Details and all other numbers: `README.md` and `artifacts/`.
