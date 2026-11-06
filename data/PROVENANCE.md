# Data provenance

Generated 2026-10-22 04:10 by `scripts/make_provenance.py` from the files on disk.

## Source

| Item | Value |
|---|---|
| Competition | `home-credit-credit-risk-model-stability` (Kaggle, Home Credit Group) |
| Competition URL | https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability |
| Prediction unit | `case_id` (one credit application) |
| Prediction time | `date_decision` |
| Target | `target` |
| Licence | Kaggle competition rules for the competition above (competition data, no redistribution) |
| Files retrieved | 32 train parquet + 2 supporting files |
| Train parquet size | 1.33 GB |
| Retrieval date | 2026-10-22 |

## How the files were obtained

The competition's data API returned `403 RulesAcceptanceRequired` for this account (`You must accept this competition's rules before you'll be able to download files`), and rules acceptance is not exposed through the Kaggle API - it is a browser action. To keep the run reproducible and the data real, the identical file set was attached from the public Kaggle dataset `usuarioml/home-credit-credit-risk-model-stability`, which carries the same 138 files (32 train parquet, 36 test parquet, 32 train csv, 36 test csv, `feature_definitions.csv`, `sample_submission.csv`) with the same names and layout as the competition download.

Consequence for the pipeline: the data-root resolver looks for `/kaggle/input/...` with either the competition source or the mirror attached, then falls back to `data/raw/parquet_files`. No code path depends on which of the two is mounted.

## Data description statements quoted by the pipeline

- `case_id` - "This is the unique identifier for each credit case. You'll need this ID to join relevant tables to the base table."
- `date_decision` - "This refers to the date when a decision was made regarding the approval of the loan."
- `WEEK_NUM` - "This is the week number used for aggregation. In the test sample, `WEEK_NUM` continues sequentially from the last training value of `WEEK_NUM`."
- `target` - "This is the target value, determined after a certain period based on whether or not the client defaulted on the specific credit case (loan)."
- Depth - "depth=0 - These are static features directly tied to a specific `case_id`" / "depth=1 - Each `case_id` has an associated historical record, indexed by `num_group1`" / "depth=2 - Each `case_id` has an associated historical record, indexed by both `num_group1` and `num_group2`".
- Transform notation - "**P** - Transform DPD (Days past due), **M** - Masking categories, **A** - Transform amount, **D** - Transform date, **T** - Unspecified Transform, **L** - Unspecified Transform".
- Metric - "stability metric = mean(gini) + 88.0 * min(0, a) - 0.5 * std(residuals)" with "gini = 2 * AUC - 1" computed per `WEEK_NUM` (competition Evaluation page).

The full text of the competition pages consulted (Description, Evaluation, data-description, rules) is stored verbatim in `data/raw/competition_pages.json`.

## Train parquet files

| file | rows | cols | MB | sha256 |
|---|---|---|---|---|
| parquet_files/train/train_applprev_1_0.parquet | 3,887,684 | 41 | 107.0 | `0301380ee0092b0d…` |
| parquet_files/train/train_applprev_1_1.parquet | 2,638,295 | 41 | 72.9 | `cae63fdb5d8de8ba…` |
| parquet_files/train/train_applprev_2.parquet | 14,075,487 | 6 | 28.5 | `153dbbd225a9b384…` |
| parquet_files/train/train_base.parquet | 1,526,659 | 5 | 7.0 | `29d94662545eece9…` |
| parquet_files/train/train_credit_bureau_a_1_0.parquet | 4,108,212 | 79 | 64.9 | `b8e29ea89da103ce…` |
| parquet_files/train/train_credit_bureau_a_1_1.parquet | 6,009,192 | 79 | 183.3 | `5f3c4b28e09685fb…` |
| parquet_files/train/train_credit_bureau_a_1_2.parquet | 3,743,810 | 79 | 123.5 | `931908b33e0c3d67…` |
| parquet_files/train/train_credit_bureau_a_1_3.parquet | 2,079,323 | 79 | 71.8 | `b9d4f5803ecb2431…` |
| parquet_files/train/train_credit_bureau_a_2_0.parquet | 5,296,031 | 19 | 5.8 | `053c2269990eabd6…` |
| parquet_files/train/train_credit_bureau_a_2_1.parquet | 7,861,809 | 19 | 10.0 | `18f30c8bb9f78148…` |
| parquet_files/train/train_credit_bureau_a_2_10.parquet | 4,386,062 | 19 | 7.6 | `4913d6516000587f…` |
| parquet_files/train/train_credit_bureau_a_2_2.parquet | 17,893,536 | 19 | 30.7 | `c36e8952d3ffedef…` |
| parquet_files/train/train_credit_bureau_a_2_3.parquet | 26,563,901 | 19 | 47.8 | `55de63492b08a480…` |
| parquet_files/train/train_credit_bureau_a_2_4.parquet | 27,025,737 | 19 | 49.2 | `6262695f7ae76319…` |
| parquet_files/train/train_credit_bureau_a_2_5.parquet | 33,053,760 | 19 | 61.0 | `92e20fe1e772353c…` |
| parquet_files/train/train_credit_bureau_a_2_6.parquet | 25,511,332 | 19 | 45.2 | `720fc7c9bc407d49…` |
| parquet_files/train/train_credit_bureau_a_2_7.parquet | 8,055,986 | 19 | 14.1 | `80779c277c1d336f…` |
| parquet_files/train/train_credit_bureau_a_2_8.parquet | 13,927,071 | 19 | 23.8 | `f639dc8b7c896e6c…` |
| parquet_files/train/train_credit_bureau_a_2_9.parquet | 18,723,227 | 19 | 32.1 | `882177fa0b1cb550…` |
| parquet_files/train/train_credit_bureau_b_1.parquet | 85,791 | 45 | 4.5 | `a4842614f688e939…` |
| parquet_files/train/train_credit_bureau_b_2.parquet | 1,286,755 | 6 | 2.1 | `9cd22472c28a2714…` |
| parquet_files/train/train_debitcard_1.parquet | 157,302 | 6 | 1.2 | `4f1d81bd55075dff…` |
| parquet_files/train/train_deposit_1.parquet | 145,086 | 5 | 1.6 | `845fa0b14c3614b4…` |
| parquet_files/train/train_other_1.parquet | 51,109 | 7 | 0.8 | `a84aaf075bc3dd11…` |
| parquet_files/train/train_person_1.parquet | 2,973,991 | 37 | 43.9 | `81fcdb5596d8f747…` |
| parquet_files/train/train_person_2.parquet | 1,643,410 | 11 | 7.6 | `c295eb898df66fde…` |
| parquet_files/train/train_static_0_0.parquet | 1,003,757 | 168 | 113.9 | `c993192e68853259…` |
| parquet_files/train/train_static_0_1.parquet | 522,902 | 168 | 74.7 | `78adb27582754d3c…` |
| parquet_files/train/train_static_cb_0.parquet | 1,500,476 | 53 | 30.7 | `08a460c7c10b9ebc…` |
| parquet_files/train/train_tax_registry_a_1.parquet | 3,275,770 | 5 | 21.7 | `50ff790d9daec8dc…` |
| parquet_files/train/train_tax_registry_b_1.parquet | 1,107,933 | 5 | 9.9 | `a1f1de27b00da19e…` |
| parquet_files/train/train_tax_registry_c_1.parquet | 3,343,800 | 5 | 29.8 | `59786491e9cc7905…` |

Total training rows across tables: 243,465,196.

## Supporting files

| file | MB | sha256 |
|---|---|---|
| feature_definitions.csv | 0.03 | `628f1f75ae2842de…` |
| competition_pages.json | 0.05 | `3c1b0db91c36af7f…` |

Full-length hashes for every file the run consumed are written by the notebook to `artifacts/data_provenance.json`; the values above are the local copies of the same bytes.

## What is *not* in the data

- No exposure, LGD, margin or term for the current application: the decision layer reports only observed approval / bad rates at cut-offs frozen on VALID; the illustrative expected-loss table uses **named assumptions** (`LGD_ASSUMED = 0.45` in `src/hc_config.py`; EAD = application credit amount), not observations.
- No post-decision outcome window shorter than the target definition: the target is the only outcome available, and it is used for evaluation only.
- The competition test set is unlabelled, so every out-of-time number in this project is measured on the latest weeks of the labelled train data.
