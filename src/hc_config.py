"""Configuration for the Home Credit out-of-time PD / stability project.

Everything in this module is a *decision*: split fractions, screening
thresholds, scorecard scaling and the cost model.  Empirical numbers never
live here - they are written to ``artifacts/*.json`` by the pipeline.

The point of keeping them in one typed module is auditability: a reviewer can
read this file and see every knob that could move the headline number, and the
notebook embeds the same values in its markdown so nothing is hidden in a cell.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Identity of the modelling problem
# --------------------------------------------------------------------------
TARGET = "target"
ID_COL = "case_id"
DATE_COL = "date_decision"
WEEK_COL = "WEEK_NUM"
MONTH_COL = "MONTH"

SEED = 42

#: Rows kept per fold for the expanding-window CV fits.  The final models are
#: refit on the *full* training window; the cap exists because the whole
#: roster (7 model families x 4 folds) has to fit in one CPU kernel session.
CV_FOLDS = 4
CV_ROW_CAP = 300_000

#: Fraction of *weeks* (not rows) allocated to each split.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# --------------------------------------------------------------------------
# Data discovery
# --------------------------------------------------------------------------
DATA_SUBDIR = "parquet_files"

#: Candidate roots, checked in order.  ``/kaggle/input/...`` first so the same
#: notebook works as a Kaggle kernel with either the competition source or the
#: dataset mirror attached.
DATA_ROOT_CANDIDATES = (
    "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files",
    "/kaggle/input/usuarioml/home-credit-credit-risk-model-stability/parquet_files",
    "data/raw/parquet_files",
    "data/raw",
)

#: Tables that are aggregated and joined onto the application row.
#: ``date_cols`` are discovered from the schema at runtime (suffix ``D``);
#: ``month_pairs`` are the (year, month) column pairs used when a table has no
#: real date column, as documented in the competition data description.
TABLE_SPECS: tuple[dict, ...] = (
    # ``prefix`` is the *logical* table.  hc_data.table_parts() resolves it to every
    # numbered part file on disk (train_static_0_0, train_static_0_1, ...), so a logical
    # table is read once and always covers all weeks.  v1 listed individual part files,
    # which left the later weeks of several tables unread (see README, "v1 parts bug").
    {"name": "static_0", "depth": 0, "prefix": "train_static_0"},
    {"name": "static_cb_0", "depth": 0, "prefix": "train_static_cb_0"},
    {"name": "person_1", "depth": 1, "prefix": "train_person_1"},
    {"name": "applprev_1", "depth": 1, "prefix": "train_applprev_1"},
    {"name": "credit_bureau_a_1", "depth": 1, "prefix": "train_credit_bureau_a_1"},
    {"name": "credit_bureau_b_1", "depth": 1, "prefix": "train_credit_bureau_b_1"},
    {"name": "debitcard_1", "depth": 1, "prefix": "train_debitcard_1"},
    {"name": "deposit_1", "depth": 1, "prefix": "train_deposit_1"},
    {"name": "other_1", "depth": 1, "prefix": "train_other_1"},
    {"name": "tax_registry_a_1", "depth": 1, "prefix": "train_tax_registry_a_1"},
    {"name": "tax_registry_b_1", "depth": 1, "prefix": "train_tax_registry_b_1"},
    {"name": "tax_registry_c_1", "depth": 1, "prefix": "train_tax_registry_c_1"},
    {"name": "applprev_2", "depth": 2, "prefix": "train_applprev_2"},
    {"name": "person_2", "depth": 2, "prefix": "train_person_2"},
    {"name": "credit_bureau_a_2", "depth": 2, "prefix": "train_credit_bureau_a_2"},
    {"name": "credit_bureau_b_2", "depth": 2, "prefix": "train_credit_bureau_b_2"},
)
#: (year column, month column) pairs, verified against the data description.
MONTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("pmts_year_1139T", "pmts_month_158T"),   # active contract payments
    ("pmts_year_507T", "pmts_month_706T"),    # closed contract payments
)

# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
#: Per-table caps on the number of raw columns carried into aggregations.
#: Columns are ordered by coverage (non-null share) on a sample, ties broken
#: alphabetically, so the selection is unsupervised and reproducible.
MAX_NUMERIC_COLS_PER_TABLE = 16
MAX_CATEGORICAL_COLS_PER_TABLE = 6
#: Numeric aggregations.  ``last`` = value at the most recent dated record.
NUM_AGGS = ("mean", "max", "min", "std")
CAT_AGGS = ("nunique",)

#: Hard cap on the model matrix width after the screening step.
MAX_TREE_FEATURES = 450
MAX_LINEAR_FEATURES = 120
MAX_WOE_FEATURES = 70

#: Coverage rule (v2).  A feature is a model candidate only if it is non-null in at least
#: MIN_RECENT_COVERAGE of the TRAIN rows of the last RECENT_TRAIN_WEEKS train weeks.  This uses
#: TRAIN rows only (no valid/OOT information) and removes sources that stopped (or had not
#: started) before the end of training, which would be structurally empty afterwards.
RECENT_TRAIN_WEEKS = 8
#: Calendar-proxy screen (v5): a relative-date feature (``__rel_*``) whose |Spearman rho| with
#: WEEK_NUM on TRAIN rows is at least this value is dropped before selection. A "days before
#: decision" that is almost a fixed calendar date identifies when an application happened.
CALENDAR_PROXY_RHO = 0.8
MIN_RECENT_COVERAGE = 0.01
MIN_COVERAGE = 0.30           # column kept only if non-null in >= 30% of rows
MAX_MISSING_FOR_MODEL = 0.98  # feature dropped if missing in > 98% of train
UNIVARIATE_AUC_SCREEN = 0.80  # leakage gate: single feature AUC above this is quarantined
IV_MIN = 0.02                 # WoE scorecard: drop features below this IV
VIF_MAX = 5.0                 # iterative VIF pruning threshold
#: saga tolerance. 1e-3 (not sklearn's 1e-4 default) is a documented compute
#: decision: on a standardised design the objective is smooth, and at 1e-4 a
#: single L2 fit needs ~1200 epochs at 34k rows, which would put the whole grid
#: outside one kernel session. Convergence is still *required* (n_iter <
#: max_iter) and reported, so a non-converged fit is excluded, never shipped.
LR_TOL = 1e-3
MIN_BIN_SIZE = 0.02
MAX_N_BINS = 8
WOE_FIT_SAMPLE = 250_000      # rows used to fit the binning (train rows only)

#: Scorecard scaling: score = OFFSET - FACTOR * logit(PD).
PDO = 20.0
BASE_SCORE = 600.0
BASE_ODDS = 20.0  # 20:1 good:bad at BASE_SCORE

# --------------------------------------------------------------------------
# Calibration / monitoring
# --------------------------------------------------------------------------
CALIBRATION_METHOD = "isotonic"   # fitted on VALID rows only
PSI_BINS = 10
PSI_WATCH = 0.10
PSI_ALERT = 0.25
GINI_DROP_TRIGGER = 0.05          # monitoring: weekly gini drop that triggers review
MONITOR_MIN_CASES_PER_WEEK = 150  # below this a weekly gini is too noisy to act on

#: Official competition stability metric (Kaggle evaluation page):
#:   metric = mean(weekly gini) + 88 * min(0, slope) - 0.5 * std(residuals)
SLOPE_PENALTY = 88.0
VOLATILITY_PENALTY = 0.5

# --------------------------------------------------------------------------
# Expected-loss assumptions (illustrative; the data has no exposure or LGD)
# --------------------------------------------------------------------------
LGD_ASSUMED = 0.45   # Basel foundation-IRB senior unsecured LGD

# --------------------------------------------------------------------------
# Leakage gate
# --------------------------------------------------------------------------
#: Columns that must never reach the model matrix.  ``case_id`` identifies the
#: row, ``num_group*`` index the history, ``date_decision`` is the split key and
#: ``target`` is the outcome.
FORBIDDEN_EXACT = {
    ID_COL, TARGET, DATE_COL, WEEK_COL, MONTH_COL, "num_group1", "num_group2",
    "WEEK_NUM", "index",
}
#: Token scan for identifiers / post-outcome fields that could arrive under a
#: different name in a future table version.
FORBIDDEN_TOKENS = ("zip", "pincode", "email", "phone", "name_", "url", "iban")


def data_root(sample: bool = False) -> Path:
    """Return the directory that holds the parquet tables.

    ``sample`` is accepted for symmetry with the pipeline entry point; the
    resolver does not care how the caller will sample downstream.
    """
    override = os.environ.get("HC_DATA_ROOT")
    candidates = ([Path(override)] if override else []) + [
        Path(p) for p in DATA_ROOT_CANDIDATES
    ]

    # Kaggle mounts each source under /kaggle/input/<slug>, and the slug is not
    # always the one written in the metadata (a dataset can be re-slugged when
    # its title changes). Rather than hard-code it, scan the mount points for a
    # directory that actually contains the train parquet files.
    for base in (Path("/kaggle/input"),):
        if not base.exists():
            continue
        # Kaggle mounts sources either at /kaggle/input/<slug> or, on newer
        # images, at /kaggle/input/datasets/<owner>/<slug>. Walk the (tiny) tree
        # instead of guessing the depth, and accept any directory that really
        # contains train parquet files.
        for cand in sorted(base.rglob("parquet_files")):
            if cand.is_dir() and any(cand.glob("train/*.parquet")):
                return cand
        for cand in sorted(base.rglob("train")):
            if cand.is_dir() and any(cand.glob("*.parquet")):
                return cand.parent

    for cand in candidates:
        if not cand.exists():
            continue
        if any(cand.glob("train/*.parquet")) or any(cand.glob("*.parquet")):
            return cand
    raise FileNotFoundError(
        "No Home Credit parquet root found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". On Kaggle attach the competition or mirror dataset; locally place "
        "files under data/raw/parquet_files/{train,test}."
    )


def train_path(prefix: str) -> Path:
    """Path of a training parquet file given its file prefix."""
    return data_root() / "train" / f"{prefix}.parquet"


def project_root() -> Path:
    """Repository root, used only by local scripts/tests."""
    return Path(__file__).resolve().parents[1]


def artifact_dir() -> Path:
    """Directory every pipeline artifact is written to."""
    out = Path(os.environ.get("HC_ARTIFACT_DIR", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    return out

