"""Data access, point-in-time filtering and history aggregation.

Design notes
------------
*Everything* is read with polars lazy scans; parquet is never converted to
pandas until the model matrix is final.  The raw tables are 1.3 GB of parquet
and ~40 million history rows, so a pandas read would blow the memory budget.

The point-in-time (PIT) rule is the heart of an honest credit-risk feature
store: a record dated *after* the application decision could not have been
known by the underwriter, so it must not reach the feature table.  Depth-1 and
depth-2 tables carry either a real date (columns whose name ends in ``D``) or a
(year, month) pair (columns ending in ``T``).  Both are used, and the number of
dropped rows is logged per table.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hc_metrics import auc

from hc_config import (
    CAT_AGGS,
    DATE_COL,
    ID_COL,
    MIN_COVERAGE,
    MONTH_PAIRS,
    NUM_AGGS,
    SEED,
    TABLE_SPECS,
    TARGET,
    WEEK_COL,
    MAX_CATEGORICAL_COLS_PER_TABLE,
    MAX_NUMERIC_COLS_PER_TABLE,
    data_root,
)


# --------------------------------------------------------------------------
# Discovery / IO
# --------------------------------------------------------------------------
def table_path(prefix: str) -> Path:
    """Resolve a table prefix to a parquet path under the discovered root."""
    p = data_root() / "train" / f"{prefix}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing table {p}; attach the competition data")
    return p


def table_parts(prefix: str) -> list[Path]:
    """Every parquet file of one logical table, in numeric part order.

    ``train_static_0`` resolves to ``train_static_0_0``, ``train_static_0_1`` and so on;
    ``train_credit_bureau_a_2`` resolves to parts 0..10.  A table without parts
    (``train_person_1``) resolves to its single file.  The part number must be purely
    numeric, so ``train_applprev_1`` never picks up ``train_applprev_2``.
    """
    root = data_root() / "train"
    pat = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?$")
    found = []
    for p in root.glob(f"{prefix}*.parquet"):
        m = pat.match(p.stem)
        if m:
            found.append((int(m.group(1)) if m.group(1) is not None else -1, p))
    return [p for _, p in sorted(found)]


def table_exists(prefix: str) -> bool:
    return len(table_parts(prefix)) > 0


def scan_part(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    if columns is not None:
        have = set(lf.collect_schema().names())
        lf = lf.select([c for c in columns if c in have])
    return lf


def scan_table(prefix: str, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lazy union of all parts of a logical table.

    Schemas are aligned explicitly: columns missing from a part become null and
    differing dtypes are relaxed to a common supertype (``diagonal_relaxed``).
    """
    parts = table_parts(prefix)
    if not parts:
        raise FileNotFoundError(f"missing table {prefix}; attach the competition data")
    scans = [scan_part(p, columns) for p in parts]
    if len(scans) == 1:
        return scans[0]
    return pl.concat(scans, how="diagonal_relaxed")


def table_columns(prefix: str) -> list[str]:
    """Union of column names over all parts (first-seen order)."""
    names: list[str] = []
    for p in table_parts(prefix):
        for c in pl.scan_parquet(p).collect_schema().names():
            if c not in names:
                names.append(c)
    return names


def available_tables() -> list[str]:
    """Prefixes actually present on disk (the smoke run may see a subset)."""
    root = data_root() / "train"
    return sorted(p.stem for p in root.glob("*.parquet"))


def file_fingerprint(path: Path, n_bytes: int = 1 << 20) -> dict:
    """sha256 over the whole file for provenance (cheap: one pass, cached)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(n_bytes), b""):
            h.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def read_base(sample_cases: int | None = None) -> pl.DataFrame:
    """Base table: one row per credit application.

    ``date_decision`` is the prediction time (the day the loan decision was
    made), ``WEEK_NUM`` is the competition's aggregation week and ``target`` is
    the default flag observed after that decision.
    """
    df = (
        pl.scan_parquet(table_path("train_base"))
        .select([ID_COL, DATE_COL, WEEK_COL, "MONTH", TARGET])
        .with_columns(pl.col(DATE_COL).str.to_date(strict=False))
        .collect()
    )
    if sample_cases is not None and sample_cases < df.height:
        df = df.sample(n=sample_cases, seed=SEED, shuffle=True).sort(ID_COL)
    return df


def split_bounds(weeks: list[int], fractions: tuple[float, float, float]) -> dict:
    """Week cut-offs for the train/valid/OOT split.

    Splitting on *weeks* (not rows) keeps every calendar week intact, which is
    required because the stability metric is computed per week: a week split
    across two partitions would put the same population in train and OOT.
    """
    weeks = sorted(weeks)
    n = len(weeks)
    n_train = int(round(n * fractions[0]))
    n_valid = int(round(n * fractions[1]))
    n_train = max(n_train, 1)
    n_valid = max(n_valid, 1)
    n_oot = n - n_train - n_valid
    if n_oot < 1:  # never let the OOT slice vanish
        n_oot = 1
        n_valid = max(1, n - n_train - n_oot)
    return {
        "train_weeks": weeks[:n_train],
        "valid_weeks": weeks[n_train : n_train + n_valid],
        "oot_weeks": weeks[n_train + n_valid :],
        "week_min": weeks[0],
        "week_max": weeks[-1],
        "n_weeks": n,
    }


def label_split(week: int, bounds: dict) -> str:
    """Map a week number to its partition name."""
    if week <= bounds["train_weeks"][-1]:
        return "train"
    if week <= bounds["valid_weeks"][-1]:
        return "valid"
    return "oot"


# --------------------------------------------------------------------------
# Point-in-time filtering
# --------------------------------------------------------------------------
def date_columns(names: list[str]) -> list[str]:
    """Real date columns by the documented naming convention (suffix ``D``)."""
    return [c for c in names if c.endswith("D")]


def year_columns() -> set[str]:
    """Year columns of the (year, month) pairs; they carry an absolute calendar year."""
    return {y for y, _ in MONTH_PAIRS}


def month_pair_of(col: str) -> tuple[str, str] | None:
    for y, m in MONTH_PAIRS:
        if col == y:
            return (y, m)
    return None


def is_calendar_column(col: str) -> bool:
    """True for a raw column that holds an absolute calendar value (date or year/month)."""
    return col.endswith("D") or col in {c for pair in MONTH_PAIRS for c in pair}


def relative_expr(col: str) -> pl.Expr:
    """Relative-to-decision value of a calendar column (needs ``date_decision`` in the frame).

    * date column (suffix ``D``, string)  -> days from the date to ``date_decision``
    * year column of a (year, month) pair -> months from (year, month) to the decision month

    Absolute calendar values never reach the design: they identify *when* an application
    happened, not how risky the borrower is, and they separate TRAIN from OOT by construction.
    """
    pair = month_pair_of(col)
    if pair is not None:
        y, m = pair
        ok = (pl.col(y).is_between(1950, 2035) & pl.col(m).is_between(1, 12))
        dec = pl.col(DATE_COL).dt.year() * 12 + pl.col(DATE_COL).dt.month()
        return pl.when(ok).then(dec - (pl.col(y) * 12 + pl.col(m))).otherwise(None).cast(pl.Float64)
    return (pl.col(DATE_COL) - pl.col(col).str.to_date(strict=False)).dt.total_days().cast(pl.Float64)


def is_absolute_calendar_feature(name: str) -> bool:
    """Gate rule: a final feature built from a calendar column without the relative transform."""
    parts = name.split("__")
    if len(parts) < 2:
        return name.endswith("D")
    base, last = parts[1], parts[-1]
    if not is_calendar_column(base):
        return False
    return not (len(parts) >= 3 and last.startswith("rel_"))


def pit_filter(
    lf: pl.LazyFrame,
    decision: pl.DataFrame,
    d_cols: list[str],
    month_pairs: list[tuple[str, str]] = list(MONTH_PAIRS),
) -> tuple[pl.LazyFrame, dict]:
    """Drop history records dated after the application's ``date_decision``.

    A record is kept when *neither* its latest real date nor its latest
    (year, month) pair lies after the decision date.  Records with no usable
    date at all are kept: the rule is "drop if after", not "drop if unknown",
    and dropping unknown-date records would silently remove whole depth-2
    tables from the feature store.

    Returns the filtered lazy frame plus a log entry describing how many rows
    were dropped and why.
    """
    lf = lf.join(decision.lazy(), on=ID_COL, how="inner")
    before_condition = None

    if d_cols:
        casts = [pl.col(c).str.to_date(strict=False) for c in d_cols]
        max_date = pl.max_horizontal(casts)
        lf = lf.with_columns(max_date.alias("__pit_date"))
        before_condition = pl.col("__pit_date").is_null() | (
            pl.col("__pit_date") <= pl.col(DATE_COL)
        )

    month_terms = []
    for year_col, month_col in month_pairs:
        if year_col in lf.collect_schema().names() and month_col in lf.collect_schema().names():
            month_terms.append(
                pl.when(
                    pl.col(year_col).is_not_null()
                    & pl.col(month_col).is_not_null()
                    & pl.col(year_col).is_between(1950, 2035)
                    & pl.col(month_col).is_between(1, 12)
                )
                .then(pl.col(year_col) * 100 + pl.col(month_col))
                .otherwise(None)
            )
    if month_terms:
        lf = lf.with_columns(pl.max_horizontal(month_terms).alias("__pit_month"))
        dec_month = pl.col(DATE_COL).dt.year() * 100 + pl.col(DATE_COL).dt.month()
        month_condition = pl.col("__pit_month").is_null() | (pl.col("__pit_month") <= dec_month)
        before_condition = month_condition if before_condition is None else (before_condition & month_condition)

    log = {
        "date_columns": d_cols,
        "month_pairs_used": [
            [y, m] for y, m in month_pairs if y in lf.collect_schema().names()
        ],
        "has_temporal_information": before_condition is not None,
    }
    if before_condition is None:
        log.update({"rows_in": None, "rows_kept": None, "rows_dropped": 0, "drop_share": 0.0})
        return lf, log

    counts = lf.select(
        pl.len().alias("rows_in"),
        before_condition.sum().alias("rows_kept"),
    ).collect()
    rows_in = int(counts["rows_in"][0])
    rows_kept = int(counts["rows_kept"][0])
    lf = lf.filter(before_condition)
    log.update(
        {
            "rows_in": rows_in,
            "rows_kept": rows_kept,
            "rows_dropped": rows_in - rows_kept,
            "drop_share": float((rows_in - rows_kept) / rows_in) if rows_in else 0.0,
        }
    )
    return lf, log


def pit_frame(df: pl.DataFrame, decision: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Eager convenience wrapper around :func:`pit_filter` (used by tests)."""
    d_cols = date_columns(df.columns)
    lf, log = pit_filter(df.lazy(), decision, d_cols)
    out = lf.collect()
    for c in ("__pit_date", "__pit_month"):
        if c in out.columns:
            out = out.drop(c)
    return out, log


# --------------------------------------------------------------------------
# Column screening and aggregation
# --------------------------------------------------------------------------
def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def screen_columns(
    prefix: str,
    decision: pl.DataFrame,
    target_by_id: pl.DataFrame,
    sample_rows: int = 200_000,
) -> dict:
    """Rank a table's columns for the feature build.

    Two filters, both fitted on *training rows only*:

    1. coverage - a column must be non-null in at least ``MIN_COVERAGE`` of the
       PIT-filtered sample, because a feature that is 90% missing mostly
       measures data availability;
    2. univariate AUC - columns are ranked by ``|AUC - 0.5|`` against the
       target on the same sample, which is a supervised screen and therefore
       must never see OOT rows.

    Returns the ordered column lists plus the screening table for the artifact.
    """
    parts = table_parts(prefix)
    names = table_columns(prefix)
    d_cols = [c for c in date_columns(names) if c != ID_COL]

    # Join the decision frame *first*: sampling before the join would draw rows
    # that belong to applications outside the (possibly sampled) base table and
    # the screen would then see a fraction of the rows it asked for.
    # Multi-part tables: every part is filtered and counted separately (memory), the
    # stride is global (total rows over all parts // sample_rows) and the per-part
    # samples are stacked, so the sample is spread over all TRAIN weeks of the table.
    part_lfs = []
    total = 0
    for part in parts:
        lf, _pit = pit_filter(pl.scan_parquet(part), decision, d_cols)
        drop = [c for c in ("__pit_date", "__pit_month") if c in lf.collect_schema().names()]
        lf = lf.drop(drop).join(target_by_id.lazy(), on=ID_COL, how="inner")
        # calendar columns -> relative to the decision date (v4); raw absolute values are dropped
        pn = lf.collect_schema().names()
        rel = [relative_expr(c).alias(c) for c in pn if c in d_cols or month_pair_of(c) is not None]
        raw_month = [m for _, m in MONTH_PAIRS if m in pn]
        if rel:
            lf = lf.with_columns(rel)
        lf = lf.drop([c for c in raw_month + [DATE_COL] if c in lf.collect_schema().names()])
        total += int(lf.select(pl.len()).collect()["len"][0])
        part_lfs.append(lf)
    stride = max(1, total // sample_rows)
    frames = [
        lf.with_row_index("__idx").filter(pl.col("__idx") % stride == 0).drop("__idx").collect()
        for lf in part_lfs
    ]
    df = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    del frames, part_lfs
    if df.height == 0:
        return {"numeric": [], "categorical": [], "screening": [], "sampled_rows": 0}

    rows = []
    numeric, categorical = [], []
    for col, dtype in zip(df.columns, df.dtypes):
        if col in (ID_COL, TARGET) or col.startswith("num_group"):
            continue   # num_group* index the history rows; they are not borrower attributes
        coverage = float(df[col].is_not_null().mean())
        is_num = dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        n_unique = int(df[col].n_unique())
        entry = {
            "table": prefix,
            "transform": "relative_to_decision" if (col in d_cols or month_pair_of(col) is not None) else "none",
            "column": col,
            "dtype": str(dtype),
            "coverage": coverage,
            "n_unique": n_unique,
            "univariate_auc": float("nan"),
        }
        if coverage >= MIN_COVERAGE and n_unique > 1:
            y = df[TARGET].to_numpy()
            if is_num:
                x = df[col].cast(pl.Float64).fill_nan(None).to_numpy()
                mask = ~np.isnan(x)
                if mask.sum() > 1000 and 0 < y[mask].mean() < 1:
                    entry["univariate_auc"] = auc(y[mask], x[mask])
            else:
                # Categorical: AUC of the category's train default rate, which is
                # an ordinal encoding of the level means.
                g = df.group_by(col).agg(pl.col(TARGET).mean().alias("rate"))
                # join instead of a per-row Python map: same statistic, ~100x faster
                x = (
                    df.select([col])
                    .with_row_index("__row")
                    .join(g, on=col, how="left")
                    .sort("__row")["rate"]
                    .to_numpy()
                )
                mask = ~np.isnan(x)
                if mask.sum() > 1000 and 0 < y[mask].mean() < 1:
                    entry["univariate_auc"] = auc(y[mask], x[mask])
            entry["signal"] = abs(entry["univariate_auc"] - 0.5) if np.isfinite(entry["univariate_auc"]) else 0.0
            (numeric if is_num else categorical).append(entry)
        rows.append(entry)

    numeric = sorted(numeric, key=lambda r: (-r["signal"], r["column"]))[:MAX_NUMERIC_COLS_PER_TABLE]
    # A categorical needs both signal and a manageable cardinality: one-hot on a
    # 4000-level column is noise, and CatBoost would choke on it.
    categorical = [r for r in categorical if r["n_unique"] <= 200]
    categorical = sorted(categorical, key=lambda r: (-r["signal"], r["column"]))[:MAX_CATEGORICAL_COLS_PER_TABLE]
    return {
        "numeric": [r["column"] for r in numeric],
        "categorical": [r["column"] for r in categorical],
        "screening": rows,
        "sampled_rows": int(df.height),
    }


def aggregate_table(
    prefix: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    decision: pl.DataFrame,
    depth: int,
    train_ids: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Collapse a history table to one row per ``case_id``.

    Aggregations: count, recency in days, and mean/max/min/std for numeric
    columns; nunique and last-value-in-alphabetical-order for categoricals.
    ``last`` is approximated by the maximum category code (stable and cheap);
    it is documented as such rather than pretending to be time-ordered.
    """
    parts = table_parts(prefix)
    if not parts:
        raise FileNotFoundError(f"missing table {prefix}")
    names = table_columns(prefix)
    d_cols = date_columns(names)
    month_cols = [c for pair in MONTH_PAIRS for c in pair if c in names]
    # The PIT filter needs the date columns, so they travel with the payload
    # even though they never become features themselves.
    keep = [ID_COL] + [c for c in numeric_cols + categorical_cols if c in names]
    keep += [c for c in d_cols + month_cols if c not in keep]

    # Category codes: a stable ordinal code per level, from the sorted TRAIN-case level
    # set of the whole logical table (all parts).  polars' own Categorical physical code
    # depends on order of appearance inside one frame, which would differ between part
    # files; levels unseen in TRAIN map to null (ignored by max), as in to_model_matrix.
    cat_maps: dict[str, dict] = {}
    categorical_cols = [c for c in categorical_cols if not is_calendar_column(c)]
    for c in categorical_cols:
        if c not in names:
            continue
        levels: set = set()
        for part in parts:
            sub = scan_part(part, [ID_COL, c])
            if c not in sub.collect_schema().names():
                continue
            u = sub.join(train_ids.lazy(), on=ID_COL, how="semi") if train_ids is not None else sub
            levels.update(u.select(pl.col(c).cast(pl.String)).unique().collect()[c].drop_nulls().to_list())
        cat_maps[c] = {lvl: i for i, lvl in enumerate(sorted(levels))}

    # Each part is aggregated on its own and the results are stacked.  This is exact
    # because a case_id lives in exactly one part (checked below), and it keeps the peak
    # memory at one part (credit_bureau_a_2 has 11 parts and tens of millions of rows).
    pieces = []
    log_in = log_kept = 0
    has_time = False
    pit_log: dict = {}
    for part in parts:
        pnames = pl.scan_parquet(part).collect_schema().names()
        lf = scan_part(part, keep)
        lf, plog = pit_filter(lf, decision, [c for c in d_cols if c in pnames])
        pit_log = plog
        if plog.get("rows_in") is not None:
            has_time = True
            log_in += plog["rows_in"]
            log_kept += plog["rows_kept"]
        drop = [c for c in ("__pit_date", "__pit_month") if c in lf.collect_schema().names()]
        lf = lf.drop(drop)
        pd_cols = [c for c in d_cols if c in pnames]

        aggs = [pl.len().alias(f"{prefix}__n_records")]
        if pd_cols:
            # Recency of the newest surviving record; negative-free by the PIT rule.
            newest = pl.max_horizontal([pl.col(c).str.to_date(strict=False) for c in pd_cols])
            lf = lf.with_columns(newest.alias("__newest"))
            aggs.append(
                (pl.col(DATE_COL).first() - pl.col("__newest")).dt.total_days().alias(f"{prefix}__recency_days")
            )
        for c in numeric_cols:
            if c not in pnames:
                continue
            is_rel = is_calendar_column(c)
            expr = relative_expr(c) if is_rel else pl.col(c)
            for agg in NUM_AGGS:
                aggs.append(getattr(expr, agg)().alias(f"{prefix}__{c}__{'rel_' if is_rel else ''}{agg}"))
        for c in categorical_cols:
            if c not in pnames:
                continue
            aggs.append(pl.col(c).n_unique().alias(f"{prefix}__{c}__nunique"))
            aggs.append(
                pl.col(c).cast(pl.String).replace_strict(cat_maps[c], default=None, return_dtype=pl.Int32)
                .max().alias(f"{prefix}__{c}__maxcode")
            )
        pieces.append(lf.group_by(ID_COL).agg(aggs).collect())

    out = pl.concat(pieces, how="diagonal_relaxed") if len(pieces) > 1 else pieces[0]
    if out[ID_COL].n_unique() != out.height:
        raise ValueError(
            f"{prefix}: a case_id appears in more than one part file; per-part aggregation "
            "would be wrong (mean/std cannot be stacked). Combine sums and counts instead."
        )
    pit_log = dict(pit_log)
    if has_time:
        pit_log.update({"rows_in": log_in, "rows_kept": log_kept, "rows_dropped": log_in - log_kept,
                        "drop_share": float((log_in - log_kept) / log_in) if log_in else 0.0})
    pit_log["table"] = prefix
    pit_log["depth"] = depth
    pit_log["n_parts"] = len(parts)
    pit_log["rows_out"] = int(out.height)
    return out, pit_log


def depth0_frame(prefix: str, decision: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Depth-0 tables are already one row per case; only the PIT rule applies.

    Multi-part tables (``train_static_0_*``) are processed part by part and stacked
    with an explicit schema alignment (missing columns -> null, dtypes relaxed).
    """
    parts = table_parts(prefix)
    if not parts:
        raise FileNotFoundError(f"missing table {prefix}")
    pieces, log_in, log_kept, has_time, pit_log = [], 0, 0, False, {}
    for part in parts:
        names = pl.scan_parquet(part).collect_schema().names()
        lf, plog = pit_filter(pl.scan_parquet(part), decision, date_columns(names))
        pit_log = plog
        if plog.get("rows_in") is not None:
            has_time = True
            log_in += plog["rows_in"]
            log_kept += plog["rows_kept"]
        drop = [c for c in ("__pit_date", "__pit_month", DATE_COL) if c in lf.collect_schema().names()]
        # A depth-0 table must not duplicate the application row; keep the last
        # record per case if it does (logged by the returned height vs base height).
        pieces.append(lf.drop(drop).unique(subset=[ID_COL], keep="last").collect())
    out = pl.concat(pieces, how="diagonal_relaxed") if len(pieces) > 1 else pieces[0]
    out = out.unique(subset=[ID_COL], keep="last")
    rename = {
        c: f"{prefix}__{c}" for c in out.columns if c not in (ID_COL,) and not c.startswith(f"{prefix}__")
    }
    out = out.rename(rename)
    pit_log = dict(pit_log)
    if has_time:
        pit_log.update({"rows_in": log_in, "rows_kept": log_kept, "rows_dropped": log_in - log_kept,
                        "drop_share": float((log_in - log_kept) / log_in) if log_in else 0.0})
    pit_log.update({"table": prefix, "depth": 0, "n_parts": len(parts), "rows_out": int(out.height)})
    return out, pit_log


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
def build_feature_frame(
    base: pl.DataFrame,
    max_tables: int | None = None,
    verbose: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """Join every configured table onto the application row.

    Returns the wide frame (``case_id``, ``target``, ``week``, ``date`` plus raw
    feature columns) and an audit dictionary: PIT logs per table, screening
    tables, and per-table column counts.
    """
    decision = base.select([ID_COL, DATE_COL])
    # Supervised column screening must see TRAIN weeks only (no valid / OOT labels).
    from hc_config import SPLIT_FRACTIONS

    _bounds = split_bounds(sorted(base[WEEK_COL].unique().to_list()), SPLIT_FRACTIONS)
    target_by_id = base.filter(pl.col(WEEK_COL) <= _bounds["train_weeks"][-1]).select([ID_COL, TARGET])
    feats = base.select([ID_COL, TARGET, WEEK_COL, DATE_COL])
    audit = {"pit": [], "screening": [], "tables_used": [], "tables_missing": []}

    specs = list(TABLE_SPECS)
    if max_tables is not None:
        specs = specs[:max_tables]

    for spec in specs:
        prefix = spec["prefix"]
        if not table_exists(prefix):
            audit["tables_missing"].append(prefix)
            continue
        if spec["depth"] == 0:
            frame, log = depth0_frame(prefix, decision)
        else:
            screen = screen_columns(prefix, decision, target_by_id)
            audit["screening"].extend(screen["screening"])
            if not screen["numeric"] and not screen["categorical"]:
                audit["tables_missing"].append(f"{prefix} (no usable columns)")
                continue
            frame, log = aggregate_table(
                prefix, screen["numeric"], screen["categorical"], decision, spec["depth"],
                train_ids=target_by_id.select(ID_COL),
            )
        feats = feats.join(frame, on=ID_COL, how="left")
        # Downcast immediately: the aggregated tables arrive as float64 and a
        # 900-column float64 frame at 1.5M rows would be 11 GB before any model
        # is fitted. Float32 halves that with no measurable loss in a tree or a
        # standardised logistic.
        float_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64)
        new_cols = [c for c in frame.columns if c != ID_COL and feats.schema.get(c) in float_types]
        if new_cols:
            feats = feats.with_columns([pl.col(c).cast(pl.Float32) for c in new_cols])
        audit["pit"].append(log)
        audit["tables_used"].append(prefix)
        if verbose:
            print(
                f"  {prefix:24s} rows_in={log.get('rows_in')} kept={log.get('rows_kept')} "
                f"dropped={log.get('rows_dropped')} cols_added={frame.width - 1}"
            )
    return feats, audit


def build_diagnostics(decision: pl.DataFrame) -> pl.DataFrame:
    """Subgroup attributes for the fairness/stability cuts - never features.

    Age, income and product type are needed to answer "does the model degrade
    for young applicants / low-income applicants / a product line".  They are
    kept in a separate frame so they cannot accidentally enter the design: a
    protected attribute used as a predictor is exactly the kind of leakage this
    project screens for.
    """
    out = decision.select([ID_COL, DATE_COL]).clone()

    try:
        cb = (
            scan_table("train_static_cb_0")
            .select([ID_COL, "dateofbirth_337D"])
            .with_columns(pl.col("dateofbirth_337D").str.to_date(strict=False).alias("dob"))
            .group_by(ID_COL)
            .agg(pl.col("dob").min())
            .collect()
        )
        out = out.join(cb, on=ID_COL, how="left").with_columns(
            ((pl.col(DATE_COL) - pl.col("dob")).dt.total_days() / 365.25).alias("age_years")
        ).drop("dob")
    except Exception as exc:  # pragma: no cover - table not attached
        print("  diagnostics: age unavailable", exc)
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("age_years"))

    try:
        inc = (
            scan_table("train_person_1")
            .select([ID_COL, "mainoccupationinc_384A"])
            .group_by(ID_COL)
            .agg(pl.col("mainoccupationinc_384A").mean().alias("income"))
            .collect()
        )
        out = out.join(inc, on=ID_COL, how="left")
    except Exception:  # pragma: no cover
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("income"))

    try:
        prod = (
            scan_table("train_static_0")
            .select([ID_COL, "credtype_322L"])
            .group_by(ID_COL)
            .agg(pl.col("credtype_322L").first().alias("product_type"))
            .collect()
        )
        out = out.join(prod, on=ID_COL, how="left")
    except Exception:  # pragma: no cover
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("product_type"))

    # exposure proxy for the illustrative expected-loss table: the application's own credit amount
    try:
        amt = scan_table("train_static_0", [ID_COL, "credamount_770A"]).group_by(ID_COL).agg(pl.col("credamount_770A").max().alias("credit_amount")).collect()
        out = out.join(amt, on=ID_COL, how="left")
    except Exception:  # pragma: no cover
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("credit_amount"))

    return out.drop(DATE_COL)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Raw feature columns that are safe to model.

    Excluded: the bookkeeping columns, every column whose name ends in ``D``
    (the documented date transform - a raw date is an identifier, not a
    predictor) and any column that is not a scalar (polars list/struct columns
    appear in a few history tables and cannot be cast to a model matrix).
    """
    scalar = (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.String, pl.Boolean,
    )
    schema = df.schema
    out = []
    for c in df.columns:
        if c in (ID_COL, TARGET, WEEK_COL, DATE_COL) or c.endswith("D"):
            continue
        if schema[c] not in scalar:
            continue
        if is_identifier_feature(c):
            continue
        out.append(c)
    return out


_AGG_SUFFIXES = {
    "nunique", "maxcode", "mean", "max", "min", "std", "n_records", "recency_days",
    "rel_mean", "rel_max", "rel_min", "rel_std",
}


def is_identifier_feature(name: str) -> bool:
    """True for a raw identifier column, False for an aggregate over one.

    A count of distinct zip codes or a most-frequent employer across a client's
    history is a behavioural feature; a *raw* zip code, name or URL is an
    identifier the model must never see.  The rule is therefore applied to the
    base column name only when the feature is not an aggregation.
    """
    from hc_config import FORBIDDEN_TOKENS

    parts = name.split("__")
    base = parts[1] if len(parts) > 2 else parts[-1]
    is_aggregate = parts[-1] in _AGG_SUFFIXES
    if is_aggregate:
        return False
    return any(tok in base.lower() for tok in FORBIDDEN_TOKENS)


def to_model_matrix(
    df: pl.DataFrame,
    features: list[str],
    train_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """Cast to float32 and encode categoricals with a train-only vocabulary.

    Categorical raw columns are strings; they are encoded as integer codes whose
    vocabulary is built from the *training rows only*.  Categories unseen in
    training map to ``-1`` so the model sees "unknown", not a value taken from
    the future.
    """
    cats = [c for c in features if df.schema[c] == pl.String]
    nums = [c for c in features if c not in cats]
    out = df.select([pl.col(c).cast(pl.Float32) for c in nums]).to_pandas()
    vocab_size = {}
    for c in cats:
        col = df[c]
        train_values = col.filter(pl.Series(train_mask)).drop_nulls().unique().sort().to_list()
        vocab_size[c] = len(train_values)
        # A left join on a tiny vocabulary table is orders of magnitude faster
        # than a per-row Python mapping (1.5M rows x ~50 categorical columns) and
        # keeps the code deterministic: unseen levels become null, which the
        # model treats as missing instead of inventing a future category.
        mapping = pl.DataFrame(
            {c: train_values, "__code": np.arange(len(train_values), dtype="float32")}
        )
        coded = (
            df.select([c])
            .with_row_index("__row")
            .join(mapping, on=c, how="left")
            .sort("__row")["__code"]
            .to_numpy()
        )
        out[c] = coded
    return out[nums + cats].astype("float32"), {"categorical_vocab_sizes": vocab_size}


def univariate_screen(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
) -> dict:
    """Leakage gate: flag any single feature whose own AUC exceeds ``threshold``."""
    flags = []
    for col in X.columns:
        x = X[col].to_numpy(dtype="float64")
        mask = np.isfinite(x)
        if mask.sum() < 1000 or y[mask].min() == y[mask].max():
            continue
        a = auc(y[mask], x[mask])
        if np.isfinite(a) and (a > threshold or a < 1 - threshold):
            flags.append({"feature": col, "univariate_auc": float(max(a, 1 - a))})
    flags.sort(key=lambda r: -r["univariate_auc"])
    return {
        "threshold": threshold,
        "n_features_checked": int(X.shape[1]),
        "flagged": flags,
        "passed": len(flags) == 0,
    }
