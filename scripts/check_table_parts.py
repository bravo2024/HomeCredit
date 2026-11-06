"""Post-run audit: which applications does each raw parquet file cover, by split?

The competition ships large tables as numbered parts (for example ``train_static_0_0`` and
``train_static_0_1``).  This script reads only the ``case_id`` column of every train file,
joins ``WEEK_NUM`` from ``train_base`` and counts, per file, the distinct applications that
appear in the TRAIN, VALID and OOT weeks of ``artifacts/split.json``.  No model is fitted.

    python scripts/check_table_parts.py [--data data/raw/parquet_files/train] [--art artifacts]

Output: ``artifacts/table_parts_check.json`` (written after the Kaggle run, so it is not listed
in ``artifacts/MANIFEST.json``).  ``build_readme.py`` reads it when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "parquet_files" / "train")
    ap.add_argument("--art", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args()

    split = json.loads((args.art / "split.json").read_text(encoding="utf-8"))
    b = split["bounds"]
    last_train, last_valid = b["train_weeks"][-1], b["valid_weeks"][-1]

    base = pl.read_parquet(args.data / "train_base.parquet").select(["case_id", "WEEK_NUM"])
    base = base.with_columns(
        pl.when(pl.col("WEEK_NUM") <= last_train).then(pl.lit("train"))
        .when(pl.col("WEEK_NUM") <= last_valid).then(pl.lit("valid"))
        .otherwise(pl.lit("oot")).alias("split")
    )
    totals = {r["split"]: int(r["n"]) for r in base.group_by("split").agg(pl.len().alias("n")).to_dicts()}

    files = []
    for path in sorted(args.data.glob("*.parquet")):
        if path.stem == "train_base":
            continue
        cases = pl.read_parquet(path, columns=["case_id"]).unique()
        j = cases.join(base, on="case_id", how="left")
        by = {r["split"]: int(r["n"]) for r in j.group_by("split").agg(pl.len().alias("n")).to_dicts() if r["split"]}
        files.append({
            "file": path.stem,
            "n_cases": int(cases.height),
            "week_min": int(j["WEEK_NUM"].min()),
            "week_max": int(j["WEEK_NUM"].max()),
            "cases_train": by.get("train", 0),
            "cases_valid": by.get("valid", 0),
            "cases_oot": by.get("oot", 0),
        })
        print(f"{path.stem:32s} cases={cases.height:>9,} weeks {files[-1]['week_min']}-{files[-1]['week_max']} "
              f"oot={files[-1]['cases_oot']:,}")

    out = {"split_totals": totals, "files": files,
           "note": "distinct case_id per file, joined to train_base WEEK_NUM; produced after the run by scripts/check_table_parts.py"}
    (args.art / "table_parts_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.art / "table_parts_check.json")


if __name__ == "__main__":
    main()
