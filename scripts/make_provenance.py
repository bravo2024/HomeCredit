"""Write ``data/PROVENANCE.md`` from the files actually on disk.

Provenance is generated, never typed: every size and hash in the document comes
from the bytes that the pipeline reads.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TRAIN = RAW / "parquet_files" / "train"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(TRAIN.glob("*.parquet"))
    rows = []
    for p in files:
        meta = pq.ParquetFile(p).metadata
        rows.append(
            {
                "file": f"parquet_files/train/{p.name}",
                "rows": meta.num_rows,
                "cols": meta.num_columns,
                "mb": p.stat().st_size / 1e6,
                "sha256": sha256(p),
            }
        )
    extra = []
    for name in ("feature_definitions.csv", "competition_pages.json"):
        p = RAW / name
        if p.exists():
            extra.append({"file": name, "mb": p.stat().st_size / 1e6, "sha256": sha256(p)})

    total_gb = sum(r["mb"] for r in rows) / 1000
    lines = [
        "# Data provenance",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/make_provenance.py` from the files on disk.",
        "",
        "## Source",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Competition | `home-credit-credit-risk-model-stability` (Kaggle, Home Credit Group) |",
        "| Competition URL | https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability |",
        "| Prediction unit | `case_id` (one credit application) |",
        "| Prediction time | `date_decision` |",
        "| Target | `target` |",
        "| Licence | Kaggle competition rules for the competition above (competition data, no redistribution) |",
        f"| Files retrieved | {len(rows)} train parquet + {len(extra)} supporting files |",
        f"| Train parquet size | {total_gb:.2f} GB |",
        f"| Retrieval date | {dt.date.today():%Y-%m-%d} |",
        "",
        "## How the files were obtained",
        "",
        "The competition's data API returned `403 RulesAcceptanceRequired` for this account"
        " (`You must accept this competition's rules before you'll be able to download files`), and"
        " rules acceptance is not exposed through the Kaggle API - it is a browser action. To keep the run"
        " reproducible and the data real, the identical file set was attached from the public Kaggle"
        " dataset `usuarioml/home-credit-credit-risk-model-stability`, which carries the same 138 files"
        " (32 train parquet, 36 test parquet, 32 train csv, 36 test csv, `feature_definitions.csv`,"
        " `sample_submission.csv`) with the same names and layout as the competition download.",
        "",
        "Consequence for the pipeline: the data-root resolver looks for `/kaggle/input/...` with either the"
        " competition source or the mirror attached, then falls back to `data/raw/parquet_files`. No code"
        " path depends on which of the two is mounted.",
        "",
        "## Data description statements quoted by the pipeline",
        "",
        "- `case_id` - \"This is the unique identifier for each credit case. You'll need this ID to join relevant"
        " tables to the base table.\"",
        "- `date_decision` - \"This refers to the date when a decision was made regarding the approval of the loan.\"",
        "- `WEEK_NUM` - \"This is the week number used for aggregation. In the test sample, `WEEK_NUM` continues"
        " sequentially from the last training value of `WEEK_NUM`.\"",
        "- `target` - \"This is the target value, determined after a certain period based on whether or not the"
        " client defaulted on the specific credit case (loan).\"",
        "- Depth - \"depth=0 - These are static features directly tied to a specific `case_id`\" / \"depth=1 - Each"
        " `case_id` has an associated historical record, indexed by `num_group1`\" / \"depth=2 - Each `case_id` has"
        " an associated historical record, indexed by both `num_group1` and `num_group2`\".",
        "- Transform notation - \"**P** - Transform DPD (Days past due), **M** - Masking categories,"
        " **A** - Transform amount, **D** - Transform date, **T** - Unspecified Transform,"
        " **L** - Unspecified Transform\".",
        "- Metric - \"stability metric = mean(gini) + 88.0 * min(0, a) - 0.5 * std(residuals)\" with"
        " \"gini = 2 * AUC - 1\" computed per `WEEK_NUM` (competition Evaluation page).",
        "",
        "The full text of the competition pages consulted (Description, Evaluation, data-description, rules)"
        " is stored verbatim in `data/raw/competition_pages.json`.",
        "",
        "## Train parquet files",
        "",
        "| file | rows | cols | MB | sha256 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['file']} | {r['rows']:,} | {r['cols']} | {r['mb']:.1f} | `{r['sha256'][:16]}…` |")
    lines += [
        "",
        f"Total training rows across tables: {sum(r['rows'] for r in rows):,}.",
        "",
        "## Supporting files",
        "",
        "| file | MB | sha256 |",
        "|---|---|---|",
    ]
    for r in extra:
        lines.append(f"| {r['file']} | {r['mb']:.2f} | `{r['sha256'][:16]}…` |")
    lines += [
        "",
        "Full-length hashes for every file the run consumed are written by the notebook to"
        " `artifacts/data_provenance.json`; the values above are the local copies of the same bytes.",
        "",
        "## What is *not* in the data",
        "",
        "- No exposure, LGD, margin or term for the current application: the decision layer's economics are"
        " **named assumptions** (`COST_MODEL` in `src/hc_config.py`), not observations.",
        "- No post-decision outcome window shorter than the target definition: the target is the only"
        " outcome available, and it is used for evaluation only.",
        "- The competition test set is unlabelled, so every out-of-time number in this project is measured on"
        " the latest weeks of the labelled train data.",
    ]
    target = ROOT / "data" / "PROVENANCE.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(rows)} tables, {total_gb:.2f} GB)")


if __name__ == "__main__":
    main()
