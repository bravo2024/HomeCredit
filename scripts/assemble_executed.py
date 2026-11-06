"""Rebuild the executed notebook from a Kaggle run.

`kaggle kernels output` returns the log (JSON stream events) and the artifacts, not the executed .ipynb.
Every code cell ends with a marker print ``[end of code cell k]``, so the log can be split per cell.
Stored outputs are the cell's real stdout/stderr from the run; saved figures are attached to the cell
that wrote them.

    python scripts/assemble_executed.py <pull_dir> <out.ipynb>
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_notebook as bn  # noqa: E402

MARK = re.compile(r"\[end of code cell (\d+)\]")


def main(pull: Path, out: Path) -> None:
    log = next(pull.glob("*.log"))
    text = log.read_text(encoding="utf-8")
    events = json.loads(text[text.index("["):])
    nb = bn.build(None)
    code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    per_cell: dict[int, list[tuple[str, str]]] = {}
    cur = 1
    for ev in events:
        data = ev.get("data", "")
        stream = ev.get("stream_name", "stdout")
        m = MARK.search(data)
        if m:
            before = data[: m.start()]
            if before:
                per_cell.setdefault(cur, []).append((stream, before))
            per_cell.setdefault(cur, []).append((stream, m.group(0) + "\n"))
            cur = int(m.group(1)) + 1
            rest = data[m.end():].lstrip("\n")
            if rest and not rest.startswith("[NbConvert"):
                per_cell.setdefault(cur, []).append((stream, rest))
        else:
            if "NbConvertApp" in data or "mistune" in data or "filter_links" in data:
                continue
            per_cell.setdefault(cur, []).append((stream, data))
    figs = ART = pull / "artifacts" / "figures"
    for k, i in enumerate(code_idx, start=1):
        cell = nb.cells[i]
        cell.execution_count = k
        outs = []
        for stream in ("stdout", "stderr"):
            s = "".join(d for st, d in per_cell.get(k, []) if st == stream)
            if s:
                outs.append(new_output("stream", name=stream, text=s))
        for png in re.findall(r'figures" / "([a-z_]+\.png)"', cell.source):
            f = figs / png
            if f.exists():
                outs.append(new_output("display_data", data={"image/png": base64.b64encode(f.read_bytes()).decode()},
                                       metadata={}))
        cell.outputs = outs
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out))
    n_empty = sum(1 for i in code_idx if not nb.cells[i].outputs)
    print(f"wrote {out}: {len(code_idx)} code cells, {n_empty} without outputs")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
