"""Post-run housekeeping: pin requirements to the run environment, render docs.

    python scripts/finalize.py

* rewrites the version pins in ``requirements.txt`` from ``artifacts/environment.json``
  so the saved models load without a version warning;
* renders ``README.md`` from the artifacts;
* renders ``FINAL_REPORT.md`` from the artifacts and the attempt log.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

PINS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "optbinning": "optbinning",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "joblib": "joblib",
}


def repin() -> None:
    env_path = ART / "environment.json"
    if not env_path.exists():
        print("environment.json missing; leaving requirements.txt untouched")
        return
    env = json.loads(env_path.read_text(encoding="utf-8"))
    req = ROOT / "requirements.txt"
    lines = req.read_text(encoding="utf-8").splitlines()
    out = []
    pinned = set()
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)(.*)$", line)
        if not match:
            out.append(line)
            continue
        name = match.group(1)
        source = next((k for k, v in PINS.items() if v == name), None)
        version = env.get(source) if source else None
        if version:
            out.append(f"{name}=={version}{match.group(3)}")
            pinned.add(name)
        else:
            out.append(line)
    req.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"requirements.txt re-pinned from artifacts/environment.json ({len(pinned)} packages)")


def run(script: str) -> None:
    proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())


def main() -> int:
    repin()
    run("build_readme.py")
    run("scripts/build_final_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
