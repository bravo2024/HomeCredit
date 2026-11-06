"""Poll the Kaggle kernel every 150 s, then pull the outputs and the executed notebook."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "vivekkopthsd/home-credit-stability-pd"
POLL_S = 150
MAX_HOURS = 8.0


def status() -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True
    )
    text = (proc.stdout + proc.stderr).strip()
    match = re.search(r'has status "([^"]+)"', text)
    if match:
        return match.group(1)
    if "404" in text or "not found" in text.lower():
        return "not_found"
    return f"unknown: {text[:200]}"


def main() -> int:
    deadline = time.time() + MAX_HOURS * 3600
    last = ""
    while time.time() < deadline:
        current = status()
        stamp = datetime.now().strftime("%H:%M:%S")
        if current != last:
            print(f"[{stamp}] {current}", flush=True)
            last = current
        if current in ("complete", "error", "canceled", "cancelacknowledged"):
            break
        time.sleep(POLL_S)
    else:
        print("poll timeout", flush=True)
        return 2

    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    print("downloading kernel output", flush=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(out)], check=False)

    # Kaggle names the executed notebook after the kernel slug.
    candidates = sorted(out.glob("*.ipynb"), key=lambda p: p.stat().st_mtime)
    executed = [p for p in candidates if p.name != "HomeCreditStability.ipynb"]
    if executed:
        target = out / "HomeCreditStability_executed.ipynb"
        if target.exists():
            target.unlink()
        executed[-1].rename(target)
        print(f"executed notebook: {target}", flush=True)

    print("final status:", status(), flush=True)
    return 0 if last == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
