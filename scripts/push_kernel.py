"""Push the generated notebook to Kaggle as a private CPU kernel with the competition attached.

    python scripts/push_kernel.py --message "full run" [--smoke 60000] [--dataset-fallback]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "vivekkopthsd/home-credit-stability-pd"
COMPETITION = "home-credit-credit-risk-model-stability"
DATASET_MIRROR = "usuarioml/home-credit-credit-risk-model-stability"


def log_attempt(message: str, status: str, note: str = "") -> None:
    path = ROOT / "logs" / "kaggle_attempts_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    hist.append({"n": len(hist) + 1, "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "message": message, "status": status, "note": note})
    path.write_text(json.dumps(hist, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="push")
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--dataset-fallback", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="separate kernel <slug>-gpu with enable_gpu (XGBoost on cuda)")
    ap.add_argument("--kdir", type=Path, default=Path(tempfile.gettempdir()) / "hc_pd_kernel")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    import build_notebook as bn
    import nbformat as nbf

    kdir = args.kdir
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    nbf.write(bn.build(args.smoke), str(kdir / "home-credit-stability-pd.ipynb"))
    kid = KERNEL_ID + "-gpu" if args.gpu else KERNEL_ID
    stem = kid.split("/")[1]
    nbf.write(bn.build(args.smoke), str(kdir / f"{stem}.ipynb"))
    meta = {
        "id": kid, "title": "Home Credit Stability PD GPU" if args.gpu else "Home Credit Stability PD", "code_file": f"{stem}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": bool(args.gpu), "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [DATASET_MIRROR] if args.dataset_fallback else [],
        "competition_sources": [] if args.dataset_fallback else [COMPETITION],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    proc = subprocess.run(["kaggle", "kernels", "push", "-p", str(kdir)], capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    print(out)
    ok = proc.returncode == 0 and "successfully" in out.lower()
    log_attempt(args.message, "pushed" if ok else "push_failed", out[:600])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
