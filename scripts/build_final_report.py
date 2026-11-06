"""Render FINAL_REPORT.md from artifacts/ only (what ran, what passed, what did not).

    python scripts/build_final_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def J(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def main() -> None:
    env, gates, champ = J("environment.json"), J("quality_gates.json"), J("champion.json")
    scj, split, boot = J("scorecard.json"), J("split.json"), J("bootstrap_ci.json")
    L = ["# Final report", ""]
    L.append(f"Kernel run: `{env.get('kernel')}`, runtime {env.get('runtime_seconds', 0) / 60:.0f} min, python {env['python']}, "
             f"xgboost {env['xgboost']}, lightgbm {env['lightgbm']}, scikit-learn {env['sklearn']}, sample_cases={env['sample_cases']}.")
    L.append("")
    L.append("## Data and split")
    for r in split["table"]:
        L.append(f"* {r['split']}: {int(r['n']):,} applications, default rate {r['default_rate']:.4f}, weeks {int(r['week_min'])}-{int(r['week_max'])}")
    L.append("")
    L.append(f"## Champion\n`{champ['champion']}`. Rule: {champ['rule']}. Frozen at {champ['frozen_at']} before OOT was scored (`artifacts/champion.json`).")
    L.append("")
    L.append("## Scorecard sign check")
    L.append(f"Before the fix {scj['sign_consistent_before_fix']}/{scj['sign_total_before_fix']}; after {scj['sign_consistent']}/{scj['sign_total']} "
             f"({len(scj['sign_fix_log'])} variables dropped).")
    L.append("")
    L.append("## Paired bootstrap, OOT Gini difference")
    for k, v in boot["results"].items():
        L.append(f"* {k}: {v['point_gini_diff']:+.4f} (95% CI {v['ci95_lo']:+.4f} to {v['ci95_hi']:+.4f}, {v['n_boot']} resamples)")
    L.append("")
    L.append("## Quality gates")
    for g in gates["gates"]:
        L.append(f"* {g['status']}: {g['gate']}")
    L.append("")
    L.append(f"{gates['n_pass']} pass, {gates['n_warn']} warn, {gates['n_fail']} fail.")
    L.append("")
    L.append("## Not done / limits")
    L.append("* Random forest is fitted on a 500k-row TRAIN subsample (compute cap).")
    L.append("* Expected loss uses an assumed LGD (0.45) and is illustrative.")
    L.append("* Details and all other numbers: `README.md` and `artifacts/`.")
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote FINAL_REPORT.md")


if __name__ == "__main__":
    main()
