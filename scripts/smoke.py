"""Local smoke run: the whole pipeline on a few thousand applications.

This exists so that the expensive Kaggle kernel is not the first place the code
runs end to end.  It exercises every module against the real tables but on a
sample, and it never writes anything into ``artifacts/`` - smoke numbers must not
leak into a README.

    python scripts/smoke.py [n_cases]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import hc_pipeline as pipe  # noqa: E402


def main(n_cases: int = 4000) -> int:
    t0 = time.time()
    print(f"[1/7] dataset for {n_cases} applications")
    ds = pipe.build_dataset(sample_cases=n_cases, verbose=False)
    print(f"      rows={len(ds['y'])} features={ds['X'].shape[1]} cats={len(ds['cat_cols'])} "
          f"({time.time() - t0:.0f}s)")
    print("      splits:", pipe.split_profile(ds))

    print("[2/7] screening")
    screen = pipe.feature_screen(ds)
    print(f"      tree={len(screen['tree_features'])} linear={len(screen['linear_numeric'])} "
          f"woe={len(screen['woe_features'])} quarantined={len(screen['quarantined'])}")

    cov = pipe.feature_coverage(ds, {"tree": screen["tree_features"], "linear": screen["linear_numeric"],
                                     "woe": screen["woe_features"]})
    print(f"      coverage gate: final={cov['n_final_features']} null_valid={cov['n_null_in_valid']} "
          f"null_oot={cov['n_null_in_oot']} passed={cov['passed']}")
    pipe.coverage_gate(cov, raise_on_fail=True)
    cal = pipe.calendar_gate(sorted(set(screen["tree_features"]) | set(screen["linear_numeric"]) | set(screen["woe_features"])), raise_on_fail=True)
    pg = pipe.calendar_proxy_gate(screen["calendar_proxy"], sorted(set(screen["tree_features"]) | set(screen["linear_numeric"]) | set(screen["woe_features"])), raise_on_fail=True)
    print("      proxy gate:", pg["status"], pg["detail"]["n_dropped"], "of", pg["detail"]["n_screened"], "rel features dropped")
    print("      calendar gate:", cal["status"], cal["detail"]["n_relative_features"], "relative-time features")

    print("[3/7] linear design")
    design, used = pipe.build_linear_design(ds, screen["linear_numeric"])
    print(f"      columns={design.shape[1]} one-hot cats={len(used)}")

    print("[4/7] scorecard")
    sc = pipe.build_scorecard(ds, screen["woe_features"])
    print(f"      retained={len(sc['features'])} vif_drops={len(sc['vif_log'])} "
          f"sign_ok={sum(r['consistent'] for r in sc['sign_check'])}/{len(sc['sign_check'])}")
    print("      point rows:", len(sc["points"]))

    print("[5/7] leakage gate")
    from hc_features import linear_pipeline

    lp = linear_pipeline()
    lp.fit(design[ds["train_mask"]], ds["y"][ds["train_mask"]])
    gate = pipe.leakage_gate(ds, screen, list(design.columns), sc=sc, linear_pipe=lp, linear_design=design)
    print("      ", [c["status"] for c in gate["checks"]])

    print("[6/7] roster")
    roster = pipe.fit_roster(ds, screen, design, sc, lr_config={"penalty": "l2", "C": 0.2, "l1_ratio": None})
    results = {name: pipe.evaluate_model(name, r["p_valid"], r["p_oot"], ds) for name, r in roster.items()}
    for name, r in results.items():
        print(f"      {name:16s} valid_auc={r['valid']['auc']:.4f} "
              f"valid_stab={r['valid']['stability']['stability_metric']:.4f} "
              f"oot_auc={r['oot']['auc']:.4f}")

    print("[7/7] monitoring")
    champ = pipe.choose_champion(results)
    mon = pipe.build_monitoring(
        roster[champ]["p_valid"], roster[champ]["p_oot"], ds, screen["tree_features"]
    )
    print(f"      psi_alert={mon['n_alert']} psi_watch={mon['n_watch']} "
          f"score_psi={mon['score_drift']['psi_train_to_oot']:.4f}")
    print(f"SMOKE OK in {time.time() - t0:.0f}s (no artifacts written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
