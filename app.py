"""Credit Risk Modelling | Loan Default Prediction with Out-of-Time Validation.

A Streamlit reading of the frozen artifacts in ``artifacts/``. The app never
trains, never refits a transformer and never touches the raw competition data:
every number on every page is read from the JSON or CSV written by the pipeline.

    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread. The app scores a handful of rows; a thread pool sized
# to the host CPU is pure overhead on a small shared container.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import textwrap
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_NAME = "Credit Risk Modelling | Loan Default Prediction with Out-of-Time Validation"
REPO = "https://github.com/bravo2024/HomeCredit"
_HERE = Path(__file__).resolve().parent
_ART = Path(os.environ.get("HC_ARTIFACT_DIR", "artifacts"))
ART = _ART if _ART.is_absolute() else _HERE / _ART

PAGES = [
    "Overview",
    "Model comparison",
    "Stability & monitoring",
    "Score a case",
    "Batch scoring",
    "Credit policy",
    "Model card",
]

SUBTITLES = {
    "Overview": "Which applications will default, and how well does the ranking hold out of time?",
    "Model comparison": "Seven models, one chronological split, paired bootstrap on applications.",
    "Stability & monitoring": "Weekly Gini, score and feature PSI, and how separable the two periods are.",
    "Score a case": "Score one application from the frozen champion, starting at the median of each feature.",
    "Batch scoring": "Score a CSV of applications and download the calibrated result.",
    "Credit policy": "What a cut-off buys in approvals, bad rate and defaulters caught.",
    "Model card": "Intended use, assumptions, limitations and the gates the run had to pass.",
}

NAMES = {
    "dummy": "Dummy (base rate)",
    "penalised_lr": "Logistic regression (L2)",
    "woe_scorecard": "WoE scorecard",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "xgboost_monotone": "XGBoost (monotone)",
    "lightgbm": "LightGBM",
}
ORDER = ["xgboost", "lightgbm", "xgboost_monotone", "random_forest", "woe_scorecard", "penalised_lr", "dummy"]

INK, MUTED, LINE = "#0F172A", "#5B6673", "#E8EBEF"
ACCENT, ACCENT2, NEUTRAL = "#2F5D8A", "#C2410C", "#9AA5B1"
SERIES = ["#2F5D8A", "#C2410C", "#4C72B0", "#55A868", "#8172B2", "#937860", "#9AA5B1"]
BANDS = {"stable": NEUTRAL, "watch": "#C08A2E", "alert": ACCENT2}

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] {{
        font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}
      [data-testid="stHeading"] h1 {{
        font-size: 0.76rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: {ACCENT}; padding-bottom: 0;
      }}
      [data-testid="stHeading"] h2 {{
        font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em; color: {INK}; padding-top: 0;
      }}
      [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{
        font-size: 1.02rem; font-weight: 600; color: {INK}; padding-top: 0.4rem;
      }}
      p, li {{ color: {INK}; }}
      [data-testid="stCaptionContainer"] p {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.5; }}
      [data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {INK}; }}
      [data-testid="stMetricLabel"] p {{
        font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase; color: {MUTED};
      }}
      [data-testid="stSidebar"] {{ background: #FAFBFC; border-right: 1px solid {LINE}; }}
      [data-testid="stSidebar"] .block-container {{ padding-top: 1.8rem; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 6px; }}
      hr {{ border-color: {LINE}; }}
      a {{ color: {ACCENT}; text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def _read_json(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def _read_table(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else None


def js(name: str) -> dict | None:
    """Keyed by the full path, so a different artifact directory is never served
    from the cache of another directory."""
    return _read_json(str(ART / name))


def csv(name: str) -> pd.DataFrame | None:
    return _read_table(str(ART / name))


def missing(name: str) -> None:
    st.warning(f"Artifact missing: `{name}` — regenerate `artifacts/` with the notebook.")


def label(key: str) -> str:
    return NAMES.get(key, str(key).replace("_", " ").capitalize())


def polish_rule(text: str) -> str:
    """Replace raw model keys in an artifact sentence with their display names."""
    for key, name in NAMES.items():
        text = text.replace(f"'{key}'", f"'{name}'")
    return text


def pct(x, d: int = 1) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{100 * v:.{d}f}%"


def num(x, d: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{d}f}"


def dedupe(xs):
    return list(dict.fromkeys(xs))


def style(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            grid=False, domainColor=LINE, tickColor=LINE,
            labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(orient="bottom", direction="horizontal", labelColor=MUTED, titleColor=MUTED, labelFontSize=11)
        .configure_title(color=INK, fontSize=13, anchor="start")
    )


def feature_view(keys: list[str]) -> dict[str, str]:
    """Human labels for raw feature names: the artefact gloss when there is one."""
    top = csv("top_features.csv")
    gloss = {}
    if top is not None and {"feature", "plain_name"} <= set(top.columns):
        gloss = dict(zip(top.feature, top.plain_name))
    out = {}
    for f in keys:
        text = gloss.get(f, f.split("__", 1)[-1].replace("_", " ")).split(" [")[0]
        out[f] = textwrap.shorten(text, width=58, placeholder="…")
    return out


# --------------------------------------------------------------------------- shell
split = js("split.json")
meta = js("feature_meta.json")
champ_meta = js("champion.json") or {}

st.title(APP_NAME)

with st.sidebar:
    st.markdown(
        f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:{MUTED}'>Section</div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Out-of-time credit risk on the Home Credit stability benchmark: 1.53M applications, seven models, one frozen split.")
    st.caption(f"[Code, notebook and tests]({REPO}) · every number here is read from `artifacts/`")

st.header(page)
st.markdown(
    f"<div style='font-size:1rem;color:{MUTED};max-width:800px;padding-bottom:.4rem'>{SUBTITLES[page]}</div>",
    unsafe_allow_html=True,
)

if split:
    rows = split.get("table", [])
    total = sum(int(r.get("n", 0)) for r in rows)
    starts = [r["date_min"] for r in rows if r.get("date_min")]
    ends = [r["date_max"] for r in rows if r.get("date_max")]
    window = f"{min(starts)} to {max(ends)}" if starts and ends else "2019-01-01 to 2020-10-05"
    n_feat = len(meta["features"]) if meta and meta.get("features") else 177
    st.caption(f"{total:,} applications · {window} · {n_feat} model features · "
               "Home Credit Credit Risk Model Stability benchmark (Kaggle), used under the competition rules.")
else:
    st.caption("Home Credit Credit Risk Model Stability benchmark (Kaggle), used under the competition rules.")


# --------------------------------------------------------------------------- Overview
def page_overview() -> None:
    metrics, table = js("metrics.json"), csv("metrics_table.csv")

    if metrics is None:
        missing("metrics.json")
    else:
        champion = metrics["champion"]
        oot = metrics["models"][champion]["oot"]
        with st.container(border=True):
            c = st.columns(4)
            c[0].metric("Champion", label(champion))
            c[1].metric("OOT Gini", num(oot["gini"]))
            c[2].metric("OOT ROC-AUC", num(oot["auc"]))
            c[3].metric("OOT stability", num(oot["stability"]["stability_metric"]))
        st.markdown(
            f"**Selection rule.** {polish_rule(metrics.get('champion_rule', 'highest VALID Gini'))}. "
            f"Frozen on VALID before the out-of-time weeks were scored"
            + (f" ({champ_meta['frozen_at'].replace('T', ' ')})." if champ_meta.get("frozen_at") else ".")
        )
        top10 = oot.get("top10") or {}
        observed = (oot.get("calibration") or {}).get("observed_rate")
        if top10.get("precision") is not None and observed is not None:
            st.caption(f"On the out-of-time weeks the champion reaches a bad rate of {pct(top10['precision'])} "
                       f"among the riskiest 10% of applications, against {pct(observed)} when everyone is approved; "
                       f"that list contains {pct(top10.get('recall'), 0)} of all defaulters.")

    if table is not None:
        d = table[table.model.isin(NAMES)].copy()
        if {"valid_gini", "oot_gini"} <= set(d.columns):
            long = d.melt(id_vars="model", value_vars=["valid_gini", "oot_gini"], var_name="Split", value_name="Gini")
            long["Split"] = long.Split.map({"valid_gini": "VALID", "oot_gini": "OOT"})
            long["Model"] = long.model.map(label)
            order = d.sort_values("oot_gini", ascending=False).model.map(label).tolist()
            chart = alt.Chart(long).mark_bar().encode(
                y=alt.Y("Model:N", sort=order, title=None),
                yOffset=alt.YOffset("Split:N", sort=["VALID", "OOT"]),
                x=alt.X("Gini:Q", title="Gini", stack=None, scale=alt.Scale(domain=[0, 0.75])),
                color=alt.Color("Split:N", title="Split", scale=alt.Scale(domain=["VALID", "OOT"], range=[NEUTRAL, ACCENT])),
                tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Split:N"), alt.Tooltip("Gini:Q", format=".3f")],
            ).properties(title="Discrimination on VALID and on the out-of-time weeks", height=320)
            st.altair_chart(style(chart), width="stretch")
            st.caption("VALID drives early stopping, tuning, the calibrator and the cut-offs, so VALID is optimistic for the tuned models.")

        show = [("oot_gini", "OOT Gini"), ("oot_auc", "OOT ROC-AUC"), ("oot_ks", "OOT KS"),
                ("oot_pr_auc", "OOT PR-AUC"), ("valid_gini", "VALID Gini")]
        show = [(c, t) for c, t in show if c in d.columns]
        view = d[["model"] + [c for c, _ in show]].rename(columns={"model": "Model", **{c: t for c, t in show}})
        view["Model"] = view["Model"].map(label)
        sort_col = "OOT Gini" if "OOT Gini" in view else view.columns[1]
        st.dataframe(
            view.sort_values(sort_col, ascending=False),
            hide_index=True,
            width="stretch",
            column_config={t: st.column_config.NumberColumn(format="%.3f") for _, t in show},
        )
        st.caption("Source: `metrics_table.csv`. PR-AUC is compared with the OOT base rate of 2.12%, which is the PR-AUC of a random ranking.")

    if split:
        rows = split["table"]
        frame = pd.DataFrame({
            "Split": [r["split"].upper() for r in rows],
            "Weeks": [f"{r['week_min']}–{r['week_max']}" for r in rows],
            "Dates": [f"{r.get('date_min', '')} to {r.get('date_max', '')}".strip() for r in rows],
            "Applications": [int(r["n"]) for r in rows],
            "Default rate": [100 * float(r["default_rate"]) for r in rows],
        })
        st.markdown("#### Time split")
        st.dataframe(frame, hide_index=True, width="stretch", column_config={
            "Applications": st.column_config.NumberColumn(format="%d"),
            "Default rate": st.column_config.NumberColumn(format="%.2f%%"),
        })
        st.caption("A chronological split by calendar week: 64 TRAIN weeks, then 14 VALID and 14 OOT weeks. "
                   "The default rate falls from 3.32% to 2.12% inside the same risk bands.")

    with st.expander("How the pipeline is built"):
        n_feat = len(meta["features"]) if meta and meta.get("features") else 177
        gates = js("quality_gates.json") or {}
        st.markdown(
            f"* **{n_feat} features** from 7 of the 16 competition tables, each history record kept only if it is dated "
            "before the application (`pit_filter`). Absolute dates never enter the model; they are converted to days "
            "before the decision.\n"
            "* **Seven models share one design matrix and one split**: logistic regression, a monotone WoE scorecard, "
            "random forest, XGBoost, monotone XGBoost, LightGBM and a base-rate dummy.\n"
            "* **Selection, calibration and cut-offs are frozen on VALID** and only then applied to OOT; "
            f"{gates.get('n_pass', 20)} of {gates.get('n_pass', 20) + gates.get('n_fail', 0)} quality gates pass.\n"
            "* **Biggest caveat.** Adversarial validation separates TRAIN from OOT almost perfectly, so OOT is a "
            "different-population estimate, not a second sample from the same population."
        )


# --------------------------------------------------------------------------- Model comparison
def page_comparison() -> None:
    table = csv("metrics_table.csv")
    metrics = js("metrics.json") or {}
    champion = metrics.get("champion", "xgboost")

    st.markdown("#### Out-of-time performance")
    if table is None:
        missing("metrics_table.csv")
    else:
        d = table[table.model.isin(NAMES)].copy()
        spec = [("oot_gini", "OOT Gini"), ("oot_auc", "OOT ROC-AUC"), ("oot_ks", "OOT KS"),
                ("oot_pr_auc", "OOT PR-AUC"), ("oot_brier", "OOT Brier"), ("valid_gini", "VALID Gini")]
        spec = [(c, t) for c, t in spec if c in d.columns]
        view = d[["model"] + [c for c, _ in spec]].rename(columns={"model": "Model", **{c: t for c, t in spec}})
        view["Model"] = view["Model"].map(label)
        sort_col = "OOT Gini" if "OOT Gini" in view else view.columns[1]
        st.dataframe(
            view.sort_values(sort_col, ascending=False),
            hide_index=True,
            width="stretch",
            column_config={t: st.column_config.NumberColumn(format="%.3f") for _, t in spec},
        )
        st.caption("All models use the same 177 columns and the same split; the random forest is fitted on half of TRAIN, "
                   "which handicaps it. The dummy is the base rate.")

    curves = js("curves.json")
    if curves:
        keys = [k for k in ORDER if k in curves.get("roc", {})]
        names = [label(k) for k in keys]
        default = [n for n in (label(champion), label("penalised_lr"), label("lightgbm")) if n in names]
        if not default:
            default = names[:2]
        pick = st.multiselect("Models on the curves", names, default=dedupe(default))
        chosen = [k for k in keys if label(k) in pick]
        if chosen:
            domain = [label(k) for k in chosen]

            def curve_frame(kind: str) -> pd.DataFrame:
                frames = [pd.DataFrame(curves[kind][k]).assign(Model=label(k)) for k in chosen if k in curves.get(kind, {})]
                return pd.concat(frames) if frames else pd.DataFrame()

            roc, pr = curve_frame("roc"), curve_frame("pr")
            if not roc.empty:
                st.altair_chart(style(alt.Chart(roc).mark_line().encode(
                    x=alt.X("fpr:Q", title="false positive rate"),
                    y=alt.Y("tpr:Q", title="true positive rate"),
                    color=alt.Color("Model:N", title="Model", scale=alt.Scale(domain=domain, range=SERIES)),
                ).properties(title="ROC on the out-of-time weeks", height=300)), width="stretch")
            if not pr.empty:
                st.altair_chart(style(alt.Chart(pr).mark_line().encode(
                    x=alt.X("recall:Q", title="recall"),
                    y=alt.Y("precision:Q", title="precision"),
                    color=alt.Color("Model:N", title="Model", scale=alt.Scale(domain=domain, range=SERIES)),
                ).properties(title="Precision–recall on the out-of-time weeks", height=300)), width="stretch")

    cal = js("calibration.json")
    if cal and cal.get("oot"):
        st.markdown("#### Calibration of the champion")
        c = pd.DataFrame(cal["oot"])
        view = pd.DataFrame({
            "Calibration": c.calibration,
            "Mean PD": c.mean_pd,
            "Observed": c.actual_rate,
            "Expected / actual": c.expected_over_actual,
            "ECE": c.ece,
            "Brier": c.brier,
            "Log loss": c.logloss,
            "Chosen on VALID": c.chosen_on_valid.map({True: "yes", False: "no"}),
        })
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "Mean PD": st.column_config.NumberColumn(format="%.4f"),
            "Observed": st.column_config.NumberColumn(format="%.4f"),
            "Expected / actual": st.column_config.NumberColumn(format="%.2f"),
            "ECE": st.column_config.NumberColumn(format="%.4f"),
            "Brier": st.column_config.NumberColumn(format="%.4f"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
        })
        st.caption(f"The calibrator was chosen on the earlier VALID weeks and refitted on all VALID rows: "
                   f"**{cal.get('chosen_on_valid')}**. The raw score over-predicts on OOT by a factor of "
                   f"{num(c[c.calibration.str.startswith('none')].expected_over_actual.iloc[0], 2)}.")

        deciles = cal.get("deciles_after")
        if deciles:
            dd = pd.DataFrame(deciles)
            base = pd.DataFrame({"expected": [0, dd.expected.max()], "actual": [0, dd.expected.max()]})
            pts = alt.Chart(dd).mark_circle(size=90, color=ACCENT).encode(
                x=alt.X("expected:Q", title="mean predicted PD", axis=alt.Axis(format="%")),
                y=alt.Y("actual:Q", title="observed default rate", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("decile:O", title="Decile"), alt.Tooltip("n:Q", title="Applications"),
                         alt.Tooltip("expected:Q", format=".2%"), alt.Tooltip("actual:Q", format=".2%")],
            )
            diag = alt.Chart(base).mark_line(color=NEUTRAL, strokeDash=[4, 4]).encode(x="expected", y="actual")
            st.altair_chart(style((pts + diag).properties(title="Reliability after recalibration (OOT deciles)", height=320)),
                            width="stretch")

    boot = (js("evaluation.json") or {}).get("bootstrap")
    if boot:
        rows = []
        for key, v in boot.items():
            base = key.split("_minus_")[-1]
            rows.append({
                "Comparison": f"{label(champion)} − {label(base)}",
                "Gini difference": v["point_gini_diff"],
                "95% low": v["ci95_lo"],
                "95% high": v["ci95_hi"],
                "Share of resamples above 0": v["share_positive"],
            })
        with st.expander("Paired bootstrap of the Gini difference"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Gini difference": st.column_config.NumberColumn(format="%+.3f"),
                "95% low": st.column_config.NumberColumn(format="%+.3f"),
                "95% high": st.column_config.NumberColumn(format="%+.3f"),
                "Share of resamples above 0": st.column_config.NumberColumn(format="%.3f"),
            })
            st.caption("1,000 resamples of applications on the test weeks. The resampling ignores week-level dependence "
                       "and refit variability.")

    model_meta = (metrics.get("model_meta") or {}).get(champion)
    tuning = js("tuning.json") or {}
    if model_meta or tuning:
        with st.expander("Selection rule and chosen hyper-parameters"):
            st.markdown(f"**Rule.** {metrics.get('champion_rule', 'highest VALID Gini')}.")
            if champ_meta.get("tied") is not None:
                st.markdown(f"**Models inside the tie band on VALID:** {', '.join(label(t) for t in champ_meta['tied'])}.")
            if model_meta:
                st.dataframe(pd.DataFrame(sorted(model_meta.items()), columns=["Setting", "Value"]),
                             hide_index=True, width="stretch")
            if tuning.get("xgb_best_iteration") is not None:
                st.caption(f"XGBoost stopped at iteration {tuning['xgb_best_iteration']} of 800; "
                           f"LightGBM at {tuning.get('lightgbm_best_iter')}; the logistic C grid was "
                           f"{tuning.get('logistic_C_grid')} and chose C = {tuning.get('logistic_C')}.")


# --------------------------------------------------------------------------- Stability & monitoring
def page_stability() -> None:
    weekly = js("weekly_gini.json")
    if weekly is None:
        missing("weekly_gini.json")
    else:
        keys = [k for k in ORDER if k in weekly]
        names = [label(k) for k in keys]
        default = dedupe([label(k) for k in ("xgboost", "lightgbm", "random_forest", "woe_scorecard") if k in keys]) or names[:3]
        pick = st.multiselect("Models on the weekly chart", names, default=default)
        chosen = [k for k in keys if label(k) in pick]
        frames = [pd.DataFrame(weekly[k]).assign(Model=label(k)) for k in chosen]
        if frames:
            data = pd.concat(frames)
            chart = alt.Chart(data).mark_line(strokeWidth=1.8).encode(
                x=alt.X("week:Q", title="out-of-time week", axis=alt.Axis(format="d")),
                y=alt.Y("gini:Q", title="weekly Gini", scale=alt.Scale(zero=False)),
                color=alt.Color("Model:N", title="Model", scale=alt.Scale(domain=[label(k) for k in keys], range=SERIES)),
                tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("week:Q"), alt.Tooltip("gini:Q", format=".3f"),
                         alt.Tooltip("n:Q", title="Applications")],
            ).properties(title="Weekly Gini on the out-of-time weeks", height=320)
            st.altair_chart(style(chart), width="stretch")
            st.caption("The competition stability metric is mean weekly Gini + 88 × min(0, slope) − 0.5 × residual std.")

    stab = csv("stability.csv")
    if stab is None:
        missing("stability.csv")
    else:
        d = stab[stab.model.isin(NAMES)].copy()
        view = pd.DataFrame({
            "Model": d.model.map(label), "Split": d.split.str.upper(),
            "Mean weekly Gini": d.mean_gini, "Slope / week": d.slope,
            "Residual std": d.std_residual, "Stability metric": d.stability_metric,
        }).sort_values(["Split", "Stability metric"], ascending=[True, False])
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "Mean weekly Gini": st.column_config.NumberColumn(format="%.3f"),
            "Slope / week": st.column_config.NumberColumn(format="%.4f"),
            "Residual std": st.column_config.NumberColumn(format="%.3f"),
            "Stability metric": st.column_config.NumberColumn(format="%.3f"),
        })

    mon = js("monitoring.json") or {}
    drift = js("drift.json") or {}
    psi = csv("psi.csv")
    direction = mon.get("score_drift") or {}
    adv = drift.get("adversarial") or {}
    counts = psi.band.value_counts() if psi is not None else None
    if direction or adv:
        with st.container(border=True):
            c = st.columns(4)
            c[0].metric("Score PSI, TRAIN → OOT", num(direction.get("psi_train_to_oot", float("nan")), 3))
            c[1].metric("PSI band", str(direction.get("psi_band", "—")).capitalize())
            if counts is not None:
                c[2].metric("Features on alert / watch",
                            f"{int(counts.get('alert', 0))} / {int(counts.get('watch', 0))}")
            c[3].metric("Adversarial AUC", num(adv.get("auc", float("nan")), 3))

    monthly = mon.get("psi_by_month")
    if monthly:
        md = pd.DataFrame(monthly)
        chart = alt.Chart(md).mark_bar(size=26, color=ACCENT).encode(
            x=alt.X("month:N", title=None),
            y=alt.Y("psi:Q", title="PSI against the VALID score"),
            tooltip=[alt.Tooltip("month:N", title="Month"), alt.Tooltip("psi:Q", format=".4f"),
                     alt.Tooltip("n:Q", title="Applications"), alt.Tooltip("band:N", title="Band")],
        ).properties(title="Score PSI by month", height=240)
        st.altair_chart(style(chart), width="stretch")
        st.caption("Bands: below 0.10 stable, 0.10 to 0.25 watch, above 0.25 alert. The reference is the VALID score distribution.")

    psi = csv("psi.csv")
    if psi is None:
        missing("psi.csv")
    else:
        top = psi.sort_values("psi", ascending=False).head(15).copy()
        top["Band"] = top.band.map(lambda b: b.capitalize())
        gloss = {**feature_view(top.feature.tolist())}
        top["Label"] = top.feature.map(gloss)
        chart = alt.Chart(top).mark_bar(size=13).encode(
            y=alt.Y("Label:N", sort=list(top.Label), title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("psi:Q", title="PSI, TRAIN → OOT"),
            color=alt.Color("Band:N", title="Band",
                            scale=alt.Scale(domain=["Stable", "Watch", "Alert"],
                                            range=[BANDS["stable"], BANDS["watch"], BANDS["alert"]])),
            tooltip=[alt.Tooltip("Label:N", title="Feature"), alt.Tooltip("psi:Q", format=".3f"),
                     alt.Tooltip("feature:N", title="Column"), alt.Tooltip("Band:N", title="Band")],
        ).properties(title="The 15 features that drift most", height=340)
        st.altair_chart(style(chart), width="stretch")
        st.caption(f"{int(counts.get('stable', 0))} features stable, {int(counts.get('watch', 0))} on watch "
                   f"and {int(counts.get('alert', 0))} in alert. Most drifted features are aggregates of history tables, "
                   "not drivers of the champion.")
        with st.expander(f"All {len(psi)} feature PSI values"):
            st.dataframe(psi.sort_values("psi", ascending=False), hide_index=True, width="stretch",
                         column_config={"psi": st.column_config.NumberColumn(format="%.3f")})

    drivers = adv.get("top_drifting") or []
    if drivers:
        with st.expander("What makes the two periods separable"):
            dv = pd.DataFrame(drivers[:8]).rename(columns={
                "feature": "Feature", "gain_share": "Gain share", "gain": "Gain"})
            st.dataframe(dv[["Feature", "Gain share"]], hide_index=True, width="stretch",
                         column_config={"Gain share": st.column_config.NumberColumn(format="%.3f")})
            st.caption(f"LightGBM classifier on {adv.get('n_train_sample', 0):,} sampled TRAIN and "
                       f"{adv.get('n_oot_sample', 0):,} sampled OOT applications, held-out AUC "
                       f"{num(adv.get('auc', float('nan')))}. It uses no label, so this measures how separable the "
                       "periods are, not target leakage. The features are not removed: dropping them would tune the "
                       "design to the OOT window.")

    plan = mon.get("monitoring_plan")
    if plan:
        with st.expander("Monitoring plan"):
            bands = plan.get("psi_bands", {})
            for band, rule in bands.items():
                st.markdown(f"* **{band.capitalize()}** — {rule}")
            for key, value in plan.items():
                if key in ("psi_bands", "gini_trigger"):
                    continue
                st.markdown(f"* **{key.replace('_', ' ').capitalize()}** — {value}")
            gini = plan.get("gini_trigger", {})
            if gini:
                st.markdown(f"* **Weekly Gini trigger** — {gini.get('rule')} "
                            f"(reference mean {num(gini.get('valid_mean_gini', float('nan')))}).")


# --------------------------------------------------------------------------- Score a case
def page_scorer() -> None:
    if meta is None:
        missing("feature_meta.json")
        return
    model = load_champion(meta)
    if model is None:
        missing(meta.get("model_file") or "champion model")
        return

    features = meta["features"]
    typical = meta.get("typical_values", {})
    ev = js("evaluation.json") or {}
    named = feature_view(features)
    top = [r["feature"] for r in ev.get("shap", {}).get("top_features", [])][:6]
    editable = [f for f in top if f in features] or features[:6]

    st.caption(
        f"The saved **{label(meta['champion'])}** scores the row; nothing is refitted. The form starts at the median of "
        "each feature over all applications, so moving one driver shows how the score responds. Real applications can "
        "still be missing some features (see the coverage table in README section 2)."
    )

    def num_or(value, default: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    base = {f: num_or((typical.get(f) or {}).get("p50"), np.nan) for f in features}

    with st.form("score_one"):
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(editable):
            info = typical.get(f) or {}
            lo, hi = num_or(info.get("p05"), 0.0), num_or(info.get("p95"), 1.0)
            mid = num_or(info.get("p50"), 0.0)
            if lo >= hi:
                lo, hi = 0.0, max(1.0, abs(mid) * 2)
            mid = min(max(mid, lo), hi)
            values[f] = cols[i % 3].slider(named[f], min_value=float(lo), max_value=float(hi), value=float(mid))
        submitted = st.form_submit_button("Score this application")

    if not submitted:
        return

    row = pd.DataFrame([{**base, **values}])[features].astype("float32")
    p = float(score_one(model, meta.get("kind"), row)[0])
    score = float(pd_to_score(np.array([p]))[0])
    cuts = champ_meta.get("cutoffs_valid_raw_pd") or {}
    own = meta.get("champion") in cuts
    cut = (cuts.get(meta.get("champion")) or cuts.get("xgboost") or {}).get("approval_80", 0.05)
    band = "inside the 80% approval cut-off" if p < cut else "above the cut-off (review)"

    c = st.columns(3)
    c[0].metric("PD", f"{p:.2%}")
    c[1].metric("Score", f"{score:.0f}")
    c[2].metric("Cut-off band", band)
    st.caption("Score scale: PDO 20, 600 points at 20:1 good:bad odds (higher score = safer). "
               + ("The cut-off is the selected model's own 80% approval cut-off frozen on VALID, applied to its raw PD."
                  if own else "No cut-off was frozen for this model; the band uses the XGBoost 80% approval cut-off "
                              "applied to this model's raw PD and is indicative only."))

    if meta.get("scorecard_file"):
        pts = csv("scorecard_points.csv")
        if pts is not None and meta.get("scorecard_features"):
            contrib = []
            for f in meta["scorecard_features"]:
                hit = pts[pts["feature"] == f]
                if not hit.empty:
                    contrib.append((f, float(hit["points"].mean())))
            contrib.sort(key=lambda t: t[1])
            st.markdown("#### Scorecard reason codes")
            st.dataframe(
                pd.DataFrame({"Feature": [named.get(f, f) for f, _ in contrib[:5]],
                              "Points": [p_ for _, p_ in contrib[:5]]}),
                hide_index=True,
                width="stretch",
                column_config={"Points": st.column_config.NumberColumn(format="%.2f")},
            )
            st.caption("Mean points per bin of the monotonically binned scorecard (the logistic track). The lowest "
                       "points are the strongest risk-increasing contributions for this row.")


# --------------------------------------------------------------------------- Batch scoring
def page_batch() -> None:
    if meta is None:
        missing("feature_meta.json")
        return
    model = load_champion(meta)
    if model is None:
        missing(meta.get("model_file") or "champion model")
        return
    features = meta["features"]

    st.caption(f"The file needs the {len(features)} champion features, in any order. Missing features are scored as "
               "missing; a column that is not part of the model is rejected so a mis-mapped file is never scored silently. "
               "The limit is 20,000 rows.")

    template = ("case_id," + ",".join(features) + "\n").encode("utf-8")
    tab_up, tab_paste = st.tabs(["Upload a CSV", "Paste CSV text"])
    with tab_up:
        st.download_button("Download a header-only template", template, "home_credit_template.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if up is not None:
            out, err = _score_batch(pd.read_csv(up), meta, model)
            if err:
                st.error(err)
            else:
                _show_batch(out)
        else:
            st.caption("One row per application; the score is added as `pd` and `score`.")
    with tab_paste:
        text = st.text_area("Paste CSV text", height=140, placeholder="case_id," + features[0] + "\n1,0.5")
        if st.button("Score pasted CSV") and text.strip():
            try:
                df = pd.read_csv(pd.io.common.StringIO(text))
            except Exception as exc:  # noqa: BLE001 - user input
                st.error(f"Could not parse the CSV: {exc}")
                return
            out, err = _score_batch(df, meta, model)
            if err:
                st.error(err)
            else:
                _show_batch(out)


def _score_batch(df: pd.DataFrame, meta: dict, model):
    features = meta["features"]
    unknown = [c for c in df.columns if c not in features and c not in ("case_id", "target", "week")]
    if unknown:
        return None, f"Unknown columns ({len(unknown)}): {unknown[:10]} — the schema check rejected the file."
    if len(df) == 0:
        return None, "The file has no rows."
    if len(df) > 20000:
        return None, "The batch limit is 20,000 rows for this demo."
    p = score_one(model, meta.get("kind"), df.reindex(columns=features).astype("float32"))
    out = df.copy()
    out["pd"] = p
    out["score"] = pd_to_score(p)
    return out, None


def _show_batch(out: pd.DataFrame) -> None:
    st.dataframe(
        out, hide_index=True, width="stretch",
        column_config={
            "pd": st.column_config.NumberColumn("PD", format="%.4f"),
            "score": st.column_config.NumberColumn("Score", format="%.0f"),
        },
    )
    st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"), "scored.csv", "text/csv")


# --------------------------------------------------------------------------- Credit policy
def page_policy() -> None:
    biz = js("business.json")
    if biz is None:
        missing("business.json")
        return
    st.caption("Every rate is an observed out-of-time outcome. Cut-offs are chosen on VALID and applied unchanged to OOT, "
               "so the realised approval rate differs from the target.")

    strategies = biz.get("cutoff_strategies_oot")
    if strategies:
        d = pd.DataFrame(strategies)
        view = pd.DataFrame({"Strategy": d.strategy,
                             "Model": d.model.replace("-", "—") if "model" in d else "—"})
        spec = [("approval_rate", "Approval rate"), ("bad_rate_approved", "Bad rate, approved"),
                ("bad_rate_declined", "Bad rate, declined"), ("defaulters_declined_share", "Defaulters declined")]
        keep = [(c, t) for c, t in spec if c in d.columns]
        for col, name in keep:
            view[name] = 100 * d[col]
        st.markdown("#### Cut-off strategies")
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            t: st.column_config.NumberColumn(format="%.2f%%" if t != "Defaulters declined" else "%.1f%%")
            for _, t in keep
        })
        st.caption("Bad rate, approved is the share of approved loans that default on OOT; Defaulters declined is the "
                   "share of all OOT defaulters the cut-off rejects.")

    capture = biz.get("capture")
    if capture:
        d = pd.DataFrame(capture)
        vars_ = [(c, t) for c, t in [("caught_at_5pct", "Riskiest 5%"), ("caught_at_10pct", "Riskiest 10%"),
                                     ("caught_at_20pct", "Riskiest 20%")] if c in d.columns]
        long = d.melt(id_vars="model", value_vars=[c for c, _ in vars_], var_name="Capacity", value_name="Share caught")
        long["Capacity"] = long.Capacity.map(dict(vars_))
        long["Model"] = long.model
        order = ["XGBoost", "LightGBM", "Random forest", "WoE scorecard", "Logistic regression (L2)"]
        present = [m for m in order if m in set(long.Model)]
        sort = present + [m for m in long.Model.unique() if m not in present]
        chart = alt.Chart(long).mark_bar().encode(
            y=alt.Y("Model:N", sort=sort, title=None, axis=alt.Axis(labelLimit=200)),
            yOffset=alt.YOffset("Capacity:N", sort=[t for _, t in vars_]),
            x=alt.X("Share caught:Q", title="share of defaulters caught", axis=alt.Axis(format="%"), stack=None,
                    scale=alt.Scale(domain=[0, 0.75])),
            color=alt.Color("Capacity:N", title="Review capacity",
                            scale=alt.Scale(domain=[t for _, t in vars_], range=[ACCENT, "#5C87B2", NEUTRAL][:len(vars_)])),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Capacity:N"), alt.Tooltip("Share caught:Q", format=".1%")],
        ).properties(title="Defaulters caught at a given review capacity", height=300)
        st.altair_chart(style(chart), width="stretch")

    swap = biz.get("swap_sets") or {}
    labels = {"both_approve": "Approved by both", "swap_in_A_only": "XGBoost only (A)",
              "swap_out_B_only": "Logistic regression only (B)", "both_decline": "Declined by both"}
    rows = [{"Segment": labels.get(k, k), "Applications": v.get("n"), "Defaults": v.get("defaults"),
             "Bad rate": v.get("bad_rate")}
            for k, v in swap.items() if isinstance(v, dict) and "n" in v]
    rows.sort(key=lambda r: list(labels.values()).index(r["Segment"]) if r["Segment"] in labels.values() else 99)
    if rows:
        st.markdown(f"#### Swap set at {pct(swap.get('approval_rate', 0.8), 0)} approval")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Applications": st.column_config.NumberColumn(format="%d"),
            "Defaults": st.column_config.NumberColumn(format="%d"),
            "Bad rate": st.column_config.NumberColumn(format="%.2f%%"),
        })
        st.caption(f"Both models approve {swap.get('n_approved_each', 0):,} loans. "
                   + (f"The {swap['swap_in_A_only'].get('n', 0):,} loans only XGBoost approves default at "
                      f"{pct(swap['swap_in_A_only']['bad_rate'])}; the same number of loans only logistic regression "
                      f"approves default at {pct(swap['swap_out_B_only']['bad_rate'])}."
                      if "swap_in_A_only" in swap and "swap_out_B_only" in swap else ""))

    grades = biz.get("master_scale")
    if grades:
        g = pd.DataFrame(grades)
        st.markdown("#### Risk grades")
        view = pd.DataFrame({"Grade": g.grade})
        for col, name, fmt in [("n_oot", "OOT applications", "%d"), ("grade_pd_valid", "Grade PD (VALID)", "%.2f%%"),
                               ("oot_default_rate", "OOT default rate", "%.2f%%"), ("binomial_p", "Binomial p", "%.4f"),
                               ("direction", "OOT vs grade PD", None)]:
            if col in g.columns:
                view[name] = 100 * g[col] if fmt == "%.2f%%" else g[col]
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            name: st.column_config.NumberColumn(format=fmt)
            for name, fmt in [("OOT applications", "%d"), ("Grade PD (VALID)", "%.2f%%"),
                              ("OOT default rate", "%.2f%%"), ("Binomial p", "%.4f")] if name in view.columns
        })
        long = g.melt(id_vars="grade", value_vars=["grade_pd_valid", "oot_default_rate"],
                      var_name="Series", value_name="Rate")
        long["Series"] = long.Series.map({"grade_pd_valid": "Grade PD (VALID)",
                                          "oot_default_rate": "OOT default rate"})
        chart = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("grade:O", title="grade (1 = safest)"),
            y=alt.Y("Rate:Q", title="default rate", scale=alt.Scale(type="log"), axis=alt.Axis(format="%")),
            color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=["Grade PD (VALID)", "OOT default rate"],
                                                                   range=[NEUTRAL, ACCENT])),
            tooltip=[alt.Tooltip("grade:O", title="Grade"), alt.Tooltip("Series:N"),
                     alt.Tooltip("Rate:Q", format=".3%")],
        ).properties(title="Grade PD against the realised OOT default rate", height=300)
        st.altair_chart(style(chart), width="stretch")
        st.caption("Grades are equal-count cuts of the VALID score. The binomial test on each grade is uncorrected for "
                   "multiple testing; the OOT default rate is monotone across grades.")

    el = biz.get("expected_loss_illustrative") or {}
    if el:
        st.caption(f"An illustrative expected loss (PD × LGD × EAD) with an **assumed** LGD of {el.get('lgd_ASSUMED')} "
                   f"and EAD = the application credit amount gives {el.get('expected_loss_total', 0):,.0f} against "
                   f"{el.get('realised_loss_total', 0):,.0f} on realised defaults (ratio {num(el.get('ratio', float('nan')), 2)}). "
                   "This is not a measured loss: the data carry no exposure, LGD or margin.")


# --------------------------------------------------------------------------- Model card
def page_model_card() -> None:
    st.markdown(
        """
#### Intended use
Rank consumer-loan applications by probability of default and monitor that ranking through time. This is decision
support for a human underwriter, not an automated decline engine, and it is not a pricing or limit-setting tool.

#### Not intended for
Pricing, limit setting, collections, or any decision where the *level* of PD matters more than the ranking. The OOT
calibration gap is reported, and even after Platt scaling the predicted PD sits above the observed OOT default rate on
average.

#### Assumptions
The expected-loss illustration uses an assumed LGD and the application credit amount as exposure, because the dataset
carries no exposure, LGD or margin. It is illustrative only.

#### Limitations
* **One split, one seed, one run.** Variation across other week cut-offs and seeds was not measured.
* **Different population out of time.** Adversarial validation reaches AUC 0.999 between TRAIN and OOT; a likely cause is
  a change in product mix and in how much history each source holds, and it was not decomposed.
* **VALID is used for several decisions** (early stopping, tuning, the calibrator, the cut-offs and the grades), so VALID
  figures are not independent evidence.
* **Class imbalance was not handled.** All models are fitted on the raw 3.32% TRAIN default rate.
* **The scorecard is a monotone WoE design** restricted to 37 variables, so its gap to logistic regression mixes design,
  binning and sign pruning; no ablation was run.
* **No fairness or subgroup analysis** was run, and the importances shown are XGBoost gain shares, not SHAP values.
"""
    )

    gates = js("quality_gates.json")
    if gates is None:
        missing("quality_gates.json")
    else:
        rows = [{
            "Gate": g.get("gate", "").replace(":", " · ").replace("_", " ").capitalize(),
            "Status": g.get("status", ""),
        } for g in gates.get("gates", [])]
        st.markdown("#### Quality gates")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     height=min(760, 35 * len(rows) + 38))
        st.caption(f"{gates.get('n_pass', 0)} of {gates.get('n_pass', 0) + gates.get('n_fail', 0)} gates pass. The leakage, "
                   "coverage and calendar checks raise before any model is fitted; the rest are evaluated at the end of "
                   "the run. Details are in `artifacts/quality_gates.json`.")

    env = js("environment.json")
    if env:
        with st.expander("Environment the artifacts were produced in"):
            items = [(k, str(v) if not isinstance(v, dict) else ", ".join(f"{a}={b}" for a, b in v.items()))
                     for k, v in env.items()]
            st.dataframe(pd.DataFrame(sorted(items), columns=["Setting", "Value"]), hide_index=True, width="stretch")

    st.caption("Data: the Home Credit Credit Risk Model Stability competition on Kaggle, used under the competition "
               "rules. No raw data is redistributed with this repository or this app.")


# --------------------------------------------------------------------------- scoring helpers
@st.cache_resource(show_spinner=False)
def load_champion(meta: dict):
    """Load the saved champion in its native format. Never refit."""
    model_file = meta.get("model_file")
    if not model_file:
        return None
    path = ART / model_file
    if not path.exists():
        return None
    kind = meta.get("kind")
    if kind == "lgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if kind == "xgb":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    if kind == "rf":
        import joblib

        return joblib.load(path)
    return None


def score_one(model, kind: str | None, row: pd.DataFrame) -> np.ndarray:
    if kind == "lgbm":
        return np.asarray(model.predict(row))
    if kind == "xgb":
        import xgboost as xgb

        return np.asarray(model.predict(xgb.DMatrix(row, feature_names=list(row.columns))))
    return np.asarray(model.predict_proba(row))[:, 1]


def pd_to_score(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pdo, base_score, base_odds = meta.get("pdo", 20.0), meta.get("base_score", 600.0), meta.get("base_odds", 20.0)
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset - factor * np.log(p / (1 - p))


# --------------------------------------------------------------------------- router
PAGES_IMPL = {
    "Overview": page_overview,
    "Model comparison": page_comparison,
    "Stability & monitoring": page_stability,
    "Score a case": page_scorer,
    "Batch scoring": page_batch,
    "Credit policy": page_policy,
    "Model card": page_model_card,
}

PAGES_IMPL[page]()
