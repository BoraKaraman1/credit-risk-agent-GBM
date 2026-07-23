"""Model governance page: the generated model card, the champion's
calibration summary, and live scoring-API status, as three tabs."""

import altair as alt
import pandas as pd
import requests
import streamlit as st

from ui import core, services


def _status_banner(pres: dict, prefix: str = ""):
    banner = {"good": st.success, "warning": st.warning, "critical": st.error}[pres["kind"]]
    banner(f"{prefix}{pres['icon']} **{pres['status']}**")


def _model_card_tab(metadata, card):
    if metadata is None and card is None:
        st.info("No champion artifacts found. Train a champion "
                "(pipeline/train.py) and mount data/models + docs into the UI.", icon="📭")
        return

    if metadata:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Champion", metadata.get("version", "?"))
        trained = str(metadata.get("trained_at", "?"))[:10]
        c2.metric("Trained", trained)
        c3.metric("Features", metadata.get("n_features", "?"))
        sc = metadata.get("scorecard") or {}
        c4.metric("Scorecard base", sc.get("base_score", "n/a"),
                  help=f"PDO {sc.get('pdo', '?')}, base odds {sc.get('base_odds', '?')}:1")

        test_metrics = (metadata.get("metrics") or {}).get("test") or {}
        st.caption(
            f"Test discrimination: AUC {test_metrics.get('auc', 'n/a')}, "
            f"KS {test_metrics.get('ks', 'n/a')}, Gini {test_metrics.get('gini', 'n/a')}.")

        frame = core.metrics_frame(metadata)
        if not frame.empty:
            st.dataframe(
                frame, hide_index=True, width="stretch",
                column_config={
                    "split": "Split",
                    "auc": st.column_config.NumberColumn("AUC", format="%.4f"),
                    "ks": st.column_config.NumberColumn("KS", format="%.4f"),
                    "gini": st.column_config.NumberColumn("Gini", format="%.4f"),
                },
            )

    if card:
        pres = core.validation_status_from_card(card)
        _status_banner(pres, prefix="Governance verdict (model card): ")
        st.markdown(card)
    elif metadata:
        st.caption("docs/model_card.md not mounted; showing metadata only.")


def _calibration_tab(metadata):
    if metadata is None:
        st.info("No champion metadata on disk.", icon="📭")
        return
    summary = core.calibration_summary(metadata)
    if summary["method"] is None:
        st.info("Champion metadata has no calibration block.", icon="📭")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method", summary["method"])
    c2.metric("Calibration rows", f"{summary['n_calibration_rows']:,}"
              if summary["n_calibration_rows"] else "n/a")
    c3.metric("Breakpoints", summary["n_breakpoints"] or "n/a")
    if summary["brier_gain"] is not None:
        c4.metric("Brier (calibrated)", f"{summary['brier_calibrated']:.6f}",
                  delta=f"{-summary['brier_gain']:.2e}", delta_color="normal",
                  help=f"Raw Brier {summary['brier_raw']:.6f}. Isotonic calibration "
                       "mostly re-ranks already-monotone scores, so gains are small.")
    else:
        c4.metric("Brier (calibrated)", "n/a")

    df = core.reliability_frame((metadata.get("calibration") or {}))
    if df.empty:
        return
    st.subheader("Reliability curve (test set)")
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], strokeWidth=1.5, color=core.TEXT_SECONDARY,
    ).encode(x="x:Q", y="y:Q")
    lines = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted PD (quantile bin)",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("observed:Q", title="Observed default rate", axis=alt.Axis(format=".0%")),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["raw", "calibrated"],
                                        range=[core.STATUS["serious"], core.SERIES])),
        tooltip=[
            alt.Tooltip("series:N"),
            alt.Tooltip("mean_predicted:Q", title="Mean predicted", format=".3f"),
            alt.Tooltip("observed:Q", title="Observed", format=".3f"),
            alt.Tooltip("n:Q", title="N", format=","),
        ],
    )
    st.altair_chart((diag + lines).properties(height=320), width="stretch")
    st.caption("Dashed diagonal: perfect calibration. Points below the line "
               "under-predict default risk; above it, over-predict.")


def _api_status_tab():
    try:
        health, status_code = services.api_health_full()
    except requests.RequestException as e:
        st.error(f"Scoring API unreachable at {services.api_url()}\n\n{e}", icon="🔌")
        return

    pres = core.health_presentation(health)
    _status_banner(pres, prefix=f"GET /health → HTTP {status_code}: ")

    c1, c2, c3 = st.columns(3)
    c1.metric("Serving model", pres["model_version"])
    c2.metric("Calibrated", "yes" if pres["calibrated"] else "no")
    c3.metric("Feature store", pres["database"],
              help="Postgres applicant_features, as reported by the API")

    db_state = "configured" if services.db_available() else "not configured"
    st.caption(f"UI-side DATABASE_URL: {db_state} (monitoring tables). "
               f"API endpoint: {services.api_url()}")

    with st.expander("Raw /health response"):
        st.json(health)


def page():
    st.title("Model governance")
    st.caption("The published champion under one roof: the auto-generated "
               "model card, its calibration evidence, and the live state of "
               "the scoring API that serves it.")

    metadata = services.champion_metadata()
    card = services.model_card_markdown()

    tab_card, tab_cal, tab_api = st.tabs(["Model card", "Calibration", "API status"])
    with tab_card:
        _model_card_tab(metadata, card)
    with tab_cal:
        _calibration_tab(metadata)
    with tab_api:
        _api_status_tab()
