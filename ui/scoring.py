"""Scoring page: pick an applicant, call POST /score, show the decision."""

import requests
import streamlit as st

from ui import core, services


def _render_health_sidebar():
    with st.sidebar:
        st.subheader("Scoring API")
        try:
            health = services.api_health()
        except requests.RequestException as e:
            st.error(f"API unreachable at {services.api_url()}\n\n{e}", icon="🔌")
            return
        st.markdown(
            f"Status: **{health.get('status', '?')}**  \n"
            f"Model: **{health.get('model_version', '?')}**  \n"
            f"Calibrated: **{'yes' if health.get('calibrated') else 'no'}**"
        )


def _pick_applicant() -> str:
    if services.db_available():
        try:
            ids = services.sample_applicant_ids()
            if ids:
                return st.selectbox(
                    "Applicant", ids,
                    help="Sample of applicants from the feature store; "
                         "type to search.",
                )
        except Exception as e:
            st.warning(f"Could not list applicants from Postgres: {e}", icon="🗄️")
    return st.text_input("Applicant ID", value="LC_0000001")


def page():
    st.title("Score an applicant")
    st.caption(
        "Calls the Go scoring API. The decision is made on the calibrated "
        "probability of default; adverse action reasons are TreeSHAP-based "
        "and carry ECOA Reg B codes."
    )
    _render_health_sidebar()

    applicant_id = _pick_applicant()
    if not st.button("Score", type="primary"):
        return

    with st.spinner("Scoring..."):
        try:
            body, status = services.api_score(applicant_id)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}", icon="🔌")
            return

    if status != 200:
        st.error(f"API returned {status}: {body.get('detail', body)}", icon="🚫")
        return

    pd_value = body.get("pd", body.get("score"))
    decision = body.get("decision", "?")
    pres = core.decision_presentation(decision)

    left, mid, right = st.columns(3)
    left.metric("Probability of default", f"{pd_value:.2%}")
    scaled = body.get("scaled_score")
    mid.metric("Scorecard score", scaled if scaled is not None else "n/a",
               help="Points-to-double-odds: 600 = 30:1 odds, PDO 20")
    right.metric("Raw model score", f"{body.get('score', float('nan')):.4f}",
                 help="Uncalibrated model output (logged and monitored)")

    banner = {"good": st.success, "warning": st.warning, "critical": st.error}[pres["status"]]
    banner(f"Decision: {pres['label']}", icon=pres["icon"])

    actions = core.adverse_actions_frame(body.get("adverse_actions"))
    st.subheader("Adverse action reasons")
    if actions.empty:
        st.caption(
            "None returned. ECOA requires reasons for decline and manual "
            "review; approvals carry no adverse action."
        )
    else:
        st.dataframe(
            actions,
            hide_index=True,
            width="stretch",
            column_config={
                "code": st.column_config.NumberColumn("Code", help="Reg B reason code"),
                "reason": "Principal reason",
                "feature": "Model feature",
                "shap": st.column_config.NumberColumn("SHAP", format="%.4f"),
                "value": st.column_config.NumberColumn("Applicant value", format="%.2f"),
            },
        )

    with st.expander("Raw API response"):
        st.json(body)
