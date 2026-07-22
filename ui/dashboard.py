"""Monitoring dashboard: PSI/AUC trends, CSI, deciles, and the fairness
breakdown, all read from Postgres (drift_log)."""

import altair as alt
import pandas as pd
import streamlit as st

from ui import core, services


def _threshold_rule(y: float, label: str, color: str) -> alt.LayerChart:
    """Dashed horizontal rule with a text label. The rule wears the
    status color; the label wears a text token, so meaning never rides
    on color alone and stays readable on the light surface."""
    df = pd.DataFrame({"y": [y], "label": [f"{label} ({y:g})"]})
    rule = alt.Chart(df).mark_rule(strokeDash=[4, 4], strokeWidth=1.5, color=color).encode(y="y:Q")
    text = alt.Chart(df).mark_text(
        align="left", baseline="bottom", dx=4, dy=-2,
        color=core.TEXT_SECONDARY, fontSize=11,
    ).encode(y="y:Q", text="label:N", x=alt.value(0))
    return rule + text


def _trend_chart(df: pd.DataFrame, value_title: str, rules: list) -> alt.LayerChart:
    base = alt.Chart(df).encode(
        x=alt.X("measured_at:T", title=None),
        y=alt.Y("metric_value:Q", title=value_title,
                scale=alt.Scale(zero=False, padding=12)),
        tooltip=[
            alt.Tooltip("measured_at:T", title="Measured", format="%Y-%m-%d %H:%M"),
            alt.Tooltip("metric_value:Q", title=value_title, format=".4f"),
            alt.Tooltip("model_version:N", title="Model"),
        ],
    )
    line = base.mark_line(strokeWidth=2, color=core.SERIES)
    points = base.mark_point(size=70, filled=True, color=core.SERIES)
    return alt.layer(line, points, *rules)


def _history_table(df: pd.DataFrame, value_name: str):
    with st.expander("Table view"):
        st.dataframe(
            df[["measured_at", "metric_value", "model_version"]].rename(
                columns={"metric_value": value_name, "measured_at": "measured at",
                         "model_version": "model"}),
            hide_index=True, width="stretch",
        )


def _psi_section(psi_rows):
    st.subheader("Score drift (PSI)")
    df = core.metric_history_frame(psi_rows)
    if df.empty:
        st.info("No PSI history yet. Run `gbm drift` with DATABASE_URL set.", icon="📭")
        return
    latest = df.iloc[-1]
    st.altair_chart(_trend_chart(df, "PSI", [
        _threshold_rule(core.PSI_WARNING, "warning", core.STATUS["warning"]),
        _threshold_rule(core.PSI_CRITICAL, "critical: retrain", core.STATUS["critical"]),
    ]), width="stretch")
    _history_table(df, "psi")

    csi = core.csi_frame(latest["details"])
    if not csi.empty:
        st.subheader("Feature stability (CSI, latest run)")
        chart = alt.Chart(csi).mark_bar(
            color=core.SERIES, cornerRadiusEnd=4, height={"band": 0.6},
        ).encode(
            x=alt.X("csi:Q", title="CSI"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("csi:Q", format=".4f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0.20]})).mark_rule(
            strokeDash=[4, 4], strokeWidth=1.5, color=core.STATUS["serious"],
        ).encode(x="x:Q")
        st.altair_chart(
            (chart + rule).properties(height=max(240, 18 * len(csi))),
            width="stretch")
        st.caption("Dashed rule: CSI 0.20 (investigate).")


def _performance_section(auc_rows):
    st.subheader("Model performance (AUC)")
    df = core.metric_history_frame(auc_rows)
    if df.empty:
        st.info("No AUC history yet. Run `gbm performance` with DATABASE_URL set.", icon="📭")
        return
    latest = df.iloc[-1]
    details = latest["details"] or {}
    baseline = latest["metric_value"] + float(details.get("auc_drop", 0.0))
    st.altair_chart(_trend_chart(df, "AUC", [
        _threshold_rule(round(baseline, 4), "training baseline", core.TEXT_SECONDARY),
        _threshold_rule(round(baseline - core.AUC_DROP_THRESHOLD, 4),
                        "retrain threshold", core.STATUS["critical"]),
    ]), width="stretch")
    src = details.get("outcomes_source", "unknown")
    st.caption(f"Latest run scored on: {src}. KS {details.get('ks', 'n/a')}, "
               f"rank-order breaks {details.get('rank_order_breaks', 'n/a')}.")
    _history_table(df, "auc")

    deciles = core.decile_frame(details)
    if not deciles.empty:
        st.subheader("Rank ordering (deciles, latest run)")
        chart = alt.Chart(deciles).mark_bar(
            color=core.SERIES, cornerRadiusEnd=4,
        ).encode(
            x=alt.X("decile:O", title="Score decile (1 = lowest risk)"),
            y=alt.Y("default_rate:Q", title="Observed default rate", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("decile:O", title="Decile"),
                alt.Tooltip("default_rate:Q", title="Default rate", format=".2%"),
                alt.Tooltip("avg_score:Q", title="Avg score", format=".4f"),
                alt.Tooltip("count:Q", title="N", format=","),
            ],
        ).properties(height=260)
        st.altair_chart(chart, width="stretch")
        with st.expander("Table view"):
            st.dataframe(deciles, hide_index=True, width="stretch")


def _fairness_section(fairness_rows):
    st.subheader("Fairness (DIR / EOD / SPD)")
    if not fairness_rows:
        st.info("No fairness rows yet. They are published by `gbm drift` "
                "from the champion's metadata (one row per model version).", icon="📭")
        return
    measured_at, min_dir, version, details = fairness_rows[-1]
    thr = float(details.get("dir_threshold", 0.8))
    df = core.fairness_frame(details)
    if df.empty:
        st.info("Fairness row has no attribute breakdown.", icon="📭")
        return

    violations = df[df["violation"]]
    if violations.empty:
        st.success(f"Model {version}: no groups below the {thr:g} four-fifths rule.", icon="✅")
    else:
        groups = ", ".join(f"{r.attribute} / {r.group}" for r in violations.itertuples())
        st.warning(
            f"Model {version}: DIR below {thr:g} for: {groups}. "
            "Structural in the public LendingClub data; the promotion gate "
            "is champion-relative (see README).", icon="⚠️")

    bars = alt.Chart(df).mark_bar(
        color=core.SERIES, cornerRadiusEnd=4, height={"band": 0.6},
    ).encode(
        x=alt.X("dir:Q", title="Disparate Impact Ratio (1.0 = parity)"),
        y=alt.Y("group:N", title=None, sort=None),
        tooltip=[
            alt.Tooltip("attribute:N"), alt.Tooltip("group:N"),
            alt.Tooltip("dir:Q", title="DIR", format=".3f"),
            alt.Tooltip("approval_rate:Q", title="Approval rate", format=".1%"),
            alt.Tooltip("default_rate:Q", title="Default rate", format=".1%"),
        ],
    )
    rule = alt.Chart(df).mark_rule(
        strokeDash=[4, 4], strokeWidth=1.5, color=core.STATUS["critical"],
    ).encode(x=alt.datum(thr))
    # Layer first, then facet: faceted charts cannot be layered. Each
    # attribute gets its own y scale so facets only list their own groups.
    chart = alt.layer(bars, rule).properties(height={"step": 26}).facet(
        row=alt.Row("attribute:N", title=None,
                    header=alt.Header(labelFontWeight="bold", labelAngle=0,
                                      labelOrient="top", labelAnchor="start")),
    ).resolve_scale(y="independent")
    st.altair_chart(chart, width="stretch")
    st.caption(f"Dashed rule: four-fifths threshold ({thr:g}). "
               "Privileged group per attribute has DIR 1.0 by definition.")

    st.markdown("**Full breakdown** (equal opportunity and statistical parity "
                "differences are vs the privileged group)")
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "attribute": "Attribute", "group": "Group",
            "privileged": st.column_config.CheckboxColumn("Privileged"),
            "dir": st.column_config.NumberColumn("DIR", format="%.3f"),
            "eod": st.column_config.NumberColumn("EOD", format="%.3f"),
            "spd": st.column_config.NumberColumn("SPD", format="%.3f"),
            "approval_rate": st.column_config.NumberColumn("Approval", format="%.1%"),
            "default_rate": st.column_config.NumberColumn("Default", format="%.1%"),
            "violation": st.column_config.CheckboxColumn("< 4/5 rule"),
        },
    )


def page():
    st.title("Monitoring & fairness")
    st.caption("Everything on this page is read from Postgres (drift_log), "
               "as written by the Go monitors.")

    if not services.db_available():
        st.error("Set DATABASE_URL to use the dashboard.", icon="🗄️")
        return
    try:
        psi_rows = services.metric_history("psi")
        auc_rows = services.metric_history("auc")
        fairness_rows = services.metric_history("fairness")
    except Exception as e:
        st.error(f"Could not read drift_log: {e}", icon="🗄️")
        return

    # Headline tiles from the latest rows
    c1, c2, c3 = st.columns(3)
    if psi_rows:
        psi = psi_rows[-1][1]
        c1.metric("Latest PSI", f"{psi:.4f}", core.psi_status(psi),
                  delta_color="off")
    if auc_rows:
        c2.metric("Latest AUC", f"{auc_rows[-1][1]:.4f}")
    if fairness_rows:
        c3.metric("Worst-group DIR", f"{fairness_rows[-1][1]:.3f}",
                  help="Lowest Disparate Impact Ratio across all groups")

    _psi_section(psi_rows)
    st.divider()
    _performance_section(auc_rows)
    st.divider()
    _fairness_section(fairness_rows)
