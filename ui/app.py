"""
Credit Risk UI: a Streamlit front end over the Go scoring API and the
Postgres monitoring tables. Two pages: interactive scoring (REST) and
the monitoring/fairness dashboard (drift_log).

Run locally:
    API_URL=http://localhost:8000 API_KEY=demo-local-key \
    DATABASE_URL=postgresql://credit_risk:credit_risk@localhost:5432/credit_risk \
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui import dashboard, scoring  # noqa: E402

st.set_page_config(
    page_title="Credit Risk GBM",
    page_icon="📊",
    layout="wide",
)

pages = st.navigation([
    st.Page(scoring.page, title="Score an applicant", icon="🧮",
            url_path="score", default=True),
    st.Page(dashboard.page, title="Monitoring & fairness", icon="📈",
            url_path="monitoring"),
])
pages.run()
