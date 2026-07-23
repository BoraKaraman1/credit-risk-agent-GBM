"""Visual theme for the UI: one CSS injection shared by all pages.

Pure presentation, no data logic. Charts keep their explicit colors from
ui/core.py; this styles the Streamlit chrome around them (metric tiles as
cards, softer surface, consistent radii and typography).
"""

import streamlit as st

_CSS = """
<style>
/* Surface: soft neutral background so white cards read as cards. */
[data-testid="stAppViewContainer"] { background: #f7f8fa; }
[data-testid="stHeader"] { background: rgba(247, 248, 250, 0.85); }

/* Typography. */
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"] { color: #6b6a67; }

/* Metric tiles as cards. */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}
[data-testid="stMetricLabel"] { color: #52514e; }

/* Alerts, dataframes, expanders: consistent radius + hairline border. */
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"],
[data-testid="stExpander"] {
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons and inputs. */
button[kind="primary"] { border-radius: 10px; font-weight: 600; }
button[kind="secondary"] { border-radius: 10px; }
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] input { border-radius: 10px; }

/* Sidebar: clean white rail. */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6e8eb;
}

/* Tabs: slightly stronger labels. */
[data-testid="stTabs"] button { font-weight: 600; }

/* Dividers: quieter. */
hr { border-color: #e6e8eb; }
</style>
"""


def inject() -> None:
    """Apply the shared CSS. Call once per run, before pages render."""
    st.markdown(_CSS, unsafe_allow_html=True)
