import streamlit as st
import plotly.express as px

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ---- Page background gradient ---- */
.stApp {
    background: radial-gradient(circle at 15% 0%, #141926 0%, #0E1117 55%);
}

/* ---- Titles ---- */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00D9C0, #7B61FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.2rem;
}
h2, h3 {
    color: #E6E8EC !important;
    font-weight: 600 !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #161A23, #1B202C);
    border: 1px solid #262B38;
    border-radius: 14px;
    padding: 18px 16px 12px 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #00D9C0;
}
[data-testid="stMetricLabel"] {
    color: #9AA3B2 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: #00D9C0 !important;
    font-weight: 700 !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10131B 0%, #0B0D12 100%);
    border-right: 1px solid #1F2430;
}
[data-testid="stSidebar"] * {
    color: #C7CCD8 !important;
}

/* ---- Dataframes / tables ---- */
[data-testid="stDataFrame"], .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #262B38 !important;
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #00D9C0, #00B3E6);
    color: #06121A;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #262B38;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9AA3B2;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #00D9C0 !important;
    border-bottom: 2px solid #00D9C0 !important;
}

/* ---- Alerts / callouts ---- */
[data-testid="stAlert"] {
    border-radius: 12px;
    border-left: 4px solid #00D9C0;
}

/* ---- Progress bar (risk score) ---- */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00D9C0, #FF5C7A);
}

/* ---- Dividers ---- */
hr {
    border-color: #1F2430 !important;
}

/* ---- Captions ---- */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #6E7686 !important;
}
</style>
"""


def apply_theme():
    """Injecte le CSS custom pour un thème sombre et professionnel, et configure
    les graphiques Plotly pour qu'ils suivent le même thème.
    A appeler en haut de chaque page, juste après st.set_page_config()."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = ["#00D9C0", "#7B61FF", "#FF5C7A", "#FFC65C", "#00B3E6", "#8FE9DA"]
    px.defaults.width = None
    px.defaults.height = 420
