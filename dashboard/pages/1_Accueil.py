import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
from components.data_loader import load_analytics_clean, load_scored_transactions

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")
apply_theme()
st.title("🏠 Accueil — Indicateurs clés")

df = load_analytics_clean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions totales", f"{len(df):,}")
col2.metric("Montant total", f"{df['InstdAmt'].sum():,.0f}")
col3.metric("Taux de rejet", f"{(df['TxSts'] == 'RJCT').mean() * 100:.1f} %")

try:
    scored = load_scored_transactions()
    col4.metric("Score de risque moyen", f"{scored['risk_score'].mean():.3f}")
except FileNotFoundError:
    col4.metric("Score de risque moyen", "modèle non entraîné")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Répartition par devise")
    st.bar_chart(df["Currency"].value_counts())
with c2:
    st.subheader("Répartition par pays émetteur")
    st.bar_chart(df["DbtrCountry"].value_counts())
