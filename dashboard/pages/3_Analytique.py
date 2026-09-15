import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
import plotly.express as px
from components.data_loader import load_analytics_clean

st.set_page_config(page_title="Analytique", page_icon="📊", layout="wide")
apply_theme()
st.title("📊 Analytique descriptive")

df = load_analytics_clean()

st.subheader("Évolution temporelle des paiements")
daily = df.set_index("TxDateTime").resample("D").size().reset_index(name="nb_transactions")
st.plotly_chart(px.line(daily, x="TxDateTime", y="nb_transactions"), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribution des montants")
    st.plotly_chart(px.histogram(df, x="InstdAmt", nbins=50), use_container_width=True)
with c2:
    st.subheader("Top banques émettrices")
    top_banks = df["DbtrAgtBIC"].value_counts().head(10).reset_index()
    top_banks.columns = ["BIC", "nb_transactions"]
    st.plotly_chart(px.bar(top_banks, x="BIC", y="nb_transactions"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Usage des devises")
    st.plotly_chart(px.pie(df, names="Currency"), use_container_width=True)
with c4:
    st.subheader("Heatmap horaire (heure x jour de la semaine)")
    tmp = df.copy()
    tmp["heure"] = tmp["TxDateTime"].dt.hour
    tmp["jour"] = tmp["TxDateTime"].dt.day_name()
    pivot = tmp.pivot_table(index="jour", columns="heure", values="MessageId", aggfunc="count", fill_value=0)
    st.plotly_chart(px.imshow(pivot, aspect="auto"), use_container_width=True)

st.subheader("Taux de rejet par motif")
rejected = df[df["TxSts"] == "RJCT"]["RejectionReason"].value_counts().reset_index()
rejected.columns = ["motif_rejet", "nb"]
st.plotly_chart(px.bar(rejected, x="motif_rejet", y="nb"), use_container_width=True)
