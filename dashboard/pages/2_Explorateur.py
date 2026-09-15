import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
from components.data_loader import load_analytics_clean

st.set_page_config(page_title="Explorateur", page_icon="🔍", layout="wide")
apply_theme()
st.title("🔍 Explorateur de données")

df = load_analytics_clean()

with st.sidebar:
    st.header("Filtres")
    pays = st.multiselect("Pays émetteur", sorted(df["DbtrCountry"].unique()))
    devise = st.multiselect("Devise", sorted(df["Currency"].unique()))
    statut = st.multiselect("Statut", sorted(df["TxSts"].unique()))
    montant_min, montant_max = st.slider(
        "Plage de montant", float(df["InstdAmt"].min()), float(df["InstdAmt"].max()),
        (float(df["InstdAmt"].min()), float(df["InstdAmt"].max())),
    )
    recherche = st.text_input("Recherche (MessageId / EndToEndId)")

filtered = df.copy()
if pays:
    filtered = filtered[filtered["DbtrCountry"].isin(pays)]
if devise:
    filtered = filtered[filtered["Currency"].isin(devise)]
if statut:
    filtered = filtered[filtered["TxSts"].isin(statut)]
filtered = filtered[(filtered["InstdAmt"] >= montant_min) & (filtered["InstdAmt"] <= montant_max)]
if recherche:
    filtered = filtered[
        filtered["MessageId"].str.contains(recherche, case=False)
        | filtered["EndToEndId"].str.contains(recherche, case=False)
    ]

st.write(f"**{len(filtered)}** transactions correspondent aux filtres.")
page_size = 50
n_pages = max(1, (len(filtered) - 1) // page_size + 1)
page = st.number_input("Page", min_value=1, max_value=n_pages, value=1)
st.dataframe(filtered.iloc[(page - 1) * page_size: page * page_size], use_container_width=True)

st.download_button(
    "📥 Exporter en CSV (résultat filtré)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="transactions_filtrees.csv",
    mime="text/csv",
)
