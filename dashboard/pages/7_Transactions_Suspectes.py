import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
from components.data_loader import load_scored_transactions

st.set_page_config(page_title="Transactions suspectes", page_icon="🚨", layout="wide")
apply_theme()
st.title("🚨 Transactions les plus à risque")

try:
    df = load_scored_transactions()
except FileNotFoundError:
    st.error("Le modèle n'a pas encore été entraîné. Lancez `python ml/train.py`.")
    st.stop()

top_n = st.slider("Nombre de transactions à afficher", 10, 500, 50)
cols = ["MessageId", "InstdAmt", "Currency", "DbtrAgtBIC", "CdtrAgtBIC",
        "DbtrCountry", "CdtrCountry", "TxSts", "risk_score", "risk_level"]
suspects = df.sort_values("risk_score", ascending=False).head(top_n)[cols]

st.dataframe(
    suspects.style.background_gradient(subset=["risk_score"], cmap="Reds"),
    use_container_width=True,
)

st.download_button(
    "📥 Exporter les transactions suspectes",
    suspects.to_csv(index=False).encode("utf-8"),
    file_name="transactions_suspectes.csv",
    mime="text/csv",
)

st.caption(
    "Astuce : croisez ces MessageId avec la page Graphe pour repérer les transferts circulaires "
    "ou les bénéficiaires partagés associés à ces comptes."
)
