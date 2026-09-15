import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
import pandas as pd
from components.data_loader import load_analytics_clean, use_neo4j_or_fallback

st.set_page_config(page_title="Graphe", page_icon="🕸️", layout="wide")
apply_theme()
st.title("🕸️ Visualisation du graphe (Neo4j)")

engine, engine_name = use_neo4j_or_fallback()
st.caption(f"Moteur de graphe actif : **{engine_name}**")

df = load_analytics_clean()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top émetteurs/receveurs", "Transferts circulaires",
    "Bénéficiaires partagés", "Transactions rejetées", "Voisinage d'une entité"
])

with tab1:
    c1, c2 = st.columns(2)
    c1.subheader("Top émetteurs")
    c1.dataframe(pd.DataFrame(engine.top_emetteurs(10)), use_container_width=True)
    c2.subheader("Top receveurs")
    c2.dataframe(pd.DataFrame(engine.top_receveurs(10)), use_container_width=True)

with tab2:
    st.subheader("Transferts circulaires détectés (A → B → C → A)")
    cycles = engine.transferts_circulaires(25)
    if cycles:
        st.dataframe(pd.DataFrame(cycles), use_container_width=True)
    else:
        st.info("Aucun cycle A→B→C→A détecté sur ce jeu de données.")

with tab3:
    st.subheader("Comptes receveurs partagés par plusieurs émetteurs distincts")
    st.dataframe(pd.DataFrame(engine.beneficiaires_partages(25)), use_container_width=True)

with tab4:
    st.subheader("Sous-graphe des transactions rejetées")
    st.dataframe(pd.DataFrame(engine.sous_graphe_transactions_rejetees(100)), use_container_width=True)

with tab5:
    bic = st.selectbox("Choisir une entité (BIC)", sorted(df["DbtrAgtBIC"].unique()))
    if bic:
        voisins = engine.voisinage(bic, 50)
        st.dataframe(pd.DataFrame(voisins), use_container_width=True)
