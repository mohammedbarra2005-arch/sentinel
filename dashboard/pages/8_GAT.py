import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml_gat"))

import streamlit as st
import pandas as pd
from components.theme import apply_theme

st.set_page_config(page_title="GAT - Graphe Neuronal", page_icon="🧠", layout="wide")
apply_theme()
st.title("🧠 GAT — Graph Attention Network")

st.info(
    "Ce modèle est un **second modèle expérimental**, en complément du XGBoost principal "
    "(page Évaluation). Contrairement à XGBoost qui évalue chaque transaction isolément, "
    "le GAT analyse chaque **compte** dans le contexte de tout le réseau de transactions "
    "(ses voisins dans le graphe), via un mécanisme d'attention.\n\n"
    "Ce modèle est entraîné sur un jeu de données **synthétique enrichi** (400 comptes "
    "individuels) avec de vrais patterns de fraude en réseau injectés (cycles de transferts, "
    "comptes collecteurs) — contrairement à la version initiale (14 comptes/banques), "
    "les étiquettes ici sont **vraies** (pas un proxy), ce qui permet au GAT de démontrer sa "
    "capacité à détecter des patterns structurels invisibles pour XGBoost."
)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "gat_model.pt"

if not MODEL_PATH.exists():
    st.warning(
        "⚠️ Le modèle GAT n'a pas encore été entraîné.\n\n"
        "Lance ces commandes dans ton terminal (depuis la racine du projet) :\n"
        "```\npython ml_gat/graph_builder.py\npython ml_gat/train_gat.py\n```\n"
        "Puis reviens rafraîchir cette page."
    )
    st.stop()

try:
    from predict_gat import predict_all_accounts
    results = predict_all_accounts()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle GAT : {e}")
    st.stop()

df = pd.DataFrame(results).sort_values("confidence", ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Comptes analysés", len(df))
col2.metric("Comptes 'à risque'", int((df["label"] == 1).sum()))
col3.metric("Confiance moyenne", f"{df['confidence'].mean():.1%}")

st.divider()
st.subheader("Score de risque par compte")
st.dataframe(
    df.rename(columns={"account_id": "Compte", "risk_level": "Niveau", "confidence": "Confiance"})
      [["Compte", "Niveau", "Confiance"]]
      .style.background_gradient(subset=["Confiance"], cmap="Reds"),
    use_container_width=True,
    height=400,
)

st.divider()
with st.expander("ℹ️ Note méthodologique"):
    st.markdown("""
    Ce graphe contient **400 comptes individuels** (contre 14 dans la version initiale
    basée uniquement sur les codes BIC des banques). Les patterns de fraude — cycles de
    transferts A→B→C→A et comptes collecteurs (fan-in) — ont été **injectés
    délibérément** dans des données synthétiques, ce qui donne de vraies étiquettes
    (pas un proxy comme pour XGBoost). Le F1-score obtenu (~0.29) montre que le GAT
    apprend un signal réel, mais reste modeste comparé à un modèle simple sur les mêmes
    features (~0.6 en régression logistique) — piste d'amélioration : plus de données,
    plus de couches, ou plus d'épochs d'entraînement.
    """)
