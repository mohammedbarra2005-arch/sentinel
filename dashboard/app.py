import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from components.theme import apply_theme

st.set_page_config(page_title="Détection de Fraude ISO 20022", page_icon="🔎", layout="wide")
apply_theme()

st.title("🔎 Plateforme de Détection de Fraude — ISO 20022")
st.markdown("""
Bienvenue. Utilisez le menu à gauche (**Pages**) pour naviguer :

1. **Accueil** — KPIs globaux
2. **Explorateur** — recherche et filtres sur les transactions
3. **Analytique** — visualisations Plotly
4. **Prédiction** — score de risque en temps réel
5. **Graphe** — exploration Neo4j / réseau de comptes
6. **Évaluation** — performance et explicabilité du modèle
7. **Transactions suspectes** — triées par score de risque

⚠️ **Note méthodologique** : en l'absence d'étiquette de fraude confirmée dans les données,
le modèle est entraîné sur une cible de substitution (statut de rejet `TxSts = RJCT`). Les scores
affichés sont donc des **probabilités de risque/anomalie**, pas des probabilités de fraude avérée.
""")
