import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from components.theme import apply_theme
from datetime import datetime
from components.data_loader import load_analytics_clean
from ml.predict import score_transaction

st.set_page_config(page_title="Prédiction", page_icon="🎯", layout="wide")
apply_theme()
st.title("🎯 Prédiction de risque — transaction unitaire")

df = load_analytics_clean()

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Montant", min_value=0.01, value=1000.0)
        currency = st.selectbox("Devise", sorted(df["Currency"].unique()))
        purpose_code = st.selectbox("Code d'objet (PurposeCode)", sorted(df["PurposeCode"].unique()))
    with c2:
        dbtr_country = st.selectbox("Pays émetteur", sorted(df["DbtrCountry"].unique()))
        cdtr_country = st.selectbox("Pays receveur", sorted(df["CdtrCountry"].unique()))
        processing_time = st.number_input("Temps de traitement (s)", min_value=0, value=5)
    with c3:
        dbtr_bic = st.selectbox("BIC émetteur", sorted(df["DbtrAgtBIC"].unique()))
        cdtr_bic = st.selectbox("BIC receveur", sorted(df["CdtrAgtBIC"].unique()))
        tx_time = st.time_input("Heure de la transaction", value=datetime.now().time())
        tx_day = st.selectbox("Jour de la semaine", list(range(7)),
                               format_func=lambda d: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][d])

    submitted = st.form_submit_button("Calculer le score de risque")

if submitted:
    proba, level = score_transaction(
        amount=amount, currency=currency, purpose_code=purpose_code,
        dbtr_country=dbtr_country, cdtr_country=cdtr_country,
        dbtr_bic=dbtr_bic, cdtr_bic=cdtr_bic,
        processing_time_secs=processing_time, hour=tx_time.hour, dow=tx_day,
    )
    color = {"faible": "green", "moyen": "orange", "élevé": "red"}[level]
    st.markdown(f"### Score de risque : **{proba:.1%}** — niveau :green[{level}]" if level == "faible"
                else f"### Score de risque : **{proba:.1%}** — niveau :{color}[{level}]")
    st.progress(min(proba, 1.0))
    st.caption("Score = probabilité de rejet/anomalie estimée par le modèle (cible de substitution, cf. note méthodologique).")
