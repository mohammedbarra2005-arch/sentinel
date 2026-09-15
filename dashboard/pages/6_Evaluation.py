import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from components.theme import apply_theme
import pandas as pd
import plotly.express as px
from components.data_loader import load_metrics, load_shap_importance

st.set_page_config(page_title="Évaluation du modèle", page_icon="📈", layout="wide")
apply_theme()
st.title("📈 Évaluation du modèle XGBoost")

st.warning(
    "⚠️ Cible de substitution : le modèle prédit le statut de rejet (`TxSts = RJCT`), "
    "en l'absence d'étiquette de fraude confirmée. Voir section 2 du cahier des charges."
)

data = load_metrics()
metrics = data["metrics"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
c2.metric("Précision", f"{metrics['precision']:.3f}")
c3.metric("Rappel", f"{metrics['recall']:.3f}")
c4.metric("F1-score", f"{metrics['f1']:.3f}")
c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Courbe ROC")
    roc = data["roc_curve"]
    fig = px.line(x=roc["fpr"], y=roc["tpr"], labels={"x": "Taux de faux positifs", "y": "Taux de vrais positifs"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Matrice de confusion")
    cm = data["confusion_matrix"]
    st.plotly_chart(
        px.imshow(cm, text_auto=True, x=["Prédit: Accepté", "Prédit: Rejeté"],
                  y=["Réel: Accepté", "Réel: Rejeté"]),
        use_container_width=True,
    )

st.divider()
col3, col4 = st.columns(2)
with col3:
    st.subheader("Importance des variables (gain XGBoost)")
    imp = pd.DataFrame(list(data["feature_importances"].items())[:15], columns=["variable", "importance"])
    st.plotly_chart(px.bar(imp, x="importance", y="variable", orientation="h"), use_container_width=True)

with col4:
    shap_imp = load_shap_importance()
    st.subheader("Importance des variables (SHAP, |valeur moyenne|)")
    if shap_imp:
        shap_df = pd.DataFrame(list(shap_imp.items())[:15], columns=["variable", "shap_importance"])
        st.plotly_chart(px.bar(shap_df, x="shap_importance", y="variable", orientation="h"), use_container_width=True)
    else:
        st.info("Lancez `python ml/evaluate.py` pour générer les valeurs SHAP.")
