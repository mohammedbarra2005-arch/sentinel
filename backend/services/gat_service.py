"""
Wrapper autour de ml_gat/predict_gat.py. Même pattern sys.path que le dashboard
Streamlit (dashboard/pages/8_GAT.py) : le dossier ml_gat/ est ajouté directement
à sys.path (pas importé comme un package), car ses fichiers font des imports
relatifs entre eux (`from gat_model import SimpleGAT`).
"""
import sys
import functools
from pathlib import Path

import torch
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
GAT_DIR = BASE_DIR / "ml_gat"

if str(GAT_DIR) not in sys.path:
    sys.path.insert(0, str(GAT_DIR))

import predict_gat  # noqa: E402


@functools.lru_cache(maxsize=1)
def _risk_scores_by_account() -> dict[str, float]:
    """Score de risque = P(classe 'à risque') pour CHAQUE compte, pas seulement
    la confiance de la classe prédite (predict_all_accounts() renvoie la confiance
    de la classe gagnante, ce qui sous-estime le risque des comptes classés
    'faible risque' avec une confiance élevée). On réutilise le modèle et le
    graphe déjà mis en cache par predict_gat._load() plutôt que de dupliquer
    la logique de chargement."""
    model, data = predict_gat._load()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)  # [n_noeuds, 2] -> colonne 1 = P(à risque)
    return {account_id: float(probs[i][1]) for i, account_id in enumerate(data.bic_list)}


@functools.lru_cache(maxsize=1)
def _accounts_table() -> pd.DataFrame:
    accounts = pd.read_parquet(PROCESSED_DIR / "gat_accounts_nodes.parquet")
    tx = pd.read_parquet(PROCESSED_DIR / "gat_accounts_transactions.parquet")

    sent = tx.groupby("source").agg(
        totalSent=("amount", "sum"), txCountSent=("message_id", "count"),
    ).reset_index().rename(columns={"source": "account_id"})
    received = tx.groupby("target").agg(
        totalReceived=("amount", "sum"), txCountReceived=("message_id", "count"),
    ).reset_index().rename(columns={"target": "account_id"})

    merged = accounts.merge(sent, on="account_id", how="left").merge(received, on="account_id", how="left")
    return merged.fillna(0.0)


def get_all_accounts() -> list[dict]:
    from .cluster_classifier import get_account_labels  # évite un import circulaire au chargement du module

    df = _accounts_table()
    risk_scores = _risk_scores_by_account()
    labels = get_account_labels()  # {account_id: 'normal' | 'ring' | 'fan_in'}

    out = []
    for row in df.itertuples():
        out.append({
            "accountId": row.account_id,
            "label": labels.get(row.account_id, "normal"),
            "riskScore": risk_scores.get(row.account_id, 0.0),
            "totalSent": float(row.totalSent),
            "totalReceived": float(row.totalReceived),
            "txCountSent": int(row.txCountSent),
            "txCountReceived": int(row.txCountReceived),
        })
    return out


def get_account(account_id: str) -> dict | None:
    for a in get_all_accounts():
        if a["accountId"] == account_id:
            return a
    return None


def predict_account(account_id: str) -> dict:
    """Appelle directement predict_gat.predict_account_risk() — lève ValueError
    si le compte n'existe pas dans le graphe (voir docstring de la fonction
    d'origine : le GAT a besoin du contexte du graphe entier, on ne peut pas
    scorer un compte hypothétique qui n'y est pas)."""
    label, confidence = predict_gat.predict_account_risk(account_id)
    risk_scores = _risk_scores_by_account()
    return {
        "accountId": account_id,
        "label": label,
        "riskLevel": "à risque" if label == 1 else "faible risque",
        "confidence": confidence,
        "riskScore": risk_scores.get(account_id, confidence if label == 1 else 1 - confidence),
    }
