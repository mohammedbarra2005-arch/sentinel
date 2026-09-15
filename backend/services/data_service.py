"""
Charge les transactions (analytics_clean.parquet) et leur score de risque
(scored_transactions.parquet), en fusionnant les deux sur MessageId — le score
est calculé sur les features one-hot (features.parquet) qui ne contiennent plus
les colonnes lisibles (Currency, BIC...), donc on le rejoint sur les données
propres pour avoir une réponse API complète et lisible.

Rien ici ne réentraîne ou ne modifie le modèle : lecture seule des fichiers déjà
générés par preprocessing/ + ml/train.py.
"""
import functools
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@functools.lru_cache(maxsize=1)
def _load_merged() -> pd.DataFrame:
    clean = pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")
    try:
        scored = pd.read_parquet(PROCESSED_DIR / "scored_transactions.parquet")
        merged = clean.merge(scored[["MessageId", "risk_score"]], on="MessageId", how="left")
    except FileNotFoundError:
        merged = clean.copy()
        merged["risk_score"] = None
    return merged


def get_transactions(limit: int = 500) -> list[dict]:
    df = _load_merged().head(limit)
    out = []
    for row in df.itertuples():
        out.append({
            "messageId": row.MessageId,
            "debtorBic": row.DbtrAgtBIC,
            "creditorBic": row.CdtrAgtBIC,
            "debtorCountry": row.DbtrCountry,
            "creditorCountry": row.CdtrCountry,
            "currency": row.Currency,
            "purposeCode": row.PurposeCode,
            "amount": float(row.InstdAmt),
            "status": row.TxSts,
            "riskScore": float(row.risk_score) if row.risk_score is not None else 0.0,
            "processingTimeSecs": int(row.ProcessingTimeSecs),
            "timestamp": row.TxDateTime.isoformat(),
        })
    return out


def get_dashboard_stats(high_risk_threshold: float = 0.5) -> dict:
    df = _load_merged()
    reject_rate = (df["TxSts"] == "RJCT").mean()
    avg_risk = df["risk_score"].mean() if df["risk_score"].notna().any() else 0.0

    from .gat_service import get_all_accounts  # import local pour éviter un cycle au chargement
    accounts = get_all_accounts()
    high_risk = sum(1 for a in accounts if a["riskScore"] >= high_risk_threshold)

    from .cluster_classifier import get_clusters
    clusters = get_clusters()
    active_rings = sum(1 for c in clusters if c["type"] == "ring")

    return {
        "totalTransactions": len(df),
        "rejectRate": float(reject_rate),
        "highRiskAccounts": high_risk,
        "activeRings": active_rings,
        "avgRiskScore": float(avg_risk),
    }
