"""
Scoring d'une transaction unique (utilisé par la page "Prédiction" du dashboard).
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_model = None
_feature_cols = None
_reference_df = None


def _load():
    global _model, _feature_cols, _reference_df
    if _model is None:
        with open(MODELS_DIR / "model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open(MODELS_DIR / "feature_cols.json") as f:
            _feature_cols = json.load(f)
        _reference_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    return _model, _feature_cols, _reference_df


def risk_level(score: float) -> str:
    if score < 0.33:
        return "faible"
    if score < 0.66:
        return "moyen"
    return "élevé"


def score_transaction(amount: float, currency: str, purpose_code: str,
                       dbtr_country: str, cdtr_country: str, dbtr_bic: str,
                       cdtr_bic: str, processing_time_secs: int, hour: int, dow: int):
    """Construit le vecteur de features pour une transaction saisie manuellement et retourne un score."""
    model, feature_cols, ref = _load()

    row = {c: 0.0 for c in feature_cols}
    row["log_amount"] = np.log1p(amount)
    row["ProcessingTimeSecs"] = processing_time_secs
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    row["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    pair_mask = (ref["DbtrCountry"] == dbtr_country) & (ref["CdtrCountry"] == cdtr_country)
    pair_mean = ref.loc[pair_mask, "InstdAmt"].mean() if pair_mask.any() else ref["InstdAmt"].mean()
    row["amount_dev_from_pair_mean"] = amount - (pair_mean if not np.isnan(pair_mean) else ref["InstdAmt"].mean())

    def freq_lookup(col, val):
        sub = ref[ref[col] == val]
        return len(sub) / len(ref) if len(sub) else 1.0 / len(ref)

    row["DbtrCountry_freq"] = freq_lookup("DbtrCountry", dbtr_country)
    row["CdtrCountry_freq"] = freq_lookup("CdtrCountry", cdtr_country)
    row["DbtrAgtBIC_freq"] = freq_lookup("DbtrAgtBIC", dbtr_bic)
    row["CdtrAgtBIC_freq"] = freq_lookup("CdtrAgtBIC", cdtr_bic)

    bic_hist = ref[ref["DbtrAgtBIC"] == dbtr_bic]
    row["dbtr_bic_reject_rate"] = (bic_hist["TxSts"] == "RJCT").mean() if len(bic_hist) else ref["dbtr_bic_reject_rate"].mean()

    recent = ref[ref["DbtrAgtBIC"] == dbtr_bic]
    row["dbtr_freq_24h"] = float(recent["dbtr_freq_24h"].mean()) if len(recent) else 1.0
    row["dbtr_freq_7d"] = float(recent["dbtr_freq_7d"].mean()) if len(recent) else 1.0

    cur_col = f"cur_{currency}"
    if cur_col in row:
        row[cur_col] = 1.0
    purpose_col = f"purpose_{purpose_code}"
    if purpose_col in row:
        row[purpose_col] = 1.0

    X = pd.DataFrame([row])[feature_cols]
    proba = float(model.predict_proba(X)[0, 1])
    return proba, risk_level(proba)


if __name__ == "__main__":
    p, lvl = score_transaction(
        amount=15000, currency="USD", purpose_code="TAXS",
        dbtr_country="GB", cdtr_country="FR", dbtr_bic="LOYDGB21", cdtr_bic="BNPAFRPP",
        processing_time_secs=5, hour=14, dow=2,
    )
    print(f"Score de risque: {p:.3f} ({lvl})")
