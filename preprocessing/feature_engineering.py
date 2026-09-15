"""
Module 4 (partie 1) — Feature engineering
Construit les variables d'entrée du modèle XGBoost à partir des données nettoyées.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_engineering")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["TxDateTime"].dt.hour
    dow = df["TxDateTime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["InstdAmt"])
    # écart au montant moyen du couple (pays émetteur, pays receveur)
    pair_mean = df.groupby(["DbtrCountry", "CdtrCountry"])["InstdAmt"].transform("mean")
    df["amount_dev_from_pair_mean"] = df["InstdAmt"] - pair_mean
    return df


def add_bank_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fréquence historique de rejet par BIC émetteur (encodage à partir de la cible -> attention à la fuite,
    calculé ici sur l'ensemble du dataset pour la démonstration ; en production on le recalculerait
    uniquement sur le train set avant de l'appliquer au test)."""
    df = df.copy()
    reject_rate_by_bic = df.groupby("DbtrAgtBIC")["TxSts"].apply(lambda s: (s == "RJCT").mean())
    df["dbtr_bic_reject_rate"] = df["DbtrAgtBIC"].map(reject_rate_by_bic)
    return df


def add_account_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre de transactions du même émetteur (proxy: BIC émetteur) sur des fenêtres 24h/7j."""
    df = df.copy()
    df = df.sort_values("TxDateTime")
    freq_24h, freq_7d = [], []
    grouped = df.groupby("DbtrAgtBIC")
    for bic, group in grouped:
        times = group["TxDateTime"].values
        idx = group.index
        for i, t in enumerate(times):
            window_24h = np.sum((times <= t) & (times > t - np.timedelta64(24, "h")))
            window_7d = np.sum((times <= t) & (times > t - np.timedelta64(7, "D")))
            freq_24h.append((idx[i], window_24h))
            freq_7d.append((idx[i], window_7d))
    freq_24h_map = dict(freq_24h)
    freq_7d_map = dict(freq_7d)
    df["dbtr_freq_24h"] = df.index.map(freq_24h_map)
    df["dbtr_freq_7d"] = df.index.map(freq_7d_map)
    return df.sort_index()


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["Currency", "PurposeCode"], prefix=["cur", "purpose"])
    # target/frequency encoding pour pays et banques (cardinalité plus haute)
    for col in ["DbtrCountry", "CdtrCountry", "DbtrAgtBIC", "CdtrAgtBIC"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
    return df


def build_feature_matrix(df: pd.DataFrame):
    df = add_cyclical_time_features(df)
    df = add_amount_features(df)
    df = add_bank_history_features(df)
    df = add_account_frequency_features(df)
    df = encode_categoricals(df)

    df["target"] = (df["TxSts"] == "RJCT").astype(int)

    feature_cols = [
        "log_amount", "amount_dev_from_pair_mean", "ProcessingTimeSecs",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "dbtr_bic_reject_rate", "dbtr_freq_24h", "dbtr_freq_7d",
        "DbtrCountry_freq", "CdtrCountry_freq", "DbtrAgtBIC_freq", "CdtrAgtBIC_freq",
    ] + [c for c in df.columns if c.startswith("cur_") or c.startswith("purpose_")]

    logger.info("Matrice de features construite: %d lignes, %d colonnes", len(df), len(feature_cols))
    return df, feature_cols


def run_pipeline():
    df = pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")
    df, feature_cols = build_feature_matrix(df)
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Features sauvegardées -> %s", out_path)
    with open(PROCESSED_DIR / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    return df, feature_cols


if __name__ == "__main__":
    df, feature_cols = run_pipeline()
    print("Colonnes de features:", feature_cols)
    print(df[feature_cols + ["target"]].describe())
