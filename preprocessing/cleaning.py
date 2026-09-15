"""
Module 1 — Import et préparation des données
Charge, valide, nettoie et profile les 3 fichiers CSV ISO 20022.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleaning")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ANALYTICS_COLS = [
    "MessageId", "EndToEndId", "TxDateTime", "StatusTimestamp", "ProcessingTimeSecs",
    "IntrBkSttlmDt", "InstdAmt", "Currency", "DbtrAgtBIC", "CdtrAgtBIC",
    "DbtrCountry", "CdtrCountry", "PurposeCode", "TxSts", "RejectionReason",
]


def load_raw():
    logger.info("Chargement des fichiers CSV bruts depuis %s", RAW_DIR)
    analytics = pd.read_csv(RAW_DIR / "iso20022_analytics.csv")
    pacs002 = pd.read_csv(RAW_DIR / "pacs002.csv")
    pacs008 = pd.read_csv(RAW_DIR / "pacs008.csv")
    logger.info("Lignes en entrée -> analytics: %d, pacs002: %d, pacs008: %d",
                len(analytics), len(pacs002), len(pacs008))
    return analytics, pacs002, pacs008


def validate_columns(df: pd.DataFrame, expected: list, name: str):
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {name}: {missing}")
    logger.info("Validation des colonnes OK pour %s", name)


def clean_analytics(df: pd.DataFrame) -> pd.DataFrame:
    n_in = len(df)
    df = df.copy()

    # RejectionReason est nul par construction pour les transactions acceptées
    # -> encodé comme catégorie "NONE" (aucun rejet), pas comme donnée manquante à imputer
    df["RejectionReason"] = df["RejectionReason"].fillna("NONE")

    # Doublons sur identifiants métier
    dup_msg = df["MessageId"].duplicated().sum()
    dup_e2e = df["EndToEndId"].duplicated().sum()
    if dup_msg or dup_e2e:
        logger.warning("Doublons détectés -> MessageId: %d, EndToEndId: %d", dup_msg, dup_e2e)
    df = df.drop_duplicates(subset=["MessageId"])

    # Typage des dates
    df["TxDateTime"] = pd.to_datetime(df["TxDateTime"])
    df["StatusTimestamp"] = pd.to_datetime(df["StatusTimestamp"])
    df["IntrBkSttlmDt"] = pd.to_datetime(df["IntrBkSttlmDt"])

    # Cohérence des montants
    df = df[df["InstdAmt"] > 0]

    n_out = len(df)
    logger.info("clean_analytics: %d -> %d lignes (%d supprimées)", n_in, n_out, n_in - n_out)
    return df


def profile_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "cardinality": {c: int(df[c].nunique()) for c in df.columns},
        "taux_rejet": float((df["TxSts"] == "RJCT").mean()),
        "rejection_reason_counts": df["RejectionReason"].value_counts().to_dict(),
        "devises": df["Currency"].value_counts().to_dict(),
        "pays_emetteur": df["DbtrCountry"].value_counts().to_dict(),
        "pays_receveur": df["CdtrCountry"].value_counts().to_dict(),
    }
    return report


def run_pipeline():
    analytics, pacs002, pacs008 = load_raw()
    validate_columns(analytics, EXPECTED_ANALYTICS_COLS, "iso20022_analytics.csv")

    analytics_clean = clean_analytics(analytics)
    report = profile_report(analytics_clean)

    logger.info("Taux de rejet global: %.2f%%", report["taux_rejet"] * 100)

    out_path = PROCESSED_DIR / "analytics_clean.parquet"
    analytics_clean.to_parquet(out_path, index=False)
    logger.info("Données nettoyées sauvegardées -> %s (%d lignes)", out_path, len(analytics_clean))

    return analytics_clean, report


if __name__ == "__main__":
    df, report = run_pipeline()
    print("\n=== RAPPORT DE PROFILING ===")
    for k, v in report.items():
        print(f"\n{k}:")
        print(v)
