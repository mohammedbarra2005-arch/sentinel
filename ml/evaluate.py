"""
Module 4 (partie 3) — Évaluation et explicabilité (SHAP)
"""
import json
import pickle
import logging
import pandas as pd
import numpy as np
import shap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def run_shap():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    test_df = pd.read_parquet(PROCESSED_DIR / "test_scored.parquet")
    X_test = test_df[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(sorted(
        zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: -x[1]
    ))

    with open(MODELS_DIR / "shap_importance.json", "w") as f:
        json.dump(shap_importance, f, indent=2)

    # Sauvegarde des valeurs SHAP par transaction du test set (pour explication individuelle)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=test_df.index)
    shap_df["MessageId"] = test_df["MessageId"].values
    shap_df.to_parquet(PROCESSED_DIR / "shap_values.parquet", index=False)

    logger.info("Top 5 features (SHAP): %s", list(shap_importance.items())[:5])
    return shap_importance


if __name__ == "__main__":
    run_shap()
