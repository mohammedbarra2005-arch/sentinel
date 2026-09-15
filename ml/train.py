"""
Module 4 (partie 2) — Entraînement du modèle XGBoost
Cible: TxSts == RJCT (proxy de risque/anomalie, cf. cahier des charges section 2 & 10.2)
"""
import pandas as pd
import numpy as np
import json
import logging
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


def load_features():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "feature_cols.txt") as f:
        feature_cols = f.read().splitlines()
    return df, feature_cols


def train():
    df, feature_cols = load_features()
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    logger.info("Split train/test: %d / %d (taux de positifs train=%.3f, test=%.3f)",
                len(X_train), len(X_test), y_train.mean(), y_test.mean())

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("scale_pos_weight = %.2f", scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    logger.info("Metriques test: %s", metrics)
    logger.info("Matrice de confusion: %s", cm)

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "confusion_matrix": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "feature_importances": importances,
        }, f, indent=2)

    # Sauvegarde du test set pour la page d'évaluation / transactions suspectes
    test_out = X_test.copy()
    test_out["target"] = y_test.values
    test_out["risk_score"] = y_proba
    test_out["MessageId"] = df.loc[X_test.index, "MessageId"].values
    test_out.to_parquet(PROCESSED_DIR / "test_scored.parquet", index=False)

    # Scoring de l'ensemble du dataset (pour la page transactions suspectes / dashboard)
    full_proba = model.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["risk_score"] = full_proba
    scored["risk_level"] = pd.cut(
        scored["risk_score"], bins=[-0.01, 0.33, 0.66, 1.0], labels=["faible", "moyen", "élevé"]
    )
    scored.to_parquet(PROCESSED_DIR / "scored_transactions.parquet", index=False)

    logger.info("Modèle, métriques et scores sauvegardés dans %s", MODELS_DIR)
    return model, metrics


if __name__ == "__main__":
    train()
