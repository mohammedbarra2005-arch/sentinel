"""
Wrapper autour de ml/predict.py (scoring) et des fichiers déjà générés par
ml/train.py + ml/evaluate.py (data/models/metrics.json, shap_importance.json).
Aucune logique de ml/ n'est réécrite ici — juste importée et exposée en JSON.
"""
import json
import sys
import functools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.predict import score_transaction  # noqa: E402


def predict_transaction(payload: dict) -> dict:
    proba, level = score_transaction(
        amount=payload["amount"],
        currency=payload["currency"],
        purpose_code=payload["purposeCode"],
        dbtr_country=payload["debtorCountry"],
        cdtr_country=payload["creditorCountry"],
        dbtr_bic=payload["debtorBic"],
        cdtr_bic=payload["creditorBic"],
        processing_time_secs=payload["processingTimeSecs"],
        hour=payload["hour"],
        dow=payload["dayOfWeek"],
    )
    return {"score": proba, "level": level}


@functools.lru_cache(maxsize=1)
def get_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json") as f:
        metrics_file = json.load(f)

    shap_path = MODELS_DIR / "shap_importance.json"
    shap_features = []
    if shap_path.exists():
        with open(shap_path) as f:
            shap_data = json.load(f)
        shap_features = [
            {"feature": k, "importance": v} for k, v in list(shap_data.items())[:8]
        ]

    m = metrics_file["metrics"]
    return {
        "rocAuc": m["roc_auc"],
        "precision": m["precision"],
        "recall": m["recall"],
        "shapFeatures": shap_features,
    }
