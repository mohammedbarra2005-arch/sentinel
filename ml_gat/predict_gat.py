"""
ml_gat/predict_gat.py
----------------------
Charge le modèle GAT entraîné et prédit le niveau de risque d'un compte (par son account_id).
Par défaut, utilise le graphe ENRICHI (400 comptes individuels avec vrais patterns de fraude,
cf. generate_synthetic_accounts.py), pas l'ancienne version à 14 noeuds BIC.

Contrairement à ml/predict.py (XGBoost) qui score UNE TRANSACTION isolée, le GAT score
UN COMPTE dans le contexte de tout le graphe -- il faut donc reconstruire le graphe complet
à chaque prédiction (le modèle a besoin de voir les voisins du compte pour "faire attention"
à eux). C'est une différence clé à expliquer si on te pose la question en entretien/soutenance.
"""
import torch
import logging
from pathlib import Path

from gat_model import SimpleGAT
from graph_builder import build_graph, build_graph_from_accounts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict_gat")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

_model = None
_data = None


def _load():
    """Charge le modèle et reconstruit le graphe une seule fois (mis en cache en mémoire)."""
    global _model, _data
    if _model is None:
        checkpoint = torch.load(MODELS_DIR / "gat_model.pt", weights_only=False)
        _model = SimpleGAT(in_channels=checkpoint["in_channels"])
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        _data = build_graph_from_accounts()
    return _model, _data


def predict_account_risk(account_id: str):
    """Retourne le niveau de risque prédit pour un compte donné (identifié par son account_id).

    Renvoie : (label, probabilite) où label est 0 (faible risque) ou 1 (à risque),
    et probabilite est la confiance du modèle pour cette classe.
    """
    model, data = _load()

    if account_id not in data.bic_list:
        raise ValueError(f"Compte inconnu dans le graphe : {account_id}")

    node_idx = data.bic_list.index(account_id)

    with torch.no_grad():
        out = model(data.x, data.edge_index)          # log-probabilités pour TOUS les noeuds
        probs = torch.exp(out)                          # reconversion en probabilités [0,1]
        node_probs = probs[node_idx]
        label = int(node_probs.argmax().item())
        confidence = float(node_probs[label].item())

    return label, confidence


def predict_all_accounts():
    """Retourne les scores de risque pour TOUS les comptes du graphe (utile pour le dashboard)."""
    model, data = _load()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)
        labels = probs.argmax(dim=1)

    results = []
    for i, account_id in enumerate(data.bic_list):
        results.append({
            "account_id": account_id,
            "label": int(labels[i].item()),
            "risk_level": "à risque" if labels[i].item() == 1 else "faible risque",
            "confidence": float(probs[i][labels[i]].item()),
        })
    return results


if __name__ == "__main__":
    results = predict_all_accounts()
    print(f"{'Compte':<15} {'Niveau':<15} {'Confiance':<10}")
    print("-" * 40)
    for r in sorted(results, key=lambda r: -r["confidence"])[:20]:
        print(f"{r['account_id']:<15} {r['risk_level']:<15} {r['confidence']:.2%}")
