"""
ml_gat/train_gat.py
--------------------
Entraîne le modèle GAT sur le graphe construit par graph_builder.py.

Contrairement à XGBoost (train/test split classique sur des LIGNES de données), ici
on fait un split sur les NOEUDS du graphe : certains comptes servent à l'entraînement,
d'autres sont mis de côté pour le test -- mais le graphe entier (toutes les arêtes)
reste visible au modèle pendant l'entraînement, seul le label de certains noeuds est
caché. C'est la façon standard de faire du "node classification" avec un GNN.

NOTE HONNÊTE (à garder pour ta documentation) : avec seulement 14 noeuds, un split
train/test n'a presque aucun sens statistique -- ce script sert de démonstration de
l'architecture et du pipeline d'entraînement, pas d'une évaluation robuste. Avec des
données réelles (des milliers de comptes clients individuels plutôt que 14 BIC), ce
même code serait directement utilisable.
"""
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from gat_model import SimpleGAT
from graph_builder import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gat")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


def make_train_test_masks(num_nodes: int, test_ratio: float = 0.3):
    """Crée un masque booléen indiquant quels noeuds servent à l'entraînement vs au test."""
    perm = torch.randperm(num_nodes)
    n_test = max(1, int(num_nodes * test_ratio))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def train(epochs: int = 100, lr: float = 0.01, weight_decay: float = 5e-4):
    data = build_graph()
    train_mask, test_mask = make_train_test_masks(data.num_nodes)

    logger.info("Graphe : %d noeuds, %d arêtes, %d features/noeud",
                data.num_nodes, data.num_edges, data.num_node_features)
    logger.info("Split : %d noeuds train, %d noeuds test", train_mask.sum().item(), test_mask.sum().item())

    model = SimpleGAT(in_channels=data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info("Epoch %3d | loss = %.4f", epoch, loss.item())

    # Évaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        y_true = data.y[test_mask].numpy()
        y_pred = pred[test_mask].numpy()

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    logger.info("Métriques test (GAT) : %s", metrics)

    # Sauvegarde du modèle entraîné
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": data.num_node_features,
        "bic_list": data.bic_list,
    }, MODELS_DIR / "gat_model.pt")
    logger.info("Modèle GAT sauvegardé -> %s", MODELS_DIR / "gat_model.pt")

    return model, data, metrics


if __name__ == "__main__":
    train()
