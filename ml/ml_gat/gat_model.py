"""
ml_gat/gat_model.py
--------------------
Définit le modèle GAT (Graph Attention Network) utilisé pour classifier chaque compte
(noeud du graphe) comme "à risque" ou "faible risque".

Comment fonctionne un GAT, en une phrase :
Contrairement à XGBoost qui regarde chaque transaction isolément, un GAT regarde aussi
LE VOISINAGE de chaque compte dans le graphe (les comptes avec qui il a transigé) et
apprend à pondérer automatiquement l'importance de chaque voisin ("attention") pour
décider si un compte est à risque.

Architecture : 2 couches GATConv (Graph Attention Convolution), comme demandé dans la spec.
  - Couche 1 : agrège les features des voisins directs, avec plusieurs "têtes d'attention"
               (multi-head attention, comme dans un Transformer) pour capturer différents
               types de relations.
  - Activation ELU + Dropout (évite le sur-apprentissage, important vu le petit nombre de noeuds).
  - Couche 2 : réduit vers 2 sorties (à risque / faible risque), une seule tête d'attention.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SimpleGAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 8, out_channels: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        """
        in_channels  : nombre de features par noeud (5 dans notre cas, cf. graph_builder.py)
        hidden_channels : taille des embeddings intermédiaires
        out_channels : nombre de classes à prédire (2 : faible risque / à risque)
        heads        : nombre de têtes d'attention de la première couche
        dropout      : probabilité de dropout (régularisation)
        """
        super().__init__()
        self.dropout = dropout

        # Première couche : plusieurs têtes d'attention, concaténées (sortie = hidden_channels * heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

        # Deuxième couche : une seule tête, produit directement les scores de classe
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # x          : [nb_noeuds, in_channels]      -> features de chaque compte
        # edge_index : [2, nb_arêtes]                  -> connexions entre comptes

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # activation non-linéaire (courante avec GAT dans le papier original)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # On retourne des log-probabilités (utilisées avec NLLLoss dans train_gat.py)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # Petit test de sanité : vérifie que le modèle tourne sur un graphe factice
    model = SimpleGAT(in_channels=5)
    x = torch.rand((14, 5))
    edge_index = torch.randint(0, 14, (2, 50))
    out = model(x, edge_index)
    print("Sortie du modèle :", out.shape, "(attendu : [14, 2])")
