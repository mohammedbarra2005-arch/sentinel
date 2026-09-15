"""
ml_gat/graph_builder.py
------------------------
Construit le graphe utilisé par le GAT (Graph Attention Network).

Ce fichier NE remplace PAS neo4j/import_graph.py : il lit les MÊMES données nettoyées
(data/processed/analytics_clean.parquet — la source utilisée pour peupler Neo4j) et
construit un objet PyTorch Geometric `Data` équivalent, prêt à être utilisé par un
modèle de deep learning sur graphe.

Schéma du graphe (simplifié par rapport à Neo4j pour le GAT) :
  - Noeud   = un compte (identifié par son BIC), comme dans Neo4j (label :Compte)
  - Arête   = une transaction entre deux comptes (DbtrAgtBIC -> CdtrAgtBIC).
              Dans Neo4j on a Compte-[:ENVOIE]->Transaction-[:RECUE_PAR]->Compte ;
              ici on simplifie en une arête directe compte->compte, car GATConv
              travaille sur des arêtes entre noeuds, pas sur des noeuds-transactions.
  - Feature de noeud = statistiques agrégées par compte (montant total envoyé/reçu,
              nombre de transactions, taux de rejet historique...)
  - Label de noeud   = 1 si le compte a un taux de rejet élevé (proxy de risque),
              0 sinon. Même logique de "cible de substitution" honnête que pour XGBoost
              (cf. cahier des charges section 2) : on n'a pas de vraie étiquette de fraude.

Pourquoi construire depuis les données et pas directement depuis une requête Neo4j
en dur ? Pour que ce script marche même si Neo4j n'est pas démarré (comme le fallback
NetworkX du dashboard). Si tu veux lire directement depuis Neo4j au lieu du parquet,
la fonction load_from_neo4j() plus bas montre comment faire.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Seuil au-delà duquel un compte est considéré "à risque" (proxy label, pas une vraie fraude)
RISK_LABEL_THRESHOLD = 0.10  # 10% de taux de rejet sur les transactions envoyées


def load_transactions() -> pd.DataFrame:
    """Charge les données nettoyées (mêmes données que celles importées dans Neo4j)."""
    return pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construit une ligne par compte (BIC), avec des features agrégées et un label proxy.

    Un compte peut apparaître comme émetteur (DbtrAgtBIC) et/ou receveur (CdtrAgtBIC) ;
    on fusionne les deux rôles dans une seule table de noeuds.
    """
    sent = df.groupby("DbtrAgtBIC").agg(
        montant_envoye_total=("InstdAmt", "sum"),
        nb_transactions_envoyees=("MessageId", "count"),
        taux_rejet_envoi=("TxSts", lambda s: (s == "RJCT").mean()),
    ).reset_index().rename(columns={"DbtrAgtBIC": "bic"})

    received = df.groupby("CdtrAgtBIC").agg(
        montant_recu_total=("InstdAmt", "sum"),
        nb_transactions_recues=("MessageId", "count"),
    ).reset_index().rename(columns={"CdtrAgtBIC": "bic"})

    nodes = pd.merge(sent, received, on="bic", how="outer").fillna(0.0)

    # Label proxy : compte "à risque" si son taux de rejet en tant qu'émetteur dépasse le seuil.
    # NOTE méthodologique : comme pour XGBoost, ceci est un proxy (pas une fraude confirmée).
    #
    # NOTE IMPORTANTE sur ce jeu de données : les comptes ici sont au niveau BIC (14 comptes
    # au total, un par banque), pas au niveau client individuel. Avec un seuil fixe de 10%,
    # AUCUN compte ne dépasse ce seuil (le taux de rejet moyen se dilue sur 5000 transactions
    # réparties sur seulement 14 comptes) -> toutes les étiquettes seraient à 0, ce qui rendrait
    # l'entraînement inutile (le modèle prédirait "sûr" pour tout le monde et "gagnerait" 100%
    # trivialement). On utilise donc un seuil ADAPTATIF (médiane) pour garantir un minimum de
    # signal à apprendre. C'est une limite à documenter : avec seulement 14 noeuds, un GAT n'a
    # pas assez de données pour apprendre un vrai pattern de graphe -> ce pipeline sert avant
    # tout de démonstration d'architecture, pas d'un modèle de production (cf. même honnêteté
    # méthodologique que pour le modèle XGBoost, section 2 du cahier des charges).
    median_reject = nodes["taux_rejet_envoi"].median()
    nodes["label"] = (nodes["taux_rejet_envoi"] > median_reject).astype(int)

    return nodes.reset_index(drop=True)


def build_edge_index(df: pd.DataFrame, bic_to_idx: dict) -> torch.Tensor:
    """Construit la liste d'arêtes (compte_emetteur -> compte_receveur), au format
    attendu par PyTorch Geometric : un tenseur [2, nb_arêtes]."""
    src = df["DbtrAgtBIC"].map(bic_to_idx).values
    dst = df["CdtrAgtBIC"].map(bic_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    return edge_index


def build_graph() -> Data:
    """Point d'entrée principal : construit l'objet Data (graphe complet) pour le GAT."""
    df = load_transactions()
    nodes = build_node_table(df)

    bic_to_idx = {bic: i for i, bic in enumerate(nodes["bic"])}

    feature_cols = [
        "montant_envoye_total", "nb_transactions_envoyees", "taux_rejet_envoi",
        "montant_recu_total", "nb_transactions_recues",
    ]
    # Normalisation simple (min-max) pour aider l'entraînement du GAT
    x = nodes[feature_cols].values.astype(np.float32)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-8)
    x = torch.tensor(x, dtype=torch.float)

    y = torch.tensor(nodes["label"].values, dtype=torch.long)
    edge_index = build_edge_index(df, bic_to_idx)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.bic_list = nodes["bic"].tolist()  # pour retrouver le BIC depuis l'index du noeud

    return data


def load_from_neo4j():
    """Variante optionnelle : construit le graphe directement depuis une requête Neo4j
    plutôt que depuis le parquet local. Nécessite une instance Neo4j démarrée
    (voir docker-compose.yml). Utile si tu veux que le graphe reflète l'état actuel
    de la base plutôt que le dernier export nettoyé."""
    import os
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "frauddetection")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    query = """
    MATCH (e:Compte)-[:ENVOIE]->(t:Transaction)-[:RECUE_PAR]->(r:Compte)
    RETURN e.bic AS emetteur, r.bic AS receveur, t.montant AS montant, t.statut AS statut
    """
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query)]
    driver.close()

    df = pd.DataFrame(rows).rename(columns={
        "emetteur": "DbtrAgtBIC", "receveur": "CdtrAgtBIC",
        "montant": "InstdAmt", "statut": "TxSts",
    })
    df["MessageId"] = range(len(df))  # id factice pour le comptage
    nodes = build_node_table(df)
    bic_to_idx = {bic: i for i, bic in enumerate(nodes["bic"])}
    feature_cols = [
        "montant_envoye_total", "nb_transactions_envoyees", "taux_rejet_envoi",
        "montant_recu_total", "nb_transactions_recues",
    ]
    x = nodes[feature_cols].values.astype(np.float32)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-8)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(nodes["label"].values, dtype=torch.long)
    edge_index = build_edge_index(df, bic_to_idx)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.bic_list = nodes["bic"].tolist()
    return data


if __name__ == "__main__":
    data = build_graph()
    print(data)
    print(f"Nombre de noeuds (comptes) : {data.num_nodes}")
    print(f"Nombre d'arêtes (transactions) : {data.num_edges}")
    print(f"Nombre de features par noeud : {data.num_node_features}")
    print(f"Répartition des labels : {torch.bincount(data.y).tolist()} (0=faible risque, 1=à risque)")

    torch.save(data, PROCESSED_DIR / "gat_graph.pt")
    print(f"\nGraphe sauvegardé -> {PROCESSED_DIR / 'gat_graph.pt'}")
