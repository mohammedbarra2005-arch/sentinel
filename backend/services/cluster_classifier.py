"""
IMPORTANT — logique NOUVELLE, ajoutée côté API, pas présente dans le projet
d'origine :

Le vrai `gat_accounts_nodes.parquet` ne contient qu'une étiquette BINAIRE
(`label` 0/1 = "à risque" ou non) — il ne distingue pas explicitement si un
compte à risque appartient à un ANNEAU (cycle A→B→C→A) ou à un SCHÉMA FAN-IN
(comptes collecteurs), contrairement à ce que l'explorateur de graphe du
frontend affiche.

Cette information existe implicitement dans la STRUCTURE du graphe des
transactions (gat_accounts_transactions.parquet), donc on la retrouve ici par
détection de motifs plutôt que de la deviner :
  - un compte qui participe à un cycle dirigé de 3 à 5 comptes -> "ring"
    (ça correspond exactement à la façon dont generate_synthetic_accounts.py
    construit les anneaux : RING_SIZE_RANGE = (3, 5))
  - les comptes à risque restants (non cycliques) -> "fan_in", regroupés par
    composante connexe ; le compte avec le plus grand nombre d'émetteurs
    distincts dans chaque composante est marqué comme collecteur

C'est nécessaire pour que la page Explorateur de graphe du frontend (qui
attend une distinction ring/fan_in) continue à fonctionner avec les vraies
données. Si tu préfères que le champ soit calculé autrement (ex: stocké
explicitement lors de la génération), c'est modifiable dans
generate_synthetic_accounts.py — mais ça demanderait de régénérer les données.
"""
import functools
from pathlib import Path

import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@functools.lru_cache(maxsize=1)
def get_clusters() -> list[dict]:
    accounts = pd.read_parquet(PROCESSED_DIR / "gat_accounts_nodes.parquet")
    tx = pd.read_parquet(PROCESSED_DIR / "gat_accounts_transactions.parquet")

    risky_ids = set(accounts.loc[accounts["label"] == 1, "account_id"])

    # On ne garde que les arêtes explicitement injectées comme faisant partie
    # d'un pattern frauduleux (is_fraud_ring == 1). Les transactions "normales"
    # (is_fraud_ring == 0) sont tirées au hasard entre TOUS les comptes, donc
    # deux comptes à risque de clusters DIFFÉRENTS peuvent être connectés par
    # une transaction normale par pur hasard — les inclure fusionnait à tort
    # plusieurs anneaux/fan-in distincts en une seule composante connexe.
    G = nx.DiGraph()
    G.add_nodes_from(risky_ids)
    for row in tx.itertuples():
        if row.is_fraud_ring == 1 and row.source in risky_ids and row.target in risky_ids:
            G.add_edge(row.source, row.target)

    ring_ids: set[str] = set()
    for cycle in nx.simple_cycles(G, length_bound=5):
        if 3 <= len(cycle) <= 5:
            ring_ids.update(cycle)

    clusters = []
    ring_sub = G.subgraph(ring_ids)
    for i, comp in enumerate(nx.weakly_connected_components(ring_sub)):
        clusters.append({"id": f"ring-{i}", "type": "ring", "accountIds": sorted(comp)})

    fanin_ids = risky_ids - ring_ids
    fanin_sub = G.subgraph(fanin_ids)

    # Les 5 schémas fan-in générés par generate_synthetic_accounts.py ne sont PAS
    # garantis disjoints (le tirage aléatoire des comptes source/collecteur ne
    # s'exclut pas entre schémas différents) — regrouper par composante connexe
    # les fusionnait donc en un seul gros bloc au lieu de 5 groupes distincts.
    # On identifie plutôt directement les comptes COLLECTEURS (beaucoup de
    # comptes différents leur envoient de l'argent, peu voire aucune sortie) et
    # on construit un cluster par collecteur — un compte "mule" peut apparaître
    # sous plusieurs collecteurs si c'est vraiment le cas dans les données,
    # ce qui reflète mieux la réalité qu'un partitionnement strict.
    collectors = [
        n for n in fanin_sub.nodes
        if fanin_sub.in_degree(n) >= 4 and fanin_sub.in_degree(n) > fanin_sub.out_degree(n)
    ]
    for i, collector in enumerate(collectors):
        senders = sorted(fanin_sub.predecessors(collector))
        clusters.append({
            "id": f"fan-{i}", "type": "fan_in",
            "accountIds": senders + [collector], "collectorId": collector,
        })

    return clusters


@functools.lru_cache(maxsize=1)
def get_account_labels() -> dict[str, str]:
    """Retourne {account_id: 'normal' | 'ring' | 'fan_in'} pour TOUS les comptes."""
    labels: dict[str, str] = {}
    for cluster in get_clusters():
        for acc_id in cluster["accountIds"]:
            labels[acc_id] = cluster["type"]
    return labels


def get_graph_edges() -> list[dict]:
    """Ne retourne que les arêtes marquées is_fraud_ring == 1 (mêmes arêtes que
    celles utilisées pour la classification des clusters ci-dessus). Les
    transactions "normales" (is_fraud_ring == 0) sont tirées au hasard entre
    tous les comptes — les inclure noierait la visualisation sous des liens
    sans rapport avec les schémas de fraude que l'explorateur doit mettre en
    évidence."""
    tx = pd.read_parquet(PROCESSED_DIR / "gat_accounts_transactions.parquet")
    fraud_tx = tx[tx["is_fraud_ring"] == 1]
    return [
        {"source": row.source, "target": row.target, "amount": float(row.amount)}
        for row in fraud_tx.itertuples()
    ]
