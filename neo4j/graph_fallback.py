"""
Moteur de graphe de secours basé sur NetworkX.
Reproduit les mêmes analyses que queries.py directement depuis les données nettoyées,
pour permettre au dashboard de fonctionner même sans instance Neo4j démarrée
(utile pour une démo rapide / portfolio sans dépendance externe).
Le schéma de noeuds/relations reste identique (Compte -ENVOIE-> Transaction -RECUE_PAR-> Compte,
Compte -RATTACHE_A-> Banque -LOCALISEE_EN-> Pays) afin que la logique soit transposable telle quelle
vers Neo4j (voir import_graph.py et queries.py).
"""
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@lru_cache(maxsize=1)
def load_data():
    return pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")


@lru_cache(maxsize=1)
def build_account_graph():
    """Graphe dirigé compte->compte, une arête par transaction (agrégée par montant/count)."""
    df = load_data()
    G = nx.MultiDiGraph()
    for row in df.itertuples():
        G.add_node(row.DbtrAgtBIC, type="Compte", pays=row.DbtrCountry)
        G.add_node(row.CdtrAgtBIC, type="Compte", pays=row.CdtrCountry)
        G.add_edge(
            row.DbtrAgtBIC, row.CdtrAgtBIC,
            message_id=row.MessageId, montant=row.InstdAmt,
            statut=row.TxSts, motif_rejet=row.RejectionReason,
            horodatage=str(row.TxDateTime),
        )
    return G


def top_emetteurs(limit=10):
    df = load_data()
    g = df.groupby("DbtrAgtBIC").agg(
        montant_total=("InstdAmt", "sum"), nb_transactions=("MessageId", "count")
    ).reset_index().rename(columns={"DbtrAgtBIC": "compte"})
    return g.sort_values("montant_total", ascending=False).head(limit).to_dict("records")


def top_receveurs(limit=10):
    df = load_data()
    g = df.groupby("CdtrAgtBIC").agg(
        montant_total=("InstdAmt", "sum"), nb_transactions=("MessageId", "count")
    ).reset_index().rename(columns={"CdtrAgtBIC": "compte"})
    return g.sort_values("montant_total", ascending=False).head(limit).to_dict("records")


def transferts_circulaires(limit=25):
    """Cycles A->B->C->A dans le graphe simple (sans multi-arêtes) des comptes."""
    G = build_account_graph()
    simple = nx.DiGraph()
    simple.add_edges_from(G.edges())
    cycles = []
    for cycle in nx.simple_cycles(simple, length_bound=3):
        if len(cycle) == 3:
            cycles.append({"compte_a": cycle[0], "compte_b": cycle[1], "compte_c": cycle[2]})
        if len(cycles) >= limit:
            break
    return cycles


def beneficiaires_partages(limit=25, seuil=3):
    df = load_data()
    g = df.groupby("CdtrAgtBIC")["DbtrAgtBIC"].nunique().reset_index()
    g.columns = ["compte_receveur", "nb_emetteurs_distincts"]
    g = g[g["nb_emetteurs_distincts"] > seuil]
    return g.sort_values("nb_emetteurs_distincts", ascending=False).head(limit).to_dict("records")


def sous_graphe_transactions_rejetees(limit=100):
    df = load_data()
    rejected = df[df["TxSts"] == "RJCT"].head(limit)
    return rejected.rename(columns={
        "DbtrAgtBIC": "emetteur", "MessageId": "transaction",
        "InstdAmt": "montant", "RejectionReason": "motif_rejet", "CdtrAgtBIC": "receveur"
    })[["emetteur", "transaction", "montant", "motif_rejet", "receveur"]].to_dict("records")


def voisinage(bic: str, limit=50):
    G = build_account_graph()
    if bic not in G:
        return []
    edges = []
    for u, v, data in list(G.in_edges(bic, data=True))[:limit]:
        edges.append({"source": u, "cible": v, **data})
    for u, v, data in list(G.out_edges(bic, data=True))[:limit]:
        edges.append({"source": u, "cible": v, **data})
    return edges


if __name__ == "__main__":
    print("Top émetteurs:", top_emetteurs(5))
    print("Cycles:", transferts_circulaires(5))
    print("Bénéficiaires partagés:", beneficiaires_partages(5))
