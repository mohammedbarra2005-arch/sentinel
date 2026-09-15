"""
Wrapper autour de neo4j/queries.py et neo4j/graph_fallback.py — reprend EXACTEMENT
la même logique de bascule que dashboard/components/data_loader.py::use_neo4j_or_fallback().
Ceci concerne le graphe au niveau BIC (14 banques), utilisé par les anciennes pages
Explorateur/Graphe du dashboard Streamlit — pas les 400 comptes GAT (voir
cluster_classifier.py et gat_service.py pour ça).
"""
import os
import sys
import functools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=1)
def _engine():
    graph_dir = str(BASE_DIR / "neo4j")
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)
    try:
        from neo4j import GraphDatabase
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd = os.environ.get("NEO4J_PASSWORD", "frauddetection")
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        driver.close()
        import queries as q
        return q, "neo4j"
    except Exception:
        import graph_fallback as gf
        return gf, "networkx (fallback local, Neo4j non connecté)"


def get_engine_status() -> str:
    _, source = _engine()
    return source


def top_emetteurs(limit: int = 10) -> list[dict]:
    engine, _ = _engine()
    return engine.top_emetteurs(limit)


def top_receveurs(limit: int = 10) -> list[dict]:
    engine, _ = _engine()
    return engine.top_receveurs(limit)


def transferts_circulaires(limit: int = 25) -> list[dict]:
    engine, _ = _engine()
    return engine.transferts_circulaires(limit)


def beneficiaires_partages(limit: int = 25) -> list[dict]:
    engine, _ = _engine()
    return engine.beneficiaires_partages(limit)


def voisinage(bic: str, limit: int = 50) -> list[dict]:
    engine, _ = _engine()
    return engine.voisinage(bic, limit)
