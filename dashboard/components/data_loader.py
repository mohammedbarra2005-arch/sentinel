import json
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"


@st.cache_data
def load_analytics_clean():
    return pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")


@st.cache_data
def load_scored_transactions():
    return pd.read_parquet(PROCESSED_DIR / "scored_transactions.parquet")


@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_shap_importance():
    path = MODELS_DIR / "shap_importance.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def use_neo4j_or_fallback():
    """Tente une connexion Neo4j (docker-compose), sinon bascule sur le moteur NetworkX local.
    Le dossier neo4j/ est ajouté lui-même à sys.path (pas la racine) pour éviter toute collision
    de nom avec le package pip `neo4j`."""
    import os
    import sys
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
