"""
Module 3 — Construction du graphe Neo4j
Noeuds : Compte, Banque, Transaction, Pays
Relations : ENVOIE, RECUE_PAR, RATTACHE_A, LOCALISEE_EN

Nécessite une instance Neo4j accessible (voir docker-compose.yml à la racine).
Variables d'environnement attendues : NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
(valeurs par défaut ci-dessous pour un déploiement local/Docker).
"""
import os
import logging
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_graph")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "frauddetection")

CONSTRAINTS = [
    "CREATE CONSTRAINT compte_bic IF NOT EXISTS FOR (c:Compte) REQUIRE c.bic IS UNIQUE",
    "CREATE CONSTRAINT banque_bic IF NOT EXISTS FOR (b:Banque) REQUIRE b.bic IS UNIQUE",
    "CREATE CONSTRAINT pays_code IF NOT EXISTS FOR (p:Pays) REQUIRE p.code_iso IS UNIQUE",
    "CREATE CONSTRAINT tx_msgid IF NOT EXISTS FOR (t:Transaction) REQUIRE t.message_id IS UNIQUE",
]

# Un compte est modélisé au niveau du BIC (le jeu de données synthétique n'expose pas
# d'IBAN/numéro de compte individuel) : chaque BIC = un "compte" agrégateur rattaché à sa banque.
IMPORT_NODES_PAYS = """
UNWIND $rows AS row
MERGE (p:Pays {code_iso: row.code})
"""

IMPORT_NODES_BANQUE = """
UNWIND $rows AS row
MERGE (b:Banque {bic: row.bic})
SET b.pays = row.pays
WITH b, row
MATCH (p:Pays {code_iso: row.pays})
MERGE (b)-[:LOCALISEE_EN]->(p)
"""

IMPORT_NODES_COMPTE = """
UNWIND $rows AS row
MERGE (c:Compte {bic: row.bic})
WITH c, row
MATCH (b:Banque {bic: row.bic})
MERGE (c)-[:RATTACHE_A]->(b)
"""

IMPORT_TRANSACTIONS = """
UNWIND $rows AS row
MERGE (t:Transaction {message_id: row.message_id})
SET t.end_to_end_id = row.end_to_end_id,
    t.montant = row.montant,
    t.devise = row.devise,
    t.statut = row.statut,
    t.motif_rejet = row.motif_rejet,
    t.horodatage = row.horodatage
WITH t, row
MATCH (emetteur:Compte {bic: row.dbtr_bic})
MATCH (recepteur:Compte {bic: row.cdtr_bic})
MERGE (emetteur)-[:ENVOIE]->(t)
MERGE (t)-[:RECUE_PAR]->(recepteur)
"""

BATCH_SIZE = 500


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_import():
    df = pd.read_parquet(PROCESSED_DIR / "analytics_clean.parquet")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        logger.info("Création des contraintes...")
        for c in CONSTRAINTS:
            session.run(c)

        pays = pd.concat([df["DbtrCountry"], df["CdtrCountry"]]).dropna().unique()
        pays_rows = [{"code": p} for p in pays]
        session.run(IMPORT_NODES_PAYS, rows=pays_rows)
        logger.info("Noeuds Pays créés: %d", len(pays_rows))

        banques = pd.concat([
            df[["DbtrAgtBIC", "DbtrCountry"]].rename(columns={"DbtrAgtBIC": "bic", "DbtrCountry": "pays"}),
            df[["CdtrAgtBIC", "CdtrCountry"]].rename(columns={"CdtrAgtBIC": "bic", "CdtrCountry": "pays"}),
        ]).drop_duplicates(subset=["bic"])
        banque_rows = banques.to_dict("records")
        session.run(IMPORT_NODES_BANQUE, rows=banque_rows)
        logger.info("Noeuds Banque créés: %d", len(banque_rows))

        comptes = pd.concat([df["DbtrAgtBIC"], df["CdtrAgtBIC"]]).drop_duplicates()
        compte_rows = [{"bic": b} for b in comptes]
        session.run(IMPORT_NODES_COMPTE, rows=compte_rows)
        logger.info("Noeuds Compte créés: %d", len(compte_rows))

        tx_rows = [
            {
                "message_id": r.MessageId,
                "end_to_end_id": r.EndToEndId,
                "montant": float(r.InstdAmt),
                "devise": r.Currency,
                "statut": r.TxSts,
                "motif_rejet": r.RejectionReason,
                "horodatage": str(r.TxDateTime),
                "dbtr_bic": r.DbtrAgtBIC,
                "cdtr_bic": r.CdtrAgtBIC,
            }
            for r in df.itertuples()
        ]
        for i, batch in enumerate(chunks(tx_rows, BATCH_SIZE)):
            session.run(IMPORT_TRANSACTIONS, rows=batch)
            logger.info("Batch transactions %d importé (%d lignes)", i + 1, len(batch))

    driver.close()
    logger.info("Import du graphe terminé : %d transactions, %d comptes, %d banques, %d pays",
                len(tx_rows), len(compte_rows), len(banque_rows), len(pays_rows))


if __name__ == "__main__":
    run_import()
