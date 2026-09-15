"""
Requêtes Cypher d'analyse de réseau (section 9.3 du cahier des charges).
Chaque fonction retourne une liste de dicts (résultats Neo4j) prête à afficher dans le dashboard.
"""
import os
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "frauddetection")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


TOP_EMETTEURS = """
MATCH (c:Compte)-[:ENVOIE]->(t:Transaction)
RETURN c.bic AS compte, sum(t.montant) AS montant_total, count(t) AS nb_transactions
ORDER BY montant_total DESC
LIMIT $limit
"""

TOP_RECEVEURS = """
MATCH (t:Transaction)-[:RECUE_PAR]->(c:Compte)
RETURN c.bic AS compte, sum(t.montant) AS montant_total, count(t) AS nb_transactions
ORDER BY montant_total DESC
LIMIT $limit
"""

# Transferts circulaires A -> B -> C -> A (3 sauts, via les comptes émetteurs/receveurs)
CYCLES_3 = """
MATCH (a:Compte)-[:ENVOIE]->(:Transaction)-[:RECUE_PAR]->(b:Compte),
      (b:Compte)-[:ENVOIE]->(:Transaction)-[:RECUE_PAR]->(c:Compte),
      (c:Compte)-[:ENVOIE]->(:Transaction)-[:RECUE_PAR]->(a:Compte)
WHERE a <> b AND b <> c AND a <> c
RETURN DISTINCT a.bic AS compte_a, b.bic AS compte_b, c.bic AS compte_c
LIMIT $limit
"""

BENEFICIAIRES_PARTAGES = """
MATCH (t:Transaction)-[:RECUE_PAR]->(c:Compte)
WITH c, count(DISTINCT t) AS nb_tx
MATCH (e:Compte)-[:ENVOIE]->(t2:Transaction)-[:RECUE_PAR]->(c)
WITH c, count(DISTINCT e) AS nb_emetteurs_distincts
WHERE nb_emetteurs_distincts > 3
RETURN c.bic AS compte_receveur, nb_emetteurs_distincts
ORDER BY nb_emetteurs_distincts DESC
LIMIT $limit
"""

SOUS_GRAPHE_REJETEES = """
MATCH (e:Compte)-[:ENVOIE]->(t:Transaction {statut: 'RJCT'})-[:RECUE_PAR]->(r:Compte)
RETURN e.bic AS emetteur, t.message_id AS transaction, t.montant AS montant,
       t.motif_rejet AS motif_rejet, r.bic AS receveur
LIMIT $limit
"""

VOISINAGE_ENTITE = """
MATCH (n {bic: $bic})
OPTIONAL MATCH (n)-[r]-(voisin)
RETURN n, r, voisin
LIMIT $limit
"""


def run_query(query: str, **params):
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    finally:
        driver.close()


def top_emetteurs(limit=10):
    return run_query(TOP_EMETTEURS, limit=limit)


def top_receveurs(limit=10):
    return run_query(TOP_RECEVEURS, limit=limit)


def transferts_circulaires(limit=25):
    return run_query(CYCLES_3, limit=limit)


def beneficiaires_partages(limit=25):
    return run_query(BENEFICIAIRES_PARTAGES, limit=limit)


def sous_graphe_transactions_rejetees(limit=100):
    return run_query(SOUS_GRAPHE_REJETEES, limit=limit)


def voisinage(bic: str, limit=50):
    return run_query(VOISINAGE_ENTITE, bic=bic, limit=limit)


if __name__ == "__main__":
    print("Top émetteurs:", top_emetteurs(5))
    print("Transferts circulaires:", transferts_circulaires(5))
