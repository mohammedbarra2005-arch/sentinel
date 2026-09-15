# Plateforme de Détection de Fraude ISO 20022

Projet de Fin d'Année (PFA) — EMSI Casablanca, Cycle Ingénieur Data Science & IA.
Combine une base de données graphe (**Neo4j**), un modèle de Machine Learning (**XGBoost**) et
un tableau de bord interactif (**Streamlit**) pour analyser des paiements ISO 20022 (pacs.008/pacs.002)
et estimer un score de risque par transaction.

📄 Le cahier des charges complet est dans `docs/Cahier_des_Charges_Fraude_ISO20022.pdf`.

## ⚠️ Note méthodologique importante

Aucune des sources de données ne contient d'étiquette de fraude confirmée. Le modèle est donc
entraîné sur une **cible de substitution** (`TxSts = RJCT`, taux de rejet 6,4 %) et restitue une
**probabilité de risque/anomalie**, pas une probabilité de fraude auditée. Cette limite est
documentée explicitement dans le dashboard (page Évaluation). Sur ce jeu de données synthétique,
le signal prédictif est volontairement modeste (ROC-AUC ≈ 0.59) — attendu puisque le rejet y est
généré de façon largement indépendante des variables disponibles ; l'intérêt du projet est
l'architecture bout-en-bout (graphe + ML + dashboard), pas la performance du modèle en tant que telle.

## Structure du projet

```
FraudDetectionProject/
├── data/
│   ├── raw/            # CSV sources (iso20022_analytics, pacs002, pacs008)
│   ├── processed/       # Données nettoyées + features (générées par les scripts)
│   └── models/          # Modèle XGBoost sérialisé + métriques (générés)
├── neo4j/
│   ├── import_graph.py  # Construction du graphe (nécessite une instance Neo4j)
│   ├── queries.py       # Requêtes Cypher (top émetteurs, cycles, bénéficiaires partagés...)
│   └── graph_fallback.py # Moteur NetworkX de secours (fonctionne sans Neo4j démarré)
├── preprocessing/
│   ├── cleaning.py
│   └── feature_engineering.py
├── ml/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py      # SHAP
├── dashboard/
│   ├── app.py
│   ├── pages/            # 7 pages Streamlit
│   └── components/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Lancement rapide (sans Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline de données et modèle
python preprocessing/cleaning.py
python preprocessing/feature_engineering.py
python ml/train.py
python ml/evaluate.py

# Dashboard (fonctionne même sans Neo4j démarré : bascule automatique sur NetworkX)
streamlit run dashboard/app.py
```

## Lancement avec Docker (Neo4j + dashboard)

```bash
docker compose up --build
```

- Dashboard : http://localhost:8501
- Neo4j Browser : http://localhost:7474 (identifiants : `neo4j` / `frauddetection`)

Une fois Neo4j démarré, peuplez le graphe :

```bash
docker compose exec dashboard python neo4j/import_graph.py
```

Le dashboard détecte automatiquement une instance Neo4j active et bascule dessus ; sinon il
utilise le moteur NetworkX local (`neo4j/graph_fallback.py`), pour une démo sans dépendance externe.

## Résultats du modèle (test set, 20%)

| Métrique | Valeur |
|---|---|
| Accuracy | voir `data/models/metrics.json` après entraînement |
| ROC-AUC | ≈ 0.59 (cf. note méthodologique ci-dessus) |

Les métriques exactes et l'importance des variables (gain XGBoost + SHAP) sont générées dans
`data/models/metrics.json` et `data/models/shap_importance.json`, et affichées dans la page
**Évaluation** du dashboard.

## Pages du dashboard

1. **Accueil** — KPIs globaux
2. **Explorateur** — recherche, filtres, pagination, export CSV
3. **Analytique** — visualisations Plotly (temporel, montants, devises, heatmap horaire)
4. **Prédiction** — score de risque en temps réel pour une transaction saisie manuellement
5. **Graphe** — top émetteurs/receveurs, cycles A→B→C→A, bénéficiaires partagés, voisinage d'une entité
6. **Évaluation** — ROC/AUC, matrice de confusion, importance des variables (gain + SHAP)
7. **Transactions suspectes** — triées par score de risque décroissant, export CSV

## Perspectives d'amélioration

Ingestion Kafka temps réel, traitement Spark à l'échelle, Graph Neural Networks, approches non
supervisées (Isolation Forest, autoencodeurs), API REST FastAPI, déploiement cloud multi-utilisateurs.
Voir section 15 du cahier des charges pour le détail.
