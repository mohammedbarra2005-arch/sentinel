# Backend FastAPI ↔ Frontend React — Guide d'intégration

## Ce qui a changé

Le dossier `backend/` a été ajouté à ton projet Python (`FraudDetectionProject/backend/`).
Il wrappe tes fonctions existantes (`ml/predict.py`, `ml_gat/predict_gat.py`,
`neo4j/queries.py` + `graph_fallback.py`) en API REST — **aucune logique de ces
fichiers n'a été réécrite**, juste importée et exposée en JSON.

Le frontend React (`src/api/client.ts`) a été mis à jour pour appeler ce backend
au lieu des données mockées. Tout le reste (composants, pages, hooks) est inchangé.

Ton dashboard **Streamlit continue de fonctionner exactement comme avant** —
rien n'a été touché de ce côté.

## Comment lancer les deux ensemble

**Terminal 1 — Backend** (depuis la racine `FraudDetectionProject/`) :
```powershell
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```
⚠️ Lance bien `python -m uvicorn` depuis la **racine du projet**, pas depuis
`backend/` — les chemins vers `data/`, `ml/`, `ml_gat/` sont relatifs à la racine.

Vérifie que ça tourne : ouvre http://localhost:8000/health dans ton navigateur,
tu dois voir `{"status":"ok",...}`.

**Terminal 2 — Frontend** (depuis `fraud-frontend/`) :
```powershell
npm run dev
```
Ouvre http://localhost:5173 — tout devrait maintenant afficher tes vraies données
(5000 transactions, 400 comptes, etc.) au lieu des données mockées.

## Ce qui est branché et testé

| Endpoint | Utilisé par | Statut |
|---|---|---|
| `GET /dashboard/stats` | Dashboard (KPIs) | ✅ testé |
| `GET /transactions` | Transactions, Alertes, Recherche | ✅ testé |
| `GET /accounts`, `/accounts/{id}` | Profils clients, Alertes | ✅ testé |
| `GET /graph/edges`, `/graph/clusters` | Explorateur de graphe | ✅ testé |
| `GET /model-metrics` | Explicabilité | ✅ testé |
| `POST /auth/login` | Login | ✅ testé |
| `POST /predict/transaction` | Prédiction manuelle (onglet Transaction) | ✅ testé |
| `POST /predict/account/{id}` | Prédiction manuelle (onglet Compte) | ✅ testé |
| `GET /graph/bic/*` (top émetteurs, cycles...) | — pas encore branché sur une page | disponible, bonus |

## Changements côté données à connaître

- **Codes de statut réels** : `ACSP` (accepté) / `RJCT` (rejeté), pas `ACSC` comme
  dans les données mockées. L'affichage gère ça correctement (il ne vérifie que
  `RJCT`).
- **BIC réels** : 13 banques (LOYDGB21, BARCGB22, DEUTDEFF...) au lieu des BIC
  marocains inventés pour la démo.
- **Page Prédiction, onglet Compte** : le formulaire a changé. Avant (mock), tu
  entrais des totaux inventés. Maintenant, il faut entrer un `account_id` qui
  EXISTE déjà dans le graphe (ex : `ACC-00270`) — c'est une contrainte du modèle
  GAT lui-même (il a besoin de voir le compte dans son contexte de graphe complet,
  cf. docstring de `predict_gat.py`), pas une limitation ajoutée arbitrairement.

## Point technique important si tu modifies le backend

Sur cette machine, `uvicorn` sans l'option `-u` (unbuffered) peut retarder
l'affichage des logs de démarrage à cause du buffering de sortie Python. Si tu
ne vois jamais "Application startup complete" dans ton terminal alors que le
serveur répond bien, ce n'est pas un bug : ça n'affecte que l'affichage des
logs, pas le fonctionnement. Sur Windows ce problème ne devrait pas se poser
(comportement lié à la redirection de sortie dans mon environnement de test),
mais si jamais : `python -u -m uvicorn backend.main:app --reload --port 8000`.

## Reste à faire / limites connues

1. **Métriques GAT non persistées** : contrairement à `ml/train.py` (qui sauvegarde
   `metrics.json`), `ml_gat/train_gat.py` ne fait que logger ses métriques (F1,
   precision, recall) sans les sauvegarder dans un fichier. Le endpoint
   `/model-metrics` renvoie donc actuellement les valeurs de ta dernière session
   d'entraînement documentée (F1=0.29), codées en dur dans `backend/main.py`
   (fonction `model_metrics`). Si tu réentraînes le GAT et que le F1 change,
   il faudra soit mettre à jour cette valeur à la main, soit ajouter la
   sauvegarde dans `train_gat.py` (2-3 lignes, même pattern que `ml/train.py`).

2. **Classification anneau/fan-in ajoutée côté API** (`backend/services/cluster_classifier.py`) :
   ton fichier `gat_accounts_nodes.parquet` n'a qu'une étiquette binaire
   (à risque ou non), sans distinguer le TYPE de pattern. Cette info est
   reconstruite par détection de cycles/convergence dans le graphe des
   transactions — voir les commentaires en tête de ce fichier pour le détail.
   Fonctionne bien (8 anneaux + 5 fan-in retrouvés correctement, testé), mais
   c'est une logique nouvelle, pas dans ton code d'origine.

3. **`/auth/login` est un placeholder** : ton projet Python n'a pas de vraie base
   d'utilisateurs. Le endpoint accepte n'importe quel email + le mot de passe
   `demo1234`, en dur dans le code. À remplacer si ce projet doit un jour avoir
   plusieurs utilisateurs réels.

4. **Neo4j** : le backend bascule automatiquement sur NetworkX si Neo4j n'est
   pas démarré (même logique que ton dashboard Streamlit). Si tu démarres Neo4j
   via `docker compose up`, les endpoints `/graph/bic/*` l'utiliseront
   automatiquement.
