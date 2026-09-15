"""
Backend FastAPI — wrapper REST autour du code ML/graphe existant du projet.
Ne réentraîne rien, ne réécrit aucune logique de ml/, ml_gat/ ou neo4j/ :
lit les modèles et données déjà générés, et les expose en JSON pour le
frontend React (voir FRONTEND_INTEGRATION.md pour le câblage côté React).

Lancement :
    uvicorn backend.main:app --reload --port 8000
(depuis la racine du projet, PAS depuis backend/ — les imports supposent
BASE_DIR = racine du projet)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.schemas import (
    AccountOut, AccountPredictionOut, DashboardStatsOut, GraphClusterOut,
    GraphEdgeOut, LoginIn, ModelMetricsOut, PredictionOut,
    TransactionOut, TransactionPredictionIn, UserOut,
)
from backend.services import chat_service, cluster_classifier, data_service, gat_service, neo4j_service, xgboost_service

app = FastAPI(title="Fraud Detection API", version="0.1.0")

# Le frontend React tourne sur localhost:5173 (Vite) en dev. Ajoute d'autres
# origines ici si tu déploies ailleurs (ex: un domaine de prod).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # npm run dev
        "http://localhost:4173", "http://127.0.0.1:4173",  # npm run preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "graph_engine": neo4j_service.get_engine_status()}


# --- Transactions & dashboard -------------------------------------------------

@app.get("/transactions", response_model=list[TransactionOut])
def list_transactions(limit: int = 500):
    return data_service.get_transactions(limit=limit)


@app.get("/dashboard/stats", response_model=DashboardStatsOut)
def dashboard_stats():
    return data_service.get_dashboard_stats()


# --- Comptes (GAT) --------------------------------------------------------

@app.get("/accounts", response_model=list[AccountOut])
def list_accounts():
    return gat_service.get_all_accounts()


@app.get("/accounts/{account_id}", response_model=AccountOut)
def get_account(account_id: str):
    account = gat_service.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Compte introuvable : {account_id}")
    return account


# --- Graphe (400 comptes, anneaux/fan-in) ---------------------------------

@app.get("/graph/edges", response_model=list[GraphEdgeOut])
def graph_edges():
    return cluster_classifier.get_graph_edges()


@app.get("/graph/clusters", response_model=list[GraphClusterOut])
def graph_clusters():
    return cluster_classifier.get_clusters()


# --- Graphe niveau BIC (Neo4j / fallback NetworkX) ------------------------
# Analyses bonus issues de neo4j/queries.py + graph_fallback.py — pas encore
# branchées sur une page du frontend React actuel, mais prêtes à l'emploi.

@app.get("/graph/bic/top-emitters")
def bic_top_emitters(limit: int = 10):
    return neo4j_service.top_emetteurs(limit)


@app.get("/graph/bic/top-receivers")
def bic_top_receivers(limit: int = 10):
    return neo4j_service.top_receveurs(limit)


@app.get("/graph/bic/cycles")
def bic_cycles(limit: int = 25):
    return neo4j_service.transferts_circulaires(limit)


@app.get("/graph/bic/shared-beneficiaries")
def bic_shared_beneficiaries(limit: int = 25):
    return neo4j_service.beneficiaires_partages(limit)


@app.get("/graph/bic/neighborhood/{bic}")
def bic_neighborhood(bic: str, limit: int = 50):
    return neo4j_service.voisinage(bic, limit)


# --- Modèles (métriques + explicabilité) ----------------------------------

@app.get("/model-metrics", response_model=ModelMetricsOut)
def model_metrics():
    xgb = xgboost_service.get_metrics()
    # Le F1 du GAT n'est pas persisté dans un fichier par train_gat.py (contrairement
    # à ml/train.py pour XGBoost) — seulement loggé pendant l'entraînement. Valeur
    # reprise de la dernière session d'entraînement documentée. Voir la note dans
    # FRONTEND_INTEGRATION.md pour comment la régénérer/persister si tu veux une
    # valeur toujours à jour automatiquement.
    gat_metrics = {"f1": 0.29, "precision": 0.24, "recall": 0.37, "logisticBaselineF1": 0.6}
    return {"xgboost": xgb, "gat": gat_metrics}


# --- Prédiction à la demande -----------------------------------------------

@app.post("/predict/transaction", response_model=PredictionOut)
def predict_transaction(payload: TransactionPredictionIn):
    return xgboost_service.predict_transaction(payload.model_dump())


@app.post("/predict/account/{account_id}", response_model=AccountPredictionOut)
def predict_account(account_id: str):
    try:
        return gat_service.predict_account(account_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Auth (placeholder — aucun système d'utilisateurs réel dans le projet) --
# Le projet Python n'a pas de base d'utilisateurs. Ceci reproduit le comportement
# mocké du frontend (email + mot de passe "demo1234") pour que le login continue
# de fonctionner une fois branché. À remplacer par une vraie table users +
# hashing de mot de passe si ce projet doit un jour avoir plusieurs utilisateurs.

@app.post("/auth/login", response_model=UserOut)
def login(payload: LoginIn):
    if "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Adresse email invalide.")
    if payload.password != "demo1234":
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    return {"id": "usr-001", "name": "Mohammed A.", "email": payload.email, "role": "analyst"}


# --- Chatbot (function-calling sur les endpoints ci-dessus) ----------------
# Nécessite GEMINI_API_KEY dans l'environnement (clé gratuite sur
# https://aistudio.google.com/apikey) — voir FRONTEND_INTEGRATION.md.

class ChatTurn(BaseModel):
    role: str  # "user" ou "model"
    text: str


class ChatIn(BaseModel):
    message: str
    history: list[ChatTurn] = []


class ChatOut(BaseModel):
    reply: str
    toolCalls: list[dict]


@app.post("/chat", response_model=ChatOut)
def chat_endpoint(payload: ChatIn):
    try:
        result = chat_service.chat(
            payload.message,
            history=[t.model_dump() for t in payload.history],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"reply": result["reply"], "toolCalls": result["tool_calls"]}
