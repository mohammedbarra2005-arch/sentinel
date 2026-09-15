"""
Chatbot avec function-calling — Groq (tier gratuit, format OpenAI-compatible).
N'appelle QUE des fonctions déjà présentes dans backend/services/ — aucune nouvelle
logique métier, juste des outils légers autour de l'existant + un peu de filtrage
côté chat (get_accounts avec filtres, par ex.) qui ne modifie rien des fichiers
d'origine du projet.

Étape 3 (RAG sur le cahier des charges) pas encore ajoutée — ce fichier ne gère que
le function-calling pour l'instant (étapes 1+2 du plan).

Note historique : la première version utilisait l'API Gemini directement. Basculé
sur Groq (modèles Llama, format OpenAI-compatible) suite à un bug non résolu côté
SDK Gemini 3 (thought_signature manquant en multi-tours) et un quota gratuit très
restrictif (20 requêtes/jour sur gemini-3.6-flash). Groq offre ~1000+ requêtes/jour
gratuites et le format function-calling standard OpenAI, beaucoup plus éprouvé.
"""
import os
import json
from openai import OpenAI

from backend.services import data_service, gat_service, xgboost_service

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """Tu es l'assistant de la plateforme Sentinel, un outil de détection de
fraude sur des transactions ISO 20022 (pacs.008/pacs.002), développé comme projet de fin
d'année. Tu réponds aux questions d'un analyste sur les transactions, les comptes, et les
modèles de détection (XGBoost pour les transactions, GAT pour les comptes).

Règles importantes :
- Utilise TOUJOURS les outils pour obtenir des chiffres réels — ne réponds JAMAIS un
  chiffre inventé ou approximatif.
- Si un outil ne renvoie pas ce qu'il faut pour répondre, dis-le clairement plutôt que
  de deviner.
- Rappelle les limites méthodologiques quand c'est pertinent : le modèle XGBoost prédit
  un proxy (rejet de transaction), pas une vraie étiquette de fraude ; le modèle GAT a
  un F1 modeste (0.29) comparé à une régression logistique de référence (0.6).
- Réponds en français, de façon concise et directe.
"""


# --- Implémentations des outils (inchangées, wrappent les services existants) --

def _tool_get_dashboard_stats() -> dict:
    return data_service.get_dashboard_stats()


def _tool_list_accounts(label: str | None = None, min_risk: float = 0.0, limit: int = 20) -> list[dict]:
    accounts = gat_service.get_all_accounts()
    if label:
        accounts = [a for a in accounts if a["label"] == label]
    accounts = [a for a in accounts if a["riskScore"] >= min_risk]
    accounts.sort(key=lambda a: a["riskScore"], reverse=True)
    return accounts[:limit]


def _tool_get_account(account_id: str) -> dict:
    account = gat_service.get_account(account_id)
    if account is None:
        return {"error": f"Compte {account_id} introuvable."}
    return account


def _tool_list_transactions(status: str | None = None, min_risk: float = 0.0, limit: int = 20) -> list[dict]:
    txs = data_service.get_transactions(limit=2000)
    if status:
        txs = [t for t in txs if t["status"] == status]
    txs = [t for t in txs if t["riskScore"] >= min_risk]
    txs.sort(key=lambda t: t["riskScore"], reverse=True)
    return txs[:limit]


def _tool_get_model_metrics() -> dict:
    xgb = xgboost_service.get_metrics()
    gat_metrics = {"f1": 0.29, "precision": 0.24, "recall": 0.37, "logisticBaselineF1": 0.6}
    return {"xgboost": xgb, "gat": gat_metrics}


def _tool_predict_account(account_id: str) -> dict:
    try:
        return gat_service.predict_account(account_id)
    except ValueError as e:
        return {"error": str(e)}


TOOL_IMPLEMENTATIONS = {
    "get_dashboard_stats": _tool_get_dashboard_stats,
    "list_accounts": _tool_list_accounts,
    "get_account": _tool_get_account,
    "list_transactions": _tool_list_transactions,
    "get_model_metrics": _tool_get_model_metrics,
    "predict_account": _tool_predict_account,
}

# --- Déclarations des outils, format OpenAI function-calling standard ------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dashboard_stats",
            "description": "Statistiques globales : nombre total de transactions, taux de rejet, "
                            "nombre de comptes à risque, nombre d'anneaux de fraude actifs, score de risque moyen.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "Liste les comptes (parmi les 400 comptes GAT), triés par score de risque décroissant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["normal", "ring", "fan_in"],
                               "description": "Filtrer par type de compte."},
                    "min_risk": {"type": "number", "description": "Score de risque minimum (0 à 1)."},
                    "limit": {"type": "integer", "description": "Nombre max de résultats (défaut 20)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account",
            "description": "Détail d'un compte précis par son account_id (ex: ACC-00270).",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "Liste les transactions, triées par score de risque décroissant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ACSP", "RJCT"],
                               "description": "Filtrer par statut (ACSP=acceptée, RJCT=rejetée)."},
                    "min_risk": {"type": "number", "description": "Score de risque minimum (0 à 1)."},
                    "limit": {"type": "integer", "description": "Nombre max de résultats (défaut 20)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Métriques des modèles : ROC-AUC/précision/rappel XGBoost, F1/précision/rappel GAT, "
                            "et le F1 de la régression logistique de référence.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_account",
            "description": "Lance une prédiction GAT en direct sur un compte qui existe déjà dans le graphe.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
]


def _get_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY n'est pas configurée. Récupère une clé gratuite sur "
            "https://console.groq.com/keys et configure-la (voir CHATBOT.md)."
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def chat(message: str, history: list[dict] | None = None) -> dict:
    """
    history : liste de {"role": "user"|"model", "text": "..."} des tours précédents.
    Retourne {"reply": str, "tool_calls": [...]} — tool_calls est renvoyé pour que le
    frontend puisse afficher "l'assistant a consulté X" si besoin (transparence).
    """
    client = _get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or []):
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    messages.append({"role": "user", "content": message})

    tool_calls_made = []

    # Boucle d'appels d'outils : au maximum 5 aller-retours pour éviter une boucle infinie
    for _ in range(5):
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS)
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return {"reply": msg.content or "", "tool_calls": tool_calls_made}

        messages.append(msg.model_dump(exclude_none=True))

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_IMPLEMENTATIONS.get(fn_name)
            if impl is None:
                result = {"error": f"Outil inconnu : {fn_name}"}
            else:
                try:
                    result = impl(**args)
                except Exception as e:
                    result = {"error": str(e)}

            tool_calls_made.append({"name": fn_name, "args": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return {"reply": "Désolé, je n'ai pas réussi à répondre après plusieurs essais.", "tool_calls": tool_calls_made}
