# Chatbot — Setup et état d'avancement

## Historique du choix de fournisseur

Première version sur l'API Gemini — abandonnée après une session de debug qui a
mis en évidence deux problèmes non résolus côté Gemini 3 : un bug de SDK
(`thought_signature` manquant en multi-tours de function-calling, bug confirmé
sur plusieurs frameworks différents, pas propre à ce projet) et un quota gratuit
très restrictif (20 requêtes/jour sur `gemini-3.6-flash`). Basculé sur **Groq**
(modèles Llama, format function-calling standard OpenAI) : quota gratuit
beaucoup plus généreux (~1000+ requêtes/jour) et format plus éprouvé.

## Où en est-on (voir chatbot_rag_plan.pdf pour le plan complet en 6 étapes)

- [x] **Étape 1 — Setup** : SDK `openai` installé (pointé vers Groq), `backend/services/chat_service.py` créé
- [x] **Étape 2 — Function-calling sur l'existant** : 6 outils branchés sur tes vrais
      services (dashboard, comptes, transactions, métriques, prédiction GAT). Testé
      avec un client Groq simulé (mocké) — la boucle d'appel d'outils fonctionne
      correctement et exécute bien tes vraies fonctions sur tes vraies données.
- [ ] **Étape 3 — RAG sur la documentation** : pas commencée
- [ ] **Étape 4 — Orchestration avancée** : pas commencée
- [ ] **Étape 5 — Widget frontend** : pas commencée (le chatbot n'existe QUE côté
      backend pour l'instant — endpoint `/chat`, pas encore de bouton/fenêtre dans
      l'interface React)
- [ ] **Étape 6 — Garde-fous et tests** : partiellement fait (le prompt système
      demande explicitement au modèle de ne jamais inventer un chiffre)

## Obtenir une clé API Groq (gratuite)

1. Va sur https://console.groq.com/keys
2. Connecte-toi (email, Google, ou GitHub) — pas de carte bancaire demandée
3. Clique sur "Create API Key"
4. Copie la clé (elle ne sera montrée qu'une fois)

## Configurer la clé

Dans la fenêtre où tu vas lancer le serveur :
```powershell
$env:GROQ_API_KEY = "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"
```

## Tester (dans cet ordre)

**1. Connexion Groq toute seule, sans le projet :**
```powershell
python test_groq.py
```

**2. Le chatbot complet avec tes vraies données, sans passer par FastAPI :**
```powershell
python test_chat_full.py
```

**3. Une fois les deux OK, lance le serveur normalement :**
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```
Puis, dans une autre fenêtre :
```powershell
Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body '{"message": "Combien de comptes a risque avec un score superieur a 0.8 ?"}'
```

## Outils actuellement disponibles pour le chatbot

| Outil | Ce qu'il fait |
|---|---|
| `get_dashboard_stats` | Stats globales (total transactions, taux de rejet, comptes à risque...) |
| `list_accounts` | Liste de comptes, filtrable par type (ring/fan_in/normal) et score min |
| `get_account` | Détail d'un compte précis |
| `list_transactions` | Liste de transactions, filtrable par statut et score min |
| `get_model_metrics` | Métriques XGBoost et GAT |
| `predict_account` | Lance une vraie prédiction GAT sur un compte existant |

## Prochaine session

Reprendre à l'étape 3 (RAG) : indexer `docs/Cahier_des_Charges_Fraude_ISO20022.pdf`
avec ChromaDB, puis étape 4 pour que le chatbot bascule entre outils et RAG selon
la question, puis étape 5 pour construire le widget dans le frontend React (pas
encore commencée). Plan détaillé dans `chatbot_rag_plan.pdf`.
