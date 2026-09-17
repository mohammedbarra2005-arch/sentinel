# Sentinel — Frontend React (ISO 20022 Fraud Detection)

## Démarrage rapide

```bash
npm install
npm run dev
```

Ouvre ensuite http://localhost:5173

## Build de production

```bash
npm run build
npm run preview
```

## Structure

- `src/api/mockData.ts` — données mockées (transactions, comptes GAT, métriques modèles)
- `src/api/client.ts` — **point de bascule unique** vers le futur backend FastAPI.
  Remplace le corps de chaque fonction par un `fetch('/api/...')` quand le backend
  sera prêt ; aucun composant ni hook n'aura besoin de changer.
- `src/hooks/useApi.ts` — hooks React Query (un par ressource)
- `src/components/layout/` — Sidebar, Topbar, AppLayout
- `src/components/ui/` — Button, Card, Badge, NetworkMotif (motif signature), ComingSoon
- `src/pages/` — une page par route

## État d'avancement (étape par étape)

- [x] Étape 1 — Setup (ce livrable)
- [x] Étape 2 — Login
- [x] Étape 3 — Dashboard (KPIs + graphiques Recharts : volume, devises, distribution du risque)
- [x] Étape 4 — Transactions (table paginée, filtres statut/devise/recherche/risque min, tri par colonne)
- [x] Étape 5 — Alertes fraude (comptes GAT + transactions XGBoost, sévérité, résolution persistée, lien vers Transactions)
- [x] Étape 6 — Profils clients (liste filtrable + fiche détaillée avec flux du graphe)
- [x] Étape 7 — Explorateur de graphe Neo4j (layout déterministe, anneaux cycliques, fan-in convergents, recherche + focus voisinage, filtre par type, clic → fiche compte)
- [x] Étape 8 — Analytics (filtre par période, tendance acceptées/rejetées, top BIC, export CSV)
- [x] Étape 9 — Explicabilité du modèle (SHAP XGBoost, comparaison GAT vs régression logistique de référence)
- [x] Étape 10 — Upload dataset (drag & drop, validation de schéma, aperçu, pipeline simulé, CSV d'exemple)
- [x] Étape 11 — Prédiction manuelle (branché sur le vrai backend : XGBoost + GAT)
- [x] Étape 12 — Recherche globale (MessageId/account_id/BIC, résultats groupés, bouton Topbar connecté)

## Backend connecté ✅
Le frontend appelle maintenant le vrai backend FastAPI (voir
`FRONTEND_INTEGRATION.md` à la racine du projet Python). `VITE_API_BASE_URL`
dans `.env` pointe vers `http://localhost:8000` par défaut.

## Chatbot ✅
Widget de chat flottant (bouton en bas à droite, sur toutes les pages),
branché sur `POST /chat`. Voir `CHATBOT.md` côté projet Python pour la config
de la clé Groq.

## Toutes les 12 étapes du frontend React sont terminées.
Prochaine étape naturelle : le backend FastAPI (voir section "Ce qui a été
explicitement demandé mais pas encore fait" côté projet Python), pour
remplacer `src/api/client.ts` par de vrais appels réseau.
