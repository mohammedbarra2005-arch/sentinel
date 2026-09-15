"""
Teste chat_service.chat() en entier, directement, SANS passer par FastAPI —
ça reproduit exactement ce que fait l'endpoint /chat, mais avec une trace
d'erreur courte et lisible si ça plante.

Lancer depuis la racine du projet, venv activé, clé configurée :
    python test_chat_full.py
"""
import os
import sys
import traceback

if not os.environ.get("GROQ_API_KEY"):
    print("❌ GROQ_API_KEY n'est pas définie dans cette fenêtre.")
    sys.exit(1)

sys.path.insert(0, ".")

from backend.services import chat_service

print("Question : Combien de comptes ont un score de risque supérieur à 0.8 ?\n")

try:
    result = chat_service.chat("Combien de comptes ont un score de risque supérieur à 0.8 ?")
    print("✅ SUCCÈS")
    print("Réponse :", result["reply"])
    print("Outils appelés :", result["tool_calls"])
except Exception:
    print("❌ ÉCHEC — voici la trace complète :\n")
    traceback.print_exc()
    sys.exit(1)
