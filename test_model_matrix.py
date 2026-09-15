"""
Teste plusieurs modèles Gemini les uns après les autres, avec un vrai aller-retour
complet de function-calling (demande d'outil + exécution + réponse finale), pour
trouver lequel fonctionne réellement sur ce compte/cette version du SDK.

Lancer depuis la racine du projet, venv activé, clé configurée :
    python test_model_matrix.py
"""
import os
import sys

if not os.environ.get("GEMINI_API_KEY"):
    print("❌ GEMINI_API_KEY n'est pas définie dans cette fenêtre.")
    sys.exit(1)

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]

tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_number",
        description="Renvoie un nombre pour un compte donné.",
        parameters={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    )
])

config = types.GenerateContentConfig(
    tools=[tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

results = {}

for model_name in CANDIDATES:
    print(f"\n--- Test : {model_name} ---")
    try:
        chat_session = client.chats.create(model=model_name, config=config)
        response = chat_session.send_message("Quel est le nombre pour le compte ACC-001 ?")
        candidate = response.candidates[0]
        calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not calls:
            print("  ⚠️  Pas d'appel d'outil demandé, on ne peut pas valider le round-trip.")
            results[model_name] = "PAS D'APPEL D'OUTIL"
            continue

        # Round-trip complet : on renvoie un faux résultat et on vérifie que
        # le modèle peut continuer sans erreur.
        function_response = types.Part.from_function_response(
            name=calls[0].name, response={"result": 42}
        )
        response2 = chat_session.send_message([function_response])
        final_text = "".join(p.text for p in response2.candidates[0].content.parts if p.text)
        print(f"  ✅ FONCTIONNE — réponse finale : {final_text[:80]}")
        results[model_name] = "OK"

    except Exception as e:
        msg = str(e)[:150]
        print(f"  ❌ ÉCHEC — {msg}")
        results[model_name] = f"ÉCHEC : {msg}"

print("\n\n=== RÉSUMÉ ===")
for model_name, status in results.items():
    print(f"{model_name:30s} {status}")

working = [m for m, s in results.items() if s == "OK"]
if working:
    print(f"\n🎉 Utilise ce modèle dans chat_service.py : {working[0]}")
else:
    print("\n😕 Aucun modèle testé ne fonctionne — colle-moi ce résumé complet.")
