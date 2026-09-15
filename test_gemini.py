"""
Script de diagnostic autonome — teste la connexion Gemini SANS passer par
FastAPI, pour isoler le problème rapidement.

Lancer depuis la racine du projet, venv activé :
    python test_gemini.py
"""
import os
import sys

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY n'est pas définie dans cette fenêtre.")
    print('   Fais : $env:GEMINI_API_KEY = "ta_clé"')
    sys.exit(1)

print(f"✅ Clé trouvée (commence par {api_key[:8]}..., longueur {len(api_key)})")

try:
    from google import genai
    print("✅ SDK google-genai importé correctement")
except ImportError as e:
    print(f"❌ Impossible d'importer google.genai : {e}")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("\n--- Test 1 : appel simple, sans outils ---")
try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Réponds juste : OK",
    )
    print("✅ Réponse reçue :", response.text)
except Exception as e:
    print(f"❌ Échec du test 1 : {type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- Test 2 : appel avec function-calling (comme le chatbot) ---")
try:
    from google.genai import types

    weather_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Donne la météo d'une ville",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ])

    chat_session = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(tools=[weather_tool]),
    )
    response = chat_session.send_message("Quelle est la météo à Paris ?")
    candidate = response.candidates[0]
    calls = [p.function_call for p in candidate.content.parts if p.function_call]

    if calls:
        print(f"✅ Le modèle a bien demandé un appel d'outil : {calls[0].name}({dict(calls[0].args)})")
    else:
        text = "".join(p.text for p in candidate.content.parts if p.text)
        print(f"⚠️  Pas d'appel d'outil, réponse directe : {text}")

except Exception as e:
    print(f"❌ Échec du test 2 : {type(e).__name__}: {e}")
    sys.exit(1)

print("\n🎉 Tous les tests sont passés — Gemini et le function-calling marchent.")
print("   Si le chatbot FastAPI plante encore, le problème est ailleurs (nos outils, le endpoint).")
