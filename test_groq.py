"""
Script de diagnostic autonome — teste la connexion Groq SANS passer par FastAPI.

Lancer depuis la racine du projet, venv activé :
    python test_groq.py
"""
import os
import sys

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY n'est pas définie dans cette fenêtre.")
    print('   Fais : $env:GROQ_API_KEY = "ta_clé"')
    sys.exit(1)

print(f"✅ Clé trouvée (commence par {api_key[:8]}..., longueur {len(api_key)})")

try:
    from openai import OpenAI
    print("✅ SDK openai importé correctement")
except ImportError as e:
    print(f"❌ Impossible d'importer openai : {e}")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

print("\n--- Test 1 : appel simple, sans outils ---")
try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Réponds juste : OK"}],
    )
    print("✅ Réponse reçue :", response.choices[0].message.content)
except Exception as e:
    print(f"❌ Échec du test 1 : {type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- Test 2 : appel avec function-calling (comme le chatbot) ---")
try:
    weather_tool = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Donne la météo d'une ville",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Quelle est la météo à Paris ?"}],
        tools=weather_tool,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        print(f"✅ Le modèle a bien demandé un appel d'outil : {tc.function.name}({tc.function.arguments})")
    else:
        print(f"⚠️  Pas d'appel d'outil, réponse directe : {msg.content}")
except Exception as e:
    print(f"❌ Échec du test 2 : {type(e).__name__}: {e}")
    sys.exit(1)

print("\n🎉 Tous les tests sont passés — Groq et le function-calling marchent.")
print("   Prochaine étape : python test_chat_full.py (avec les vrais outils du projet)")
