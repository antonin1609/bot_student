import os
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"

def get_client(api_key=None):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY manquante")
    return Groq(api_key=key)

def system_prompt(mode):
    prompts = {
        "Question libre": "Réponds clairement, pédagogiquement et uniquement avec le contexte. Si une info manque, dis-le.",
        "Résumé": "Résume de manière structurée, claire et fidèle. Utilise titres et listes.",
        "Questions d'examen": "Génère 5 questions d'examen variées avec réponses attendues.",
        "Explication": "Explique simplement avec définition, exemple concret et application.",
        "Fiche de révision": "Crée une fiche de révision très claire avec notions clés, définitions et points importants."
    }
    return f"""
Tu es un assistant pédagogique expert pour étudiants universitaires.
Tu réponds dans la langue de l'utilisateur.
Tu restes précis, bienveillant, structuré et honnête.
Tu n'inventes jamais d'information absente du contexte.
Tu peux citer les sources si elles sont visibles dans le contexte.

Mode: {mode}
Consigne: {prompts[mode]}
"""
import base64
from groq import Groq

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_groq(api_key, image_file, prompt):
    client = Groq(api_key=api_key)
    encoded_image = encode_image(image_file)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_completion_tokens=1200
    )
    return response.choices[0].message.content.strip()
def build_messages(question, context, history, mode):
    messages = [{"role": "system", "content": system_prompt(mode)}]
    if history:
        messages.extend(history[-10:])
    messages.append({
        "role": "user",
        "content": f"Contexte des documents:\n{context}\n\nQuestion:\n{question}"
    })
    return messages

def stream_answer(client, question, context, history, mode, temperature=0.2, max_completion_tokens=2000):
    messages = build_messages(question, context, history, mode)
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        stream=True
    )
