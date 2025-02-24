import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import uvicorn

# === Configuration des logs et Azure Application Insights ===
instrumentation_key = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY", "ad29b7fd-4184-4a49-a1f4-97371d2cb7ae")

# Configuration du logger
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
azure_handler = AzureLogHandler(connection_string=f'InstrumentationKey={instrumentation_key}')
logger.addHandler(azure_handler)

if logger.hasHandlers():
    logger.info("Logger Azure Insights actif")
else:
    logger.warning("Attention : Azure Insights ne reçoit pas les logs")

# === Chargement du modèle ===
MODEL_DIR = os.getenv("MODEL_DIR", "models")  # Flexibilité pour test local/distant
model_path = os.path.join(MODEL_DIR, "best_model_fasttext.keras")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    logger.info("Modèle chargé avec succès !")
except Exception as e:
    logger.error(f"Erreur de chargement du modèle : {e}")
    raise ValueError(f"Erreur de chargement du modèle : {e}")

# === Chargement du tokenizer JSON ===
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer_fasttext.json")  # Adapter selon le modèle utilisé

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer non trouvé : {tokenizer_path}")

try:
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    logger.info("Tokenizer chargé avec succès !")
except Exception as e:
    logger.error(f"Erreur de chargement du tokenizer : {e}")
    raise ValueError(f"Erreur de chargement du tokenizer : {e}")

# === Initialisation de FastAPI ===
app = FastAPI()

# Redirection vers la documentation automatique
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

# Définition des classes d'entrée
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

# === Endpoint de prédiction ===
@app.post("/predict")
async def predict(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Le texte d'entrée est vide")

        # Transformer le texte en séquence avec le tokenizer
        sequence = tokenizer.texts_to_sequences([input.text])

        # Appliquer le padding pour correspondre à la taille attendue par le modèle (max_length=50)
        sequence_padded = pad_sequences(sequence, maxlen=50, padding="post", truncating="post")

        # Faire la prédiction avec le modèle
        prediction = model.predict(sequence_padded)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

        logger.info(f"Prédiction : {sentiment} | Texte : {input.text}")
        return {"prediction": sentiment}

    except HTTPException as he:
        logger.warning(f"Mauvaise requête : {he.detail}")
        raise he

    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur serveur lors de la prédiction")

# === Gestion du feedback utilisateur ===
error_feedback_counter = {}

@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Le texte du tweet est vide")
        if input.prediction not in ["positive", "negative"]:
            raise HTTPException(status_code=400, detail="Prédiction invalide")

        if not input.validation:
            logger.warning(f"Tweet mal prédit : {input.text} | Prédiction : {input.prediction}")

            error_feedback_counter[input.text] = error_feedback_counter.get(input.text, 0) + 1

            if error_feedback_counter[input.text] >= 3:
                logger.error("ALERTE : Plusieurs tweets mal prédits détectés !")

        return {"message": "Feedback enregistré, merci !"}

    except HTTPException as he:
        logger.warning(f"Mauvaise requête : {he.detail}")
        raise he

    except Exception as e:
        logger.error(f"Erreur lors du feedback : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur serveur lors du feedback")
        
@app.get("/test_log")
async def test_log():
    logger.info("Ceci est un test de log manuel envoyé à Application Insights.")
    return {"message": "Log envoyé avec succès à Azure !" }


# === Exécution locale ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
