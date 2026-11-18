
import os

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Chemin du modèle téléchargé pendant la construction Docker
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")


# Création de l'API
app = FastAPI(
    title="Digits Prediction API",
    description="API de prédiction de chiffres manuscrits",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)


# Schéma des données reçues
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Liste de 64 valeurs représentant une image 8x8"
    )


# Chargement du modèle MLflow local
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print(f"Modèle MLflow chargé depuis {MODEL_PATH}")

except Exception as e:
    raise RuntimeError(
        f"Impossible de charger le modèle MLflow local : {e}"
    )


# Endpoint unique de prédiction
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Conversion des 64 valeurs en tableau NumPy
        data = np.array(
            request.features,
            dtype=float
        ).reshape(1, 64)

        # Prédiction
        prediction = model.predict(data)

        # Résultat
        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
