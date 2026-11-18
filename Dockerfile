
# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application et le script de téléchargement
COPY app/ ./app/
COPY scripts/ ./scripts/

# Paramètres MLflow
ARG MLFLOW_TRACKING_URI=https://dagshub.com/Victoriabrl/ML-Flow-Daghub.mlflow
ARG MLFLOW_EXPERIMENT_ID=0

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_EXPERIMENT_ID=${MLFLOW_EXPERIMENT_ID}
ENV MODEL_PATH=/app/model

# Télécharger le dernier modèle READY depuis DagsHub
# Les identifiants sont fournis temporairement par BuildKit
RUN --mount=type=secret,id=dagshub_username \
    --mount=type=secret,id=dagshub_token \
    export MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_username)" && \
    export MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" && \
    python scripts/download_model.py

# Port de l'API
EXPOSE 8000

# Démarrer FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
