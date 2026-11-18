
import os
from pathlib import Path

import mlflow


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/Victoriabrl/ML-Flow-Daghub.mlflow"
)

EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "0")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))


def download_latest_model():
    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"Recherche des modèles dans l'expérience {EXPERIMENT_ID}")

    models = mlflow.search_logged_models(
        experiment_ids=[EXPERIMENT_ID],
        output_format="list"
    )

    ready_models = [
        model for model in models
        if str(model.status).upper().endswith("READY")
    ]

    if not ready_models:
        raise RuntimeError("Aucun modèle READY trouvé sur DagsHub.")

    latest_model = max(
        ready_models,
        key=lambda model: model.creation_timestamp
    )

    print(f"Modèle sélectionné : {latest_model.name}")
    print(f"Model ID : {latest_model.model_id}")

    if not latest_model.artifact_location:
        raise RuntimeError("Le modèle ne possède pas d'artifact_location.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=latest_model.artifact_location,
        dst_path=str(MODEL_DIR)
    )

    print(f"Modèle téléchargé dans : {downloaded_path}")


if __name__ == "__main__":
    download_latest_model()
