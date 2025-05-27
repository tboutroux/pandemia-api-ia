import os
from fastapi import APIRouter, HTTPException
from app.ai.database import create_db_engine, load_data
from app.ai.data_processing import create_features
from app.ai.model import PandemicModel
from app.ai.visualization import plot_predictions
from dotenv import load_dotenv
import numpy as np
from fastapi import Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.security import verify_token, security
from app.config.settings import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)

@router.get("/predict")
def predict(
    country_name: str = "France",
    target: str = "new_cases",
    days_ahead: int = 7,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    verify_token(credentials)
    """
    Route pour prédire les cas de pandémie pour un pays donné.
    - **country_name**: Nom du pays pour lequel faire la prédiction (par défaut "France").
    - **target**: La cible de la prédiction, par exemple "new_cases" ou "new_deaths".
    - **days_ahead**: Nombre de jours pour lesquels faire la prédiction (par défaut 7).
    - **credentials**: Clé API pour sécuriser l'accès à la route.
    - **returns**: Un dictionnaire contenant les prédictions pour le pays et la cible spécifiés.
    - **raises**: HTTPException en cas d'erreur lors de la prédiction ou de la configuration de la base de données.
    - **example**:
    ```json
    {
        "country": "France",
        "target": "new_cases",
        "days_ahead": 7,
        "predictions": {
            "0": {"date": "2023-10-01", "prediction": 100},
            "1": {"date": "2023-10-02", "prediction": 120},
            ...
        }
    }
    """

    try:
        # Vérification des variables d'environnement
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise ValueError("Les variables d'environnement de la base de données ne sont pas correctement configurées.")

        # Initialisation
        engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

        # Chargement des données
        df = load_data(engine, country_name, targets=[target])

        # Création des features pour chaque cible
        df_features = create_features(df.copy(), target, look_back=30, 
                                    use_lags=True, use_rolling=True, use_calendar=True)

        feature_names = [col for col in df_features.columns 
            if col.startswith('lag_') or 
            col.startswith('rolling_') or 
            col in ['day_of_week', 'day_of_month', 'month', 
                    'cases_per_100k', 'deaths_per_100k', 'recovered_per_100k']
            and col in df_features.columns]

        # Initialisation du gestionnaire de modèles
        model_manager = PandemicModel()

        # Prédictions futures
        predictions = model_manager.predict_future(
            df_features, target, feature_names=feature_names, 
            days_ahead=days_ahead
        )

        # Enregistrement du graphique
        plot_predictions(df, predictions, target, country_name)

        # Conversion des prédictions en liste
        predictions = predictions.to_dict(orient="records")

        predictions = {i: pred for i, pred in enumerate(predictions)}

        # Retourne les prédictions sous forme de liste (ou adapter selon besoin)
        return {
            "country": country_name,
            "target": target,
            "days_ahead": days_ahead,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
