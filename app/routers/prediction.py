import os
from fastapi import APIRouter
from app.ai.database import create_db_engine, load_data
from app.ai.data_processing import create_features
from app.ai.model import PandemicModel
from app.ai.visualization import plot_predictions

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)


@router.get("/predict")
def predict(country_name: str = "France", target: str = "new_cases", days_ahead: int = 7):
    """
    Ce projet implémente une IA pour prédire l'évolution des cas ou des décès liés à une maladie dans un pays donné.
    L'objectif principal est de fournir des prédictions basées sur des données historiques, en utilisant des modèles
    d'apprentissage automatique (XGBoost). Les étapes incluent :

    1. Chargement des données depuis une base de données MySQL.
    2. Préparation des données avec des features temporelles (décalages, moyennes mobiles, etc.).
    3. Entraînement d'un modèle de prédiction pour estimer les valeurs futures.
    4. Génération de graphiques comparant les données historiques et les prédictions.

    Ce système peut être utilisé pour anticiper les tendances et aider à la prise de décision dans des contextes
    sanitaires ou épidémiologiques.
    """

    try:
        # Configuration de la base de données
        DB_USER = os.environ.get("DB_USER")
        DB_PASSWORD = os.environ.get("DB_PASSWORD")
        DB_HOST = os.environ.get("DB_HOST")
        DB_NAME = os.environ.get("DB_NAME")

        # Vérification des variables d'environnement
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise ValueError("Les variables d'environnement de la base de données ne sont pas correctement configurées.")

        # Initialisation
        engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

        # Chargement des données
        df = load_data(engine, country_name)

        # Création des features
        df = create_features(df, target)

        # Initialisation du gestionnaire de modèles
        model_manager = PandemicModel()

        # Prédictions futures
        predictions = model_manager.predict_future(df, target, days_ahead=days_ahead)

        # Enregistrement du graphique
        plot_predictions(df, predictions, target, country_name)

        # Retourne les prédictions sous forme de liste (ou adapter selon besoin)
        return {
            "country": country_name,
            "target": target,
            "days_ahead": days_ahead,
            "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions
        }

    except Exception as e:
        return {"error": str(e)}