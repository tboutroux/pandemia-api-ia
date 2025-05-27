import os
from fastapi import APIRouter, HTTPException
from app.ai.database import create_db_engine, load_data
from app.ai.data_processing import create_features
from app.ai.model import PandemicModel
from app.ai.visualization import visualize_all_results
from app.core.security import security
from dotenv import load_dotenv
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials
from app.config.settings import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
from fastapi import Query
import pandas as pd

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
)

@router.get("/predict")
def predict(
    country: str = "France",
    days: int = 7,
    no_train: bool = False,
    tune: bool = False,
    targets: str = Query("new_cases,new_deaths,new_recovered", description="Liste des cibles séparées par des virgules"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Route pour prédire les cas de pandémie pour un pays donné.
    - **country**: Nom du pays à prédire (par défaut "France")
    - **days**: Nombre de jours à prédire (par défaut 7)
    - **no_train**: Utiliser les modèles existants sans ré-entraînement (par défaut False)
    - **tune**: Effectuer un tuning des hyperparamètres (par défaut False)
    - **targets**: Cibles à prédire (par défaut: "new_cases,new_deaths,new_recovered")
    - **credentials**: Clé API pour sécuriser l'accès à la route
    - **returns**: Un dictionnaire contenant les résultats de la prédiction
    """
    try:
        # Convertir la chaîne de targets en liste
        targets_list = [t.strip() for t in targets.split(",") if t.strip()]
        
        valid_targets = ["new_cases", "new_deaths", "new_recovered"]
        for target in targets_list:
            if target not in valid_targets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target '{target}' invalide. Les targets valides sont: {valid_targets}"
                )

        # Si aucune target valide n'est spécifiée
        if not targets_list:
            targets_list = ["new_cases", "new_deaths", "new_recovered"]

        # Vérification des variables d'environnement
        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise ValueError("Les variables d'environnement de la base de données ne sont pas correctement configurées.")

        # Initialisation
        engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)
        country_name = country
        days_ahead = days

        # Chargement des données (en passant les targets)
        df = load_data(engine, country_name, targets=targets_list)

        model_manager = PandemicModel()
        predictions = {}
        metrics = {}

        for target in targets_list:
            # Vérifier que la colonne cible existe bien dans les données chargées
            if target not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"La cible '{target}' n'existe pas dans les données chargées."
                )
                continue

            # Création des features pour chaque cible
            df_features = create_features(df.copy(), target, look_back=30, 
                                        use_lags=True, use_rolling=True, use_calendar=True)

            # Définition de la liste des features
            feature_names = [col for col in df_features.columns 
                if col.startswith('lag_') or 
                col.startswith('rolling_') or 
                col in ['day_of_week', 'day_of_month', 'month', 
                        'cases_per_100k', 'deaths_per_100k', 'recovered_per_100k']
                and col in df_features.columns]

            if not no_train:
                # Entraînement du modèle avec option de tuning
                model, target_metrics = model_manager.train_model(
                    df_features, target, feature_names=feature_names, 
                    tune_hyperparams=tune
                )
                # Sauvegarde du modèle
                model_manager.save_model(model, target)
                metrics[target] = target_metrics
            else:
                # Chargement du modèle existant
                model = model_manager.load_model(target)
                if model is None:
                    model, target_metrics = model_manager.train_model(
                        df_features, target, feature_names=feature_names, 
                        tune_hyperparams=tune
                    )
                    model_manager.save_model(model, target)
                    metrics[target] = target_metrics

            # Prédictions futures
            preds = model_manager.predict_future(
                df_features, target, feature_names=feature_names, 
                days_ahead=days_ahead
            )
            
            # On clippe les valeurs prédites pour éviter les valeurs négatives
            preds[f"predicted_{target}"] = preds[f"predicted_{target}"].clip(lower=0)
            predictions[target] = preds.to_dict(orient="records")

            # Sauvegarde des prédictions dans un CSV
            preds.to_csv(f"visualization/{country_name}_{target}_predictions.csv")

        # Génération de toutes les visualisations
        if predictions: 
            visualize_all_results(df, {k: pd.DataFrame(v) for k, v in predictions.items()}, country_name)
            
            # Sauvegarde des métriques
            for target, target_metrics in metrics.items():
                with open(f"visualization/{country_name}_{target}_metrics.txt", "w") as f:
                    for key, value in target_metrics.items():
                        f.write(f"{key}: {value}\n")

            # Préparation de la réponse
            response = {
                "country": country_name,
                "days_ahead": days_ahead,
                "predictions": predictions,
                "metrics": metrics
            }
            
            return response
        else:
            raise HTTPException(
                status_code=400,
                detail="Aucune prédiction générée. Vérifiez les données chargées et les targets spécifiées."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )