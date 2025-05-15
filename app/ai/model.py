from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class PandemicModel:
    def __init__(self, model_dir="models"):
        self.models = {}
        self.model_dir = model_dir

    def train_model(self, df: pd.DataFrame, target: str, look_back: int = 30, test_size: float = 0.2):
        """
        Entraîne un modèle XGBoost pour prédire les nouveaux cas ou décès et le sauvegarde.

        Args:
            df (pd.DataFrame): DataFrame contenant les données historiques.
            target (str): Nom de la colonne cible pour la prédiction.
            look_back (int): Nombre de jours pour les features de décalage temporel.
            test_size (float): Proportion des données à utiliser pour le test.

        Returns:
            model (xgboost.XGBRegressor): Modèle entraîné.
            metrics (dict): Dictionnaire contenant les métriques d'évaluation du modèle.
        """
        # Création des features
        X = df[[col for col in df.columns if col.startswith('lag_') or 
                col.startswith('rolling_') or 
                col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]]
        y = df[target]
        
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Configuration du modèle
        """
        Paramètres du modèle XGBoost :
        - objective: 'reg:squarederror' pour la régression, elle minimise l'erreur quadratique moyenne entre les prédictions et les valeurs réelles.
        - n_estimators: 1000, le nombre d'arbres à construire.
        - learning_rate: 0.05, le taux d'apprentissage, qui contrôle la contribution de chaque arbre à la prédiction finale.
        - max_depth: 6, la profondeur maximale des arbres, qui contrôle la complexité du modèle. (trop profond = surapprentissage)
        - subsample: 0.9, la fraction d'échantillons à utiliser pour chaque arbre, ce qui aide à éviter le surapprentissage. (on prend pas tous les échantillons seulement 90%)
        - colsample_bytree: 0.8, la fraction de caractéristiques à utiliser pour chaque arbre, ce qui aide également à éviter le surapprentissage.
        - early_stopping_rounds: 50, le nombre d'itérations sans amélioration avant d'arrêter l'entraînement, ce qui aide à éviter le surapprentissage.
        - random_state: 42, pour la reproductibilité des résultats. (surtout pour le débug)

        """
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42
        )
        
        # Entraînement
        logger.info("Début de l'entraînement du modèle XGBoost.")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        # Évaluation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        logger.info(f"Modèle entraîné pour {target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Sauvegarde du modèle
        model_path = f"{self.model_dir}/{target}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")
        
        return model, {'MAE': mae, 'RMSE': rmse}
    
    def load_model(self, target: str):
        """
        Charge un modèle sauvegardé pour une cible donnée.

        Args:
            - target (str): Nom de la colonne cible pour la prédiction.

        Returns:
            - model (xgboost.XGBRegressor): Modèle chargé.
        """
        model_path = f"{self.model_dir}/{target}_model.pkl"
        try:
            model = joblib.load(model_path)
            logger.info(f"Modèle chargé depuis {model_path}")
            return model
        except FileNotFoundError:
            logger.warning(f"Aucun modèle trouvé pour {target} dans {model_path}.")
            return None

    def predict_future(self, df: pd.DataFrame, target: str, days_ahead: int = 7, look_back: int = 30):
        """
        Prédit les valeurs futures pour la cible spécifiée.

        Args:
            - df (pd.DataFrame): DataFrame contenant les données historiques.
            - target (str): Nom de la colonne cible pour la prédiction.
            - days_ahead (int): Nombre de jours à prédire dans le futur.
            - look_back (int): Nombre de jours pour les features de décalage temporel.

        Returns:
            - predictions (pd.DataFrame): DataFrame contenant les dates et les valeurs prédites.
        """
        # Charger un modèle existant ou entraîner un nouveau modèle
        model = self.load_model(target)
        if model is None:
            logger.info(f"Entraînement d'un nouveau modèle pour {target}.")
            model, _ = self.train_model(df, target, look_back)
        
        # Préparer les données pour la prédiction
        last_data = df.iloc[-1:]
        X_pred = last_data[[col for col in df.columns if col.startswith('lag_') or 
                            col.startswith('rolling_') or 
                            col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]]
        
        # Générer les prédictions
        predictions = []
        current_date = df.index.max()
        
        for i in range(1, days_ahead + 1):
            pred = model.predict(X_pred)[0]
            predictions.append({
                'date': current_date + timedelta(days=i),
                'predicted_' + target: pred
            })
        
        return pd.DataFrame(predictions).set_index('date')