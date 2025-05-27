from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
import logging
import numpy as np
import joblib
import os
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

class PandemicModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.best_params = None

    def _get_default_model(self):
        """Retourne un modèle XGBoost avec des paramètres par défaut."""
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42
        )

    def _time_series_split(self, X, y, n_splits=5):
        """Effectue une validation croisée temporelle."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in tscv.split(X):
            yield X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    def train_model(self, df, target, feature_names, test_size=0.2, tune_hyperparams=False):
        """
        Entraîne un modèle XGBoost pour prédire les nouveaux cas ou décès.

        Args:
            df: DataFrame contenant les données d'entraînement.
            target: Nom de la colonne cible à prédire.
            feature_names: Liste des noms de colonnes à utiliser comme features.
            test_size: Proportion des données à utiliser pour le test.
            tune_hyperparams: Si True, effectue un tuning des hyperparamètres.

        Returns:
            model: Modèle entraîné.
            metrics: Dictionnaire contenant les métriques d'évaluation.
        """
        X = df[feature_names]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        if tune_hyperparams:
            logger.info("Début du tuning des hyperparamètres...")
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                estimator=self._get_default_model(),
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_params = grid_search.best_params_
            model = grid_search.best_estimator_
            logger.info(f"Meilleurs paramètres trouvés: {self.best_params}")
        else:
            model = self._get_default_model()
            logger.info("Début de l'entraînement du modèle avec paramètres par défaut.")
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=10
            )

        preds = model.predict(X_test)
        metrics = {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'R2': r2_score(y_test, preds)
        }
        logger.info(f"Modèle entraîné pour {target} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.2f}")
        return model, metrics

    def save_model(self, model, target):
        """Sauvegarde le modèle entraîné."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"Répertoire {self.model_dir} créé.")
        model_path = f"{self.model_dir}/{target}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")

    def load_model(self, target):
        """Charge le modèle XGBoost sauvegardé."""
        model_path = f"{self.model_dir}/{target}_model.pkl"
        try:
            model = joblib.load(model_path)
            logger.info(f"Modèle chargé depuis {model_path}")
            return model
        except FileNotFoundError:
            logger.warning(f"Aucun modèle trouvé pour {target} dans {model_path}.")
            return None

    def predict_future(self, df, target, feature_names, days_ahead=7, look_back=30):
        """Prédit les valeurs futures pour la cible spécifiée."""
        model = self.load_model(target)
        if model is None:
            logger.info(f"Entraînement d'un nouveau modèle pour {target}.")
            model, _ = self.train_model(df, target, feature_names)
            self.save_model(model, target)
        
        predictions = []
        current_date = df.index.max()
        last_data = df.iloc[-1:].copy()
        
        for i in range(1, days_ahead + 1):
            current_date = current_date + timedelta(days=1)
            X_pred = last_data[feature_names]
            pred = model.predict(X_pred)[0]
            predictions.append({
                'date': current_date,
                f'predicted_{target}': pred
            })
            
            # Mise à jour des features pour la prochaine prédiction
            new_row = last_data.copy()
            new_row[target] = pred
            for lag in range(look_back, 1, -1):
                new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}']
            if 'lag_1' in new_row.columns:
                new_row['lag_1'] = pred
            if 'rolling_7_mean' in new_row.columns:
                vals = list(last_data[target].values[-6:]) + [pred]
                new_row['rolling_7_mean'] = np.mean(vals)
            if 'rolling_30_mean' in new_row.columns:
                vals = list(last_data[target].values[-29:]) + [pred]
                new_row['rolling_30_mean'] = np.mean(vals)
            
            new_row.index = [current_date]
            last_data = pd.concat([last_data, new_row]).iloc[1:]
        
        return pd.DataFrame(predictions).set_index('date')

    def predict_multiple_targets(self, df, targets=None, feature_names=None, days_ahead=7, look_back=30):
        """
        Prédit les valeurs futures pour plusieurs cibles.
        
        Args:
            df: DataFrame contenant les données historiques.
            targets: Liste des cibles à prédire (par défaut ['new_cases', 'new_deaths', 'new_recovered']).
            feature_names: Liste des noms de colonnes à utiliser comme features.
            days_ahead: Nombre de jours à prédire.
            look_back: Nombre de jours pour les features de décalage temporel.
            
        Returns:
            Dictionnaire contenant les prédictions pour chaque cible.
        """
        if targets is None:
            targets = ['new_cases', 'new_deaths', 'new_recovered']
            
        predictions = {}
        for target in targets:
            predictions[target] = self.predict_future(df, target, feature_names, days_ahead, look_back)
        
        return predictions
