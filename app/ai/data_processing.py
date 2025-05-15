import pandas as pd

def create_features(df: pd.DataFrame, target: str, look_back: int = 30) -> pd.DataFrame:
    """
    Fonction pour créer des features à partir des données historiques.

    Args:
        df (pd.DataFrame): DataFrame contenant les données historiques.
        target (str): Nom de la colonne cible pour la prédiction.
        look_back (int): Nombre de jours pour les features de décalage temporel.

    Returns:
        df (pd.DataFrame): DataFrame avec les nouvelles features ajoutées.

    Explanation:
        - Une moyenne mobile est une technique de lissage qui calcule la moyenne d'un ensemble de valeurs sur une période donnée pour éviter le bruit.
            - exemple : Au lieu de prendre la valeur brute d'un jour, on prend une moyenne de plusieurs jours pour avoir un aperçu plus clair de la tendance.
        - Lag features : De base le modèle ne peut pas prédire la valeur d'un jour en fonction de lui-même, il faut donc créer des features qui prennent en compte les jours précédents.
    """
    # Features de décalage temporel
    for i in range(1, look_back + 1):
        df[f'lag_{i}'] = df[target].shift(i)
    
    # Moyennes mobiles
    df['rolling_7_mean'] = df[target].rolling(7).mean()
    df['rolling_30_mean'] = df[target].rolling(30).mean()
    
    # Features temporelles
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    
    # Ratio cas/population
    if 'population' in df.columns:
        df['cases_per_100k'] = (df['new_cases'] / (df['population'] / 100000)).fillna(0)
        df['deaths_per_100k'] = (df['new_deaths'] / (df['population'] / 100000)).fillna(0)
    
    # Suppression des lignes avec valeurs manquantes
    df = df.dropna()
    return df