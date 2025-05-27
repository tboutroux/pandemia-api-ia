import pandas as pd

def _create_lag_features(df: pd.DataFrame, target: str, look_back: int) -> pd.DataFrame:
    """Crée les features de décalage temporel."""
    for i in range(1, look_back + 1):
        df[f'lag_{i}'] = df[target].shift(i)
    return df

def _create_rolling_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Crée les moyennes mobiles."""
    df['rolling_7_mean'] = df[target].rolling(7).mean()
    df['rolling_30_mean'] = df[target].rolling(30).mean()
    return df

def _create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features temporelles."""
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    return df

def _create_population_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les ratios basés sur la population."""
    if 'population' in df.columns:
        if 'new_cases' in df.columns:
            df['cases_per_100k'] = (pd.to_numeric(df['new_cases'], errors='coerce') / (pd.to_numeric(df['population'], errors='coerce') / 100000)).fillna(0)
        if 'new_deaths' in df.columns:
            df['deaths_per_100k'] = (pd.to_numeric(df['new_deaths'], errors='coerce') / (pd.to_numeric(df['population'], errors='coerce') / 100000)).fillna(0)
        if 'new_recovered' in df.columns:
            df['recovered_per_100k'] = (pd.to_numeric(df['new_recovered'], errors='coerce') / (pd.to_numeric(df['population'], errors='coerce') / 100000)).fillna(0)
    return df

def create_features(df: pd.DataFrame, target: str, look_back: int = 30,use_lags: bool = True, use_rolling: bool = True, use_calendar: bool = True) -> pd.DataFrame:
    """
    Fonction pour créer des features à partir des données historiques.

    Args:
        df: DataFrame contenant les données historiques.
        target: Nom de la colonne cible à prédire.
        look_back: Nombre de jours pour les features de décalage temporel.
        use_lags: Si les features de décalage temporel doivent être créées.
        use_rolling: Si les moyennes mobiles doivent être créées.
        use_calendar: Si les features temporelles doivent être créées.

    Returns:
        DataFrame avec les nouvelles features ajoutées.
    """
    # Filtre dynamique des colonnes existantes
    feature_cols = []
    if use_lags:
        feature_cols += [f'lag_{i}' for i in range(1, look_back + 1)]
    if use_rolling:
        feature_cols += ['rolling_7_mean', 'rolling_30_mean']
    if use_calendar:
        feature_cols += ['day_of_week', 'day_of_month', 'month']
    if 'population' in df.columns:
        if 'new_cases' in df.columns:
            feature_cols.append('cases_per_100k')
        if 'new_deaths' in df.columns:
            feature_cols.append('deaths_per_100k')
        if 'new_recovered' in df.columns:
            feature_cols.append('recovered_per_100k')
    
    # Ne garder que les colonnes non-features et ajouter les nouvelles
    existing_cols = [col for col in df.columns if col not in feature_cols]
    df = df[existing_cols].copy()
    
    # Application des transformations
    if use_lags:
        df = _create_lag_features(df, target, look_back)
    if use_rolling:
        df = _create_rolling_features(df, target)
    if use_calendar:
        df = _create_calendar_features(df)
    if 'population' in df.columns:
        df = _create_population_features(df)

    # Suppression des lignes avec valeurs manquantes
    cols_to_check = [col for col in [target] + feature_cols if col in df.columns]
    return df.dropna(subset=cols_to_check)