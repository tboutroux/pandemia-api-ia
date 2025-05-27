import pandas as pd
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

def create_db_engine(user: str, password: str, host: str, database: str):
    """
    Fonctions pour interagir avec la base de données MySQL. (la bdd est mariadb mais le connecteur pymsql est compatible avec mariaDB)

    Args: 
        user (str): Nom d'utilisateur de la base de données.
        password (str): Mot de passe de la base de données.
        host (str): Adresse de l'hôte de la base de données.
        database (str): Nom de la base de données.

    Returns: 
        engine (sqlalchemy.engine.base.Engine): Moteur SQLAlchemy pour la connexion à la base de données.
    """
    connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
    return create_engine(connection_string)


def load_data(engine, country_name: str, targets: list = None) -> pd.DataFrame:
    """
    Fonction pour charger les données d'un pays spécifique depuis la base de données.

    Args:
        engine (sqlalchemy.engine.base.Engine): Moteur SQLAlchemy pour la connexion à la base de données.
        country_name (str): Nom du pays pour lequel charger les données.
        targets (list): Liste des colonnes à récupérer (new_cases, new_deaths, new_recovered)

    Returns:
        df (pd.DataFrame): DataFrame contenant les données du pays spécifié.
    """
    # Définir les targets par défaut si non spécifiées
    if targets is None:
        targets = ["new_cases", "new_deaths", "new_recovered"]
    
    # Vérifier que les targets sont valides
    valid_targets = ["new_cases", "new_deaths", "new_recovered"]
    for target in targets:
        if target not in valid_targets:
            raise ValueError(f"Target '{target}' invalide. Les targets valides sont: {valid_targets}")

    # Construire la requête dynamiquement
    select_columns = ["gd.date", "c.population"]
    
    if "new_cases" in targets:
        select_columns.append("gd.new_cases")
    if "new_deaths" in targets:
        select_columns.append("gd.new_deaths")
    if "new_recovered" in targets:
        select_columns.append("gd.new_recovered")
    
    query = f"""
    SELECT 
        {', '.join(select_columns)}
    FROM Global_Data gd
    JOIN Country c ON gd.country_id = c.id
    WHERE c.name = %s
    AND gd.date IS NOT NULL
    """
    
    # Ajouter les conditions NOT NULL pour les colonnes sélectionnées
    conditions = []
    if "new_cases" in targets:
        conditions.append("gd.new_cases IS NOT NULL")
    if "new_deaths" in targets:
        conditions.append("gd.new_deaths IS NOT NULL")
    if "new_recovered" in targets:
        conditions.append("gd.new_recovered IS NOT NULL")
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " ORDER BY gd.date"

    try:
        logger.debug(f"Exécution de la requête SQL pour {country_name} avec targets: {targets}")
        df = pd.read_sql(query, engine, params=(country_name,))
        if df.empty:
            logger.warning(f"Aucune donnée trouvée pour {country_name}.")
            raise ValueError(f"Aucune donnée trouvée pour {country_name}.")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        logger.info(f"Données chargées avec {len(df)} enregistrements pour {country_name}.")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise