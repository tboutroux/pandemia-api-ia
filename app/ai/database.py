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


def load_data(engine, country_name: str) -> pd.DataFrame:
    """
    Fonction pour charger les données d'un pays spécifique depuis la base de données.

    Args:
        engine (sqlalchemy.engine.base.Engine): Moteur SQLAlchemy pour la connexion à la base de données.
        country_name (str): Nom du pays pour lequel charger les données.

    Returns:
        df (pd.DataFrame): DataFrame contenant les données du pays spécifié.
    """
    query = """
    SELECT 
        gd.date, 
        gd.new_cases, 
        gd.new_deaths,
        c.population
    FROM Global_Data gd
    JOIN Country c ON gd.country_id = c.id
    WHERE c.name = %s
    AND gd.date IS NOT NULL
    AND gd.new_cases IS NOT NULL
    AND gd.new_deaths IS NOT NULL
    ORDER BY gd.date
    """
    try:
        logger.debug(f"Exécution de la requête SQL pour {country_name}.")
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