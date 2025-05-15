import matplotlib.pyplot as plt
import os

def plot_predictions(df, preds, target, country_name, output_dir="visualization"):
    """
    Enregistre un graphique comparant les données historiques et les prédictions dans un fichier.

    Args:
        - df (pd.DataFrame): DataFrame contenant les données historiques.
        - preds (pd.DataFrame): DataFrame contenant les prédictions.
        - target (str): Nom de la colonne cible pour la prédiction.
        - country_name (str): Nom du pays pour lequel générer le graphique.
        - output_dir (str): Répertoire de sortie pour enregistrer le graphique.

    Returns:
        None
    """
    # Vérification et création du répertoire `visualization/` si nécessaire
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Données historiques', color='blue')
    plt.plot(preds.index, preds[f'predicted_{target}'], label='Prédictions', color='red', linestyle='--')
    plt.title(f"Prédiction des {target} pour {country_name}")
    plt.xlabel('Date')
    plt.ylabel(f"Nombre de {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Enregistrement du graphique
    output_path = os.path.join(output_dir, f"{country_name}_{target}_predictions.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé dans {output_path}")