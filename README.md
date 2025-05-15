# API IA Pandemia
Le projet contient une API permettant de faire de la prédiction sur les pandémies.

## Installation

1. Cloner le repo Github
```sh
git clone git@github.com:tboutroux/pandemia-api-ia.git
```

2. Créer un venv
```bash
python3 -m venv venv
```

3. Installer les dépendances
```bash
source venv/bin/activate # Linux
venv\Scripts\activate # Windows

pip install -r requirements.txt
```

## Architecture du projet 
```
└── tboutroux/pandemia-api-ia/
    ├── README.md
    ├── requirements.txt
    ├── .env
    ├── app/
    │   ├── main.py
    │   ├── routers/
    │   │   ├── prediction.py
    |   ├── ai/
    │   │   ├── data_processing.py
    │   │   ├── database.py
    │   │   ├── model.py
    │   │   ├── vizualisation.py

```

## Exemple de .env
```
DB_HOST=your_db_host
DB_USER=your_db_user
DB_PASSWORD=your_user_password
DB_NAME=your_db_name
```