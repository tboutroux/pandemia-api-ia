# app/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.security import verify_token
from fastapi.security import HTTPAuthorizationCredentials
from app.core.security import security



app = FastAPI(title="Pandemia API IA", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"],  # Autoriser toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

app.include_router(prediction.router)


# Je veux que la route / redirige vers le swagger
@app.get("/")
def root():
    return {"message": "Welcome to the Pandemia API IA. Visit /docs for the Swagger UI."}


@app.get("/health")
def health_check():
    """
    Route de vérification de l'état de l'API.

    - **returns**: Un dictionnaire indiquant que l'API est opérationnelle.
    - **example**:
    ```json
    {
        "status": "ok"
    }
    ```
    """
    return {"status": "ok"}


@app.get("/connect/test")
def test_connection(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Route pour tester la connexion à la base de données.
    - **credentials**: Clé API pour sécuriser l'accès à la route.
    - **returns**: Un message de succès si la connexion est établie.
    - **raises**: HTTPException en cas d'erreur de connexion ou de configuration de la base de données.
    """
    verify_token(credentials)

    return {"message": "Connexion réussie"}