# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction


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
    return {"status": "ok"}
