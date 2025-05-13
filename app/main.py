# app/main.py
from fastapi import FastAPI
from app.routers import prediction


app = FastAPI(title="Pandemia API IA", version="1.0.0")

app.include_router(prediction.router)


# Je veux que la route / redirige vers le swagger
@app.get("/")
def root():
    return {"message": "Welcome to the Pandemia API IA. Visit /docs for the Swagger UI."}


@app.get("/health")
def health_check():
    return {"status": "ok"}
