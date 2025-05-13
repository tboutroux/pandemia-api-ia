from fastapi import APIRouter


router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
)


@router.get("/new")
def get_prediction():
    return {"message": "This is the prediction endpoint."}


@router.post("/")
def post_prediction(data: dict):
    # Here you would typically process the data and return a prediction
    # For demonstration, we'll just echo the data back
    return {"received_data": data}


@router.put("/")
def update_prediction(data: dict):
    # Here you would typically process the data and return a prediction
    # For demonstration, we'll just echo the data back
    return {"received_data": data}


@router.delete("/")
def delete_prediction(prediction_id: int):
    # Here you would typically process the data and return a prediction
    # For demonstration, we'll just echo the data back
    return {"deleted_prediction_id": prediction_id}
