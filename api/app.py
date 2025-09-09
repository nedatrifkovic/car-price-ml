from pathlib import Path
from typing import List, Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import ModelBundle

# Initialize FastAPI app
app = FastAPI(title="Car Price API")

# Load model and feature config at startup
MODEL_PATH = Path("models/car_price_model_XGBoost_Tuned.pkl")
FEATURE_CONFIG_PATH = Path("models/feature_config.json")
model_bundle = ModelBundle(
    model_path=MODEL_PATH, 
    config_path=FEATURE_CONFIG_PATH
)

# Pydantic model for input validation


class CarFeatures(BaseModel):
    manufacturer: Optional[str]
    model: Optional[str]
    prod_year: Optional[int]
    category: Optional[str]
    leather_interior: Optional[str]
    fuel_type: Optional[str]
    engine_volume: Optional[float]
    mileage: Optional[int]
    cylinders: Optional[float]
    gear_box_type: Optional[str]
    drive_wheels: Optional[str]
    doors: Optional[int]
    wheel: Optional[str]
    color: Optional[str]
    airbags: Optional[int]
    is_Turbo: Optional[str]


@app.get("/")
async def root():
    """Root endpoint to confirm model path."""
    return {"status": "ok", "model_path": str(MODEL_PATH.resolve())}


@app.post("/predict")
async def predict(features: List[CarFeatures]):
    """
    Accepts one or multiple CarFeatures objects and returns predictions.
    """
    # Convert to list of dicts if only one input is provided
    if isinstance(features, CarFeatures):
        data = [features.dict()]
    else:
        data = [f.dict() for f in features]

    preds = model_bundle.predict(data)
    # Return predictions as a list of floats
    return {"predictions": [float(p) for p in preds]}
