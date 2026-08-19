from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import FORECAST_FEATURES, MODEL_PATH


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Lag_1: float = Field(ge=0)
    Lag_7: float = Field(ge=0)
    Rolling_7: float = Field(ge=0)
    DayOfWeek: int = Field(ge=0, le=6)
    Month: int = Field(ge=1, le=12)
    Quarter: int = Field(ge=1, le=4)
    Year: int = Field(ge=2000, le=2100)


class PredictionResponse(BaseModel):
    predicted_quantity: float
    model_path: str


app = FastAPI(
    title="Project FORESIGHT Prediction API",
    description="Demand prediction service backed by the saved best forecasting model.",
    version="1.0.0",
)


def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Run `python src/train_models.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    try:
        load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = load_model()
        features = pd.DataFrame(
            [[getattr(request, feature) for feature in FORECAST_FEATURES]],
            columns=FORECAST_FEATURES,
        )
        prediction = float(model.predict(features)[0])
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    return PredictionResponse(
        predicted_quantity=max(0.0, prediction),
        model_path=str(MODEL_PATH),
    )
