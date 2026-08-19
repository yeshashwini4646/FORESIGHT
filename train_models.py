import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

from pathlib import Path

from config import (
    FORECAST_DATA_PATH,
    FORECAST_FEATURES,
    FORECAST_RESULTS_PATH,
    MODEL_PATH,
    MODEL_RESULTS_PATH,
)


def _build_models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def train_models(
    forecast_data_path: Path = FORECAST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    results_path: Path = MODEL_RESULTS_PATH,
    predictions_path: Path = FORECAST_RESULTS_PATH,
    train_fraction: float = 0.8,
):
    """Train time-ordered models and persist the best test-set model."""
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be between 0.5 and 1")

    df = pd.read_csv(forecast_data_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    missing_features = set(FORECAST_FEATURES).difference(df.columns)
    if missing_features:
        raise ValueError(
            "Forecast data is missing features: "
            + ", ".join(sorted(missing_features))
        )

    split = int(len(df) * train_fraction)
    if split <= 0 or split >= len(df):
        raise ValueError("Forecast data must contain rows in both train and test sets")

    X = df[FORECAST_FEATURES]
    y = df["Quantity"]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    results = []
    predictions = pd.DataFrame(
        {
            "Date": df.loc[split:, "Date"].to_numpy(),
            "Actual_Quantity": y_test.to_numpy(),
        }
    )

    # A lag-7 forecast is a transparent benchmark for the ML models.
    baseline_prediction = X_test["Lag_7"].to_numpy()
    predictions["Baseline_Predicted_Quantity"] = baseline_prediction
    results.append(
        {
            "Model": "Lag7Baseline",
            "MAE": mean_absolute_error(y_test, baseline_prediction),
            "RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
        }
    )

    best_model = None
    best_name = None
    best_rmse = float("inf")

    for name, model in _build_models().items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mae = mean_absolute_error(y_test, prediction)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        predictions[f"{name}_Predicted_Quantity"] = prediction
        results.append({"Model": name, "MAE": mae, "RMSE": rmse})

        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse

    assert best_model is not None and best_name is not None
    predictions["Predicted_Quantity"] = predictions[
        f"{best_name}_Predicted_Quantity"
    ]
    predictions["Best_Model"] = best_name

    for path in (model_path, results_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    pd.DataFrame(results).sort_values("RMSE").to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    return pd.DataFrame(results), predictions, best_name


if __name__ == "__main__":
    results, _, best_name = train_models()
    print(results.to_string(index=False))
    print(f"Best model saved: {best_name}")