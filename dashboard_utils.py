from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DAILY_SALES_PATH,
    FORECAST_DATA_PATH,
    FORECAST_FEATURES,
    MODEL_PATH,
    MODEL_RESULTS_PATH,
    PROCESSED_DATA_PATH,
    RISK_RESULTS_PATH,
)

RISK_COLORS = {
    "Healthy": "#42d392",
    "Monitor": "#f5c451",
    "Overstock": "#f59e68",
    "Overstock Risk": "#f59e68",
    "Stockout Risk": "#ef6b73",
}


def _read_csv(path, **kwargs):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_daily_sales():
    df = _read_csv(DAILY_SALES_PATH, parse_dates=["Date"])
    return df.sort_values("Date") if not df.empty else df


@st.cache_data(show_spinner=False)
def load_forecast_data():
    df = _read_csv(FORECAST_DATA_PATH, parse_dates=["Date"])
    return df.sort_values("Date") if not df.empty else df


@st.cache_data(show_spinner=False)
def load_retail_data():
    df = _read_csv(PROCESSED_DATA_PATH, parse_dates=["InvoiceDate"])
    return df


@st.cache_data(show_spinner=False)
def load_risk_data():
    df = _read_csv(RISK_RESULTS_PATH, parse_dates=["Date"])
    return df.sort_values("Date") if not df.empty else df


@st.cache_data(show_spinner=False)
def load_model_results():
    return _read_csv(MODEL_RESULTS_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)


def model_predictions(df):
    if df.empty or load_model() is None:
        return df.copy()
    result = df.copy()
    result["Predicted_Quantity"] = np.maximum(
        0, load_model().predict(result[FORECAST_FEATURES])
    )
    return result


def calculate_forecast_metrics(df):
    if df.empty or "Predicted_Quantity" not in df:
        return {"mae": None, "rmse": None, "mape": None, "accuracy": None}
    actual = df["Quantity"].astype(float)
    predicted = df["Predicted_Quantity"].astype(float)
    non_zero = actual != 0
    mape = (
        (np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])).mean()
        * 100
        if non_zero.any()
        else None
    )
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": max(0, 100 - mape) if mape is not None else None,
    }


def product_summary(retail, risk):
    if retail.empty:
        return pd.DataFrame()
    grouped = retail.groupby(["StockCode", "Description"], dropna=False).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
        Average_Price=("Price", "mean"),
        Active_Days=("InvoiceDate", lambda value: value.dt.date.nunique()),
        Last_Sale=("InvoiceDate", "max"),
    ).reset_index()
    grouped["Average_Daily_Demand"] = grouped["Units_Sold"] / grouped[
        "Active_Days"
    ].replace(0, np.nan)
    if not risk.empty:
        latest_risk = risk.sort_values("Date").iloc[-1]
        grouped["Estimated_Inventory"] = grouped["Average_Daily_Demand"] * 1.5
        grouped["Forecast_Demand"] = grouped["Average_Daily_Demand"]
        grouped["Inventory_Status"] = np.select(
            [
                grouped["Estimated_Inventory"] < grouped["Forecast_Demand"],
                grouped["Estimated_Inventory"] >= grouped["Forecast_Demand"] * 2,
            ],
            ["Stockout Risk", "Overstock"],
            default="Healthy",
        )
        grouped["Risk_Score"] = np.clip(
            50
            + (grouped["Forecast_Demand"] - grouped["Estimated_Inventory"])
            / grouped["Forecast_Demand"].replace(0, np.nan)
            * 50,
            0,
            100,
        ).fillna(0)
        grouped["Recommended_Action"] = grouped["Inventory_Status"].map(
            {
                "Stockout Risk": "Initiate replenishment",
                "Healthy": "Maintain current inventory",
                "Overstock": "Reduce replenishment or promote",
            }
        )
        _ = latest_risk
    return grouped.sort_values("Revenue", ascending=False)


def format_number(value, decimals=0):
    if value is None or pd.isna(value):
        return "--"
    return f"{value:,.{decimals}f}"


def format_currency(value):
    return f"${format_number(value)}"


def apply_dark_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#aeb8c9"},
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return fig
