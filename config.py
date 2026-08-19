from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_retail.csv"
DAILY_SALES_PATH = BASE_DIR / "data" / "processed" / "daily_sales.csv"
FORECAST_DATA_PATH = BASE_DIR / "data" / "processed" / "forecast_data.csv"
MODEL_PATH = BASE_DIR / "models" / "best_forecast_model.pkl"
MODEL_RESULTS_PATH = BASE_DIR / "outputs" / "tables" / "model_results.csv"
FORECAST_RESULTS_PATH = BASE_DIR / "outputs" / "tables" / "forecast_results.csv"
RISK_RESULTS_PATH = BASE_DIR / "outputs" / "tables" / "risk_analysis.csv"

FORECAST_FEATURES = [
	"Lag_1",
	"Lag_7",
	"Rolling_7",
	"DayOfWeek",
	"Month",
	"Quarter",
	"Year",
]