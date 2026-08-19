from data_loader import load_data
from data_cleaning import clean_data
from feature_engineering import create_features
from forecasting import (
    prepare_daily_sales,
    create_forecasting_features
)
from config import (
    DAILY_SALES_PATH,
    FORECAST_DATA_PATH,
    PROCESSED_DATA_PATH,
    RISK_RESULTS_PATH,
)
from risk_scoring import calculate_risk
from train_models import train_models


def main():

    print("=" * 50)
    print("PROJECT FORESIGHT - DATA PIPELINE")
    print("=" * 50)


    # Step 1: Load Data
    print("\nLoading dataset...")

    df = load_data()

    print("Original Shape:", df.shape)



    # Step 2: Data Cleaning
    print("\nCleaning dataset...")

    df = clean_data(df)

    print("Cleaned Shape:", df.shape)



    # Step 3: Feature Engineering
    print("\nCreating features...")

    df = create_features(df)


    # Save cleaned dataset
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Cleaned dataset saved!")



    # Step 4: Daily Sales Creation
    print("\nPreparing daily sales...")

    daily_sales = prepare_daily_sales(df)


    daily_sales.to_csv(DAILY_SALES_PATH, index=False)

    print("Daily sales saved!")



    # Step 5: Forecast Feature Engineering
    print("\nCreating forecasting features...")

    forecast_data = create_forecasting_features(
        daily_sales
    )


    forecast_data.to_csv(FORECAST_DATA_PATH, index=False)

    print("Forecast dataset saved!")



    # Step 6: Train and compare forecasting models
    print("\nTraining forecasting models...")
    results, _, best_name = train_models()
    print(f"Best model: {best_name}")
    print(results.to_string(index=False))



    # Step 7: Risk Scoring
    print("\nCalculating inventory risk...")

    risk_data = calculate_risk(
        forecast_data
    )


    RISK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    risk_data.to_csv(RISK_RESULTS_PATH, index=False)

    print("Risk analysis saved!")



    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)



if __name__ == "__main__":
    main()