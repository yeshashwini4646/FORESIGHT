import pandas as pd


def prepare_daily_sales(df):
    """
    Create daily aggregated sales.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    daily_sales = (
        df.groupby(df["InvoiceDate"].dt.date)
        .agg({
            "Quantity": "sum",
            "Revenue": "sum"
        })
        .reset_index()
    )

    daily_sales.rename(columns={"InvoiceDate": "Date"}, inplace=True)

    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
    daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)

    return daily_sales


def create_forecasting_features(daily_sales):

    daily_sales = daily_sales.copy()
    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
    daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)

    # Calendar Features
    daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek
    daily_sales["Month"] = daily_sales["Date"].dt.month
    daily_sales["Quarter"] = daily_sales["Date"].dt.quarter
    daily_sales["Year"] = daily_sales["Date"].dt.year

    # Lag Features
    daily_sales["Lag_1"] = daily_sales["Quantity"].shift(1)
    daily_sales["Lag_7"] = daily_sales["Quantity"].shift(7)

    # Rolling Mean
    daily_sales["Rolling_7"] = daily_sales["Quantity"].shift(1).rolling(7).mean()

    # Remove rows with NaN values created by lag features
    daily_sales = daily_sales.dropna()

    return daily_sales