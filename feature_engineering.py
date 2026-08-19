import pandas as pd

def create_features(df):

    # Convert InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Date Features
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Quarter"] = df["InvoiceDate"].dt.quarter
    df["Week"] = df["InvoiceDate"].dt.isocalendar().week

    # Weekend Indicator
    df["Weekend"] = df["InvoiceDate"].dt.dayofweek >= 5

    return df