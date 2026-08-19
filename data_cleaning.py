import pandas as pd

def clean_data(df):

    print("\nStarting Data Cleaning...")

    required_columns = {
        "Invoice",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    df = df.copy().drop_duplicates()

    # Online Retail II marks cancelled invoices with a leading C.
    invoice_text = df["Invoice"].astype(str).str.strip().str.upper()
    df = df.loc[~invoice_text.str.startswith("C")].copy()

    # Remove rows with missing Description
    df = df.dropna(subset=["Description"])

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])

    # Remove negative or zero Quantity
    df = df[df["Quantity"] > 0]

    # Remove zero or negative Price
    df = df[df["Price"] > 0]
    # Create Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    print("Cleaning Completed!")

    return df