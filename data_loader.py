import pandas as pd
from config import RAW_DATA_PATH

def load_data():
    print("Loading dataset...")
    df = pd.read_excel(RAW_DATA_PATH)
    print("Dataset Loaded Successfully!")
    return df