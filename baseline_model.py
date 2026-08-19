from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_baseline_model(df):

    # Features
    X = df[[
        "Lag_1",
        "Lag_7",
        "Rolling_7",
        "DayOfWeek",
        "Month",
        "Quarter",
        "Year"
    ]]

    # Target
    y = df["Quantity"]

    # Time-based split (80% train, 20% test)
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("=" * 40)
    print("Baseline Random Forest Results")
    print("=" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    return model