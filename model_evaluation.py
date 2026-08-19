from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\nModel Performance")
    print("-"*40)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")