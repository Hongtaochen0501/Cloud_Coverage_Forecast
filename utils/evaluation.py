from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{model_name}: MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
    return mse, mae, mape
