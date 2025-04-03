from utils.preprocessing import load_and_split_data
from models.exponential_model import run_exponential_model
from models.sarimax_model import run_sarimax_model
from models.lstm_model import run_lstm_model
from utils.evaluation import compute_metrics
from utils.visualization import plot_final_comparison
from utils.visualization import (
    plot_data_overview,
    plot_final_comparison,
    plot_exp_decay_prediction,
    plot_sarimax_predictions,
    plot_lstm_prediction
)

# 1. Load data
train, test, df = load_and_split_data("data/Data.csv")

# 2. Visualize raw data
plot_data_overview(df)

# 3. Exponential Decay Model
pred_exp, k = run_exponential_model(train, test)
compute_metrics(test["I_allsky"].values, pred_exp, model_name="Exponential Decay")
plot_exp_decay_prediction(train, test, pred_exp)

# 4. SARIMAX Model
pred_sarimax, forecast_R = run_sarimax_model(train, test)
compute_metrics(test["I_allsky"].values, pred_sarimax.values, model_name="SARIMAX")
plot_sarimax_predictions(train, test, forecast_R, pred_sarimax)

# 5. LSTM Model
pred_lstm, y_true_lstm, aligned_index = run_lstm_model(df)
compute_metrics(y_true_lstm, pred_lstm, model_name="LSTM")
plot_lstm_prediction(train, aligned_index, y_true_lstm, pred_lstm)

# 6. Plot final comparison
plot_final_comparison(test, pred_exp, pred_sarimax, pred_lstm, y_true_lstm)
