import matplotlib.pyplot as plt

def plot_data_overview(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["I_allsky"], label="I_allsky (with clouds)", marker='o')
    plt.plot(df.index, df["I_clearsky"], label="I_clearsky (clear sky)", marker='s')
    plt.plot(df.index, df["Cloud_Amount"], label="Cloud_Amount (normalized)", marker='^')
    plt.title("Original Monthly Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_data_overview.png")

def plot_exp_decay_prediction(train, test, test_pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train["I_allsky"], label="Train I_allsky", marker='o', linestyle='-', color='steelblue')
    plt.plot(test.index, test["I_allsky"], label="Test Actual I_allsky", marker='s', linestyle='-', color='chocolate')
    plt.plot(test.index, test_pred, label="Exp Decay Prediction", marker='x', linestyle='--', color='green')
    plt.title("Exponential Decay Model: Prediction")
    plt.xlabel("Time")
    plt.ylabel("Irradiance (W/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_exp_decay.png")

def plot_sarimax_predictions(train, test, forecast_R, pred_I):
    import matplotlib.pyplot as plt
    # R prediction
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train["R"], label="Train R", marker='o', linestyle='-', color='gray')
    plt.plot(test.index, test["R"], label="Test Actual R", marker='s', linestyle='-', color='chocolate')
    plt.plot(test.index, forecast_R, label="Predicted R", marker='D', linestyle='--', color='green')
    plt.title("SARIMAX Model: Prediction of Cloud Factor R")
    plt.xlabel("Time")
    plt.ylabel("R = I_allsky / I_clearsky")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_sarimax_predict_R.png")

    # I_allsky prediction
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train["I_allsky"], label="Train I_allsky", marker='o', linestyle='-', color='steelblue')
    plt.plot(test.index, test["I_allsky"], label="Test Actual I_allsky", marker='s', linestyle='-', color='chocolate')
    plt.plot(test.index, pred_I, label="Predicted I_allsky", marker='D', linestyle='--', color='green')
    plt.title("SARIMAX Model: I_allsky Prediction via R")
    plt.xlabel("Time")
    plt.ylabel("Irradiance (W/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_sarimax_predict_I.png")

def plot_lstm_prediction(train, aligned_index, y_true, y_pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train["I_allsky"], label="Train I_allsky", marker='o', linestyle='-', color='steelblue')
    plt.plot(aligned_index, y_true, label="Test Actual I_allsky", marker='s', linestyle='-', color='chocolate')
    plt.plot(aligned_index, y_pred, label="LSTM Prediction", marker='^', linestyle='--', color='purple')
    plt.title("LSTM Model: Test Set Prediction")
    plt.xlabel("Time")
    plt.ylabel("Irradiance (W/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_lstm_prediction.png")

def plot_final_comparison(test, exp_pred, sarimax_pred, lstm_pred, y_true_lstm):
    aligned_index = test.index[-len(lstm_pred):]
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, test["I_allsky"], label="Test Actual", marker='o', linestyle='-', color='steelblue')
    plt.plot(test.index, exp_pred, label="Exp Decay", marker='x', linestyle='--', color='orange')
    plt.plot(test.index, sarimax_pred, label="SARIMAX", marker='D', linestyle='-.', color='green')
    plt.plot(aligned_index, lstm_pred, label="LSTM", marker='^', linestyle=':', color='purple')
    plt.title("Comparison: Exponential Decay vs. SARIMAX vs. LSTM (Test Set)")
    plt.xlabel("Time")
    plt.ylabel("Irradiance (W/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig_comparison_final.png")
