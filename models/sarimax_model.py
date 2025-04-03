from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarimax_model(train, test):
    model = SARIMAX(train["R"], exog=train[["Cloud_Amount"]],
                    order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    forecast_R = fit.predict(start=test.index[0], end=test.index[-1], exog=test[["Cloud_Amount"]])
    predicted_I = test["I_clearsky"] * forecast_R
    return predicted_I, forecast_R
