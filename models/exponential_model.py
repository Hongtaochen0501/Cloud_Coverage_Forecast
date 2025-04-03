import numpy as np
from scipy.optimize import curve_fit

def run_exponential_model(train, test):
    def exp_decay_model(C, k, I_clear):
        return I_clear * np.exp(-k * C)

    popt, _ = curve_fit(lambda C, k: exp_decay_model(C, k, train["I_clearsky"].values),
                        train["Cloud_Amount"].values, train["I_allsky"].values, p0=[0.01])
    k_opt = popt[0]
    pred = test["I_clearsky"].values * np.exp(-k_opt * test["Cloud_Amount"].values)
    return pred, k_opt
