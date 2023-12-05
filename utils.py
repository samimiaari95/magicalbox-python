import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
import math
import matplotlib.pyplot as plt


def fit_powerlaw(x_axis, y_axis):
    def powerlaw(h, a, b):
        y = a*(h**b)
        #y = (a)*np.exp(-b*h)
        return y
    
    params, covariance = curve_fit(powerlaw, x_axis, y_axis)
    # Extract the fitted parameters
    a_fit, b_fit = params

    # Print the results
    print(f"Fitted a: {a_fit} and b:{b_fit}")

    y_fit = [powerlaw(x, a_fit, b_fit) for x in x_axis]

    R_square = r2_score(y_axis, y_fit)
    print(f"R2 = {R_square}")
    #return R_square
    MSE = np.square(np.subtract(y_axis,y_fit)).mean()
    print(f"MSE = {MSE*10**-9} x10⁹")
    RMSE = math.sqrt(MSE)
    print(f"RMSE = {RMSE}")
    MAE = mean_absolute_error(y_axis, y_fit)
    print(f"MAE = {MAE}")
    
    x_fit = [min(x_axis), max(x_axis)]
    y_fit = [powerlaw(x, a_fit, b_fit) for x in x_fit]

    return x_fit, y_fit

def plotlog_show(x_axis, y_axis, color="k", linestyle="-", label=None, xlabel="x", ylabel="y", scatter=True):
    plt.figure(figsize=(16,9))
    plt.grid(True)

    if label and scatter:
        plt.scatter(x_axis, y_axis, color=f"{color}{linestyle}", label=label)
    elif not label and scatter:
        plt.scatter(x_axis, y_axis, color=f"{color}{linestyle}")
    elif label and not scatter:
        plt.plot(x_axis, y_axis, color=f"{color}{linestyle}", label=label)
    elif not label and not scatter:
        plt.plot(x_axis, y_axis, color=f"{color}{linestyle}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.show()

def powerlaw(h, a, b):
        y = a*(h**b)
        return y