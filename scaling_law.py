from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
import math
import pandas as pd
import numpy as np
import netCDF4 as nc
import SLOTH.sloth.toolBox
import SLOTH.sloth.IO
import SLOTH.sloth.PlotLib
import SLOTH.sloth.mapper
import os
import re
import shutil
from datetime import date, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def qk_ktd():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math

    def powerlaw_fit(h, a, b):
        y = a*(h**b)
        return y
    plt.rcParams.update({'font.size': 22})
    filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv' # with watertable depth, n, alfa, theta, Ks from rosetta
    
    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "time"
    df = pd.read_csv(filepath)

    t_list = []
    x_axis = []
    y_axis = []

    for i in range(len(df)):
        q = df[q_column].iloc[i]
        k = df[k_column].iloc[i]
        d = df[d_column].iloc[i]
        n = df[n_column].iloc[i]
        alfa = df[alfa_column].iloc[i]
        theta_r = df[theta_r_column].iloc[i]
        theta_s = df[theta_s_column].iloc[i]
        t = df[t_column].iloc[i]
        if k>=q and t!=0 and (q/k)>=0.001:
            x = t*q/d
            #x = (alfa*d*q/k)
            #y = (k*k*t/(q*d*d*alfa))**(1-1/n) # best performance
            #y = (k*t/(d*d*alfa))**(1-1/n)
            #y = (q*q*t/(k*alfa*d*d))**(1-1/n)
            #y = (q*d*d*alfa/(k*k*t))**(1-1/n) # best fit
            y = q/k
            #y = (k*t/(d))#**(1-1/n)
            #y = alfa*d*t*k*alfa/(theta_s-theta_r)#*q/d)#**gamma

            #y_pred = 1.9886875606876062*(x**(-0.5468849399415959))
            #t_pred = (y_pred*y_pred*q*alfa*d*d)/(k*k)
            #t_list.append(abs(t_pred-t))
            y_axis.append(y)
            x_axis.append(x)
    
    params, covariance = curve_fit(powerlaw_fit, x_axis, y_axis)
    # Extract the fitted parameters
    a_fit, b_fit = params

    # Print the results
    print(f"Fitted a: {a_fit} and b:{b_fit}")
    # Generate data points for the fitted curve
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_axis]

    R_square = r2_score(y_axis, y_fit)
    print(f"R2 = {R_square}")
    MSE = np.square(np.subtract(y_axis,y_fit)).mean()
    print(f"MSE = {MSE*10**-9} x10⁹")
    RMSE = math.sqrt(MSE)
    print(f"RMSE = {RMSE}")
    MAE = mean_absolute_error(y_axis, y_fit)
    print(f"MAE = {MAE}")
    
    x_fit = [min(x_axis), max(x_axis)]
    y_fit = [powerlaw_fit(x, a_fit, b_fit) for x in x_fit]

    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.scatter(x_axis, y_axis, s=30, facecolors='r', edgecolors='r', label="data points")
    #plt.plot(x_fit, y_fit, color="black", label="fitted line")
    #plt.scatter(x_axis, y_fit, s=30, facecolors='b', edgecolors='b')# color="black", label="fitted")
    #plt.annotate(f"R²={round(R_square, 3)}\nf(x)={round(a_fit, 3)}x^({round(b_fit, 3)})", xy=(10, 0.1), color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("tk/d (-)")
    plt.ylabel("q/k=Kr (-)") # α
    plt.legend()
    plt.show()

def soil_type():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math

    plt.rcParams.update({'font.size': 22})
    filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv' # with watertable depth, n, alfa, theta, Ks from rosetta
    
    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "time"
    df = pd.read_csv(filepath)
        
    soil_types = list(df[k_column].unique())
    soil_types.sort()

    ax = plt
    ax.figure(figsize=(16,9))

    for soil in soil_types:
        x_axis = []
        y_axis = []
        df_soil = df[df[k_column]==soil]
        for i in range(len(df_soil)):
            q = df_soil[q_column].iloc[i]
            k = df_soil[k_column].iloc[i]
            d = df_soil[d_column].iloc[i]
            n = df_soil[n_column].iloc[i]
            alfa = df_soil[alfa_column].iloc[i]
            theta_r = df_soil[theta_r_column].iloc[i]
            theta_s = df_soil[theta_s_column].iloc[i]
            t = df_soil[t_column].iloc[i]
            if k>=q and t!=0 and (q/k)>=0.001:
                x = t*k/(n*n*d)
                y = q/k
                y_axis.append(y)
                x_axis.append(x)
        print(soil)

        ax.scatter(x_axis, y_axis, s=30, label=f"Ks={round(soil, 4)}")


    
    ax.grid(True)
    ax.xscale("log")
    ax.yscale("log")
    ax.xlabel("tk/d (-)")
    ax.ylabel("q/k=Kr (-)") # α
    ax.legend()
    ax.show()

soil_type()