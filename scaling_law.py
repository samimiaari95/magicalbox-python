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
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from utils import powerlaw_func, linear_law

plt.rcParams.update({'font.size': 22})


def fit_scalinglaw():
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
            
            x = q/k
            y = (d/(t*k))
            #*((q/k)**(8.6))
            
            y_axis.append(y)
            x_axis.append(x)
    
    # Fit the function
    params, covariance = curve_fit(powerlaw_func, x_axis, y_axis)
    a_fit, b_fit = params
    print(f"Fitted a: {a_fit} and b:{b_fit}")

    # Generate data points for the fitted curve
    y_fit = [powerlaw_func(x, a_fit, b_fit) for x in x_axis]

    # calculate fitting accuracy
    R_square = r2_score(y_axis, y_fit)
    print(f"R2 = {R_square}")
    MSE = np.square(np.subtract(y_axis,y_fit)).mean()
    print(f"MSE = {MSE*10**-9} x10⁹")
    RMSE = math.sqrt(MSE)
    print(f"RMSE = {RMSE}")
    MAE = mean_absolute_error(y_axis, y_fit)
    print(f"MAE = {MAE}")
    
    x_fit = [min(x_axis), max(x_axis)]
    y_fit = [powerlaw_func(x, a_fit, b_fit) for x in x_fit]

    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.scatter(x_axis, y_axis, s=30, facecolors='k', edgecolors='k', label="data points")
    plt.plot(x_fit, y_fit, color="red", label="fitted line")
    #plt.annotate(f"R²={round(R_square, 3)}\nf(x)={round(a_fit, 3)}x^({round(b_fit, 3)})", xy=(10, 0.1), color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("q/k=Kr (-)")
    plt.ylabel("(tq/d)*(q/k)^8.6 (-)") # α
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
                x = t
                y = k
                y_axis.append(y)
                x_axis.append(x)
        print(soil)
        ax.scatter(x_axis, y_axis, s=30, label=f"Ks={round(soil, 4)}")

    ax.grid(True)
    ax.xscale("log")
    ax.yscale("log")
    ax.xlabel("t (hr)")
    ax.ylabel("k (m/hr)") # α
    ax.legend()
    ax.show()

def soil_type_colormap():
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

    x_all = []
    y_all = []
    colors = []
    for soil in soil_types:
        x_axis = []
        y_axis = []
        df_soil = df[df[k_column]==soil]
        index = 0
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
                x = alfa*d
                y = t*q*alfa
                #x = q/k
                #y = k*t*alfa/(theta_s-theta_r)
                y_axis.append(y)
                x_axis.append(x)
                index += 1
        print(soil)
        x_all.extend(x_axis)
        y_all.extend(y_axis)
        colors.extend([soil]*len(x_axis))

    ax.scatter(x_all, y_all,c=colors, cmap='jet', s=30, norm=matplotlib.colors.LogNorm())
    ax.grid(True)
    ax.xscale("log")
    ax.yscale("log")
    ax.xlabel("x")
    ax.ylabel("y") # α
    ax.colorbar().ax.set_ylabel('Ks (m/hr)')
    ax.show()

def exf_soil_type_colormap():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math

    plt.rcParams.update({'font.size': 22})
    filepath = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_drainage.csv' # with watertable depth, n, alfa, theta, Ks from rosetta
    
    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "exf_time"
    df = pd.read_csv(filepath)
        
    soil_types = list(df[k_column].unique())
    soil_types.sort()

    ax = plt
    ax.figure(figsize=(16,9))

    x_all = []
    y_all = []
    colors = []
    for soil in soil_types:
        x_axis = []
        y_axis = []
        df_soil = df[df[k_column]==soil]
        index = 0
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
                x = alfa*d
                y = t*k*alfa
                #x = q/k
                #y = k*t*alfa/(theta_s-theta_r)
                y_axis.append(y)
                x_axis.append(x)
                index += 1
        print(soil)
        x_all.extend(x_axis)
        y_all.extend(y_axis)
        colors.extend([soil]*len(x_axis))

    ax.scatter(x_all, y_all,c=colors, cmap='jet', s=30, norm=matplotlib.colors.LogNorm())
    ax.grid(True)
    ax.xscale("log")
    ax.yscale("log")
    ax.xlabel("x")
    ax.ylabel("y") # α
    ax.colorbar().ax.set_ylabel('Ks (m/hr)')
    ax.show()

def constantq_constantd():
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_absolute_error
    import math

    plt.rcParams.update({'font.size': 22})
    filepath1 = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_constantq_constantd.csv' # without watertable depth
    filepath2 = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_constantq_constantd_v3.csv' # without watertable depth

    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "time"
    # first file
    df1 = pd.read_csv(filepath1)
    df1["q/k"] = df1[q_column]/df1[k_column]
    df1 = df1[df1["q/k"]<=1]
    df1.reset_index(drop=True, inplace=True)

    # second file
    df2 = pd.read_csv(filepath2)
    df2["q/k"] = df2[q_column]/df2[k_column]
    df2 = df2[df2["q/k"]<=1]
    df2.reset_index(drop=True, inplace=True)

    k1 = df1[k_column].to_list()
    t1 = df1[t_column].to_list()
    k2 = df2[k_column].to_list()
    t2 = df2[t_column].to_list()


    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.scatter(k1, t1, s=30, facecolors='b', edgecolors='b', label="q=0.0002 m/hr")
    plt.scatter(k2, t2, s=30, facecolors='r', edgecolors='r', label="q=0.001 m/hr")
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Ks (m/hr)")
    plt.ylabel("t (hr)")
    plt.title("d=4m & constant q")
    plt.legend()
    plt.show()

def sensitivity_alfa_n_k(sens_alfa=None, sens_n=None, sens_k=None, sens_alfan=None):
    def plot5(q, k, d, n, alfa, theta_r, theta_s, t):
        x = (q)/(k*alfa*d)
        y = t
        xlabel = "q/(k*d*alpha)"
        ylabel = "t"
        return x, y, xlabel, ylabel
    inf_cases_path = '/p/project/cslts/miaari1/python_scripts/outputs/infiltration_testcases.csv'
    if sens_alfa:
        title = "Sensitivity analysis for alpha"
        var_alfan_path = '/p/project/cslts/miaari1/python_scripts/parflow/sensitivity_analysis/infiltration_sensitivity_alfa.csv'
    elif sens_n:
        title = "Sensitivity analysis for n"
        var_alfan_path = '/p/project/cslts/miaari1/python_scripts/parflow/sensitivity_analysis/infiltration_sensitivity_n.csv'
    elif sens_k:
        title = "Sensitivity analysis for saturated hydraulic conductivity Ks"
        var_alfan_path = '/p/project/cslts/miaari1/python_scripts/parflow/sensitivity_analysis/infiltration_sensitivity_k.csv'    
    elif sens_alfan:
        title = "Sensitivity analysis for alpha and n"
        var_alfan_path = '/p/project/cslts/miaari1/python_scripts/parflow/sensitivity_analysis/infiltration_sensitivity_alfa_n.csv'    

    q_column = "q"
    k_column = "k"
    d_column = "d"
    n_column = "n"
    alfa_column = "alfa"
    theta_r_column = "theta_r"
    theta_s_column = "theta_s"
    t_column = "time"
    df = pd.read_csv(inf_cases_path)

    soil_types = list(df[k_column].unique())
    soil_types.sort()

    ax = plt
    ax.figure(figsize=(16,9))
    x_all = []
    y_all = []
    colors = []
    for soil in soil_types:
        x_axis = []
        y_axis = []
        df_soil = df[df[k_column]==soil]
        index = 0
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
                x, y, xlabel, ylabel = plot5(q, k, d, n, alfa, theta_r, theta_s, t)
                y_axis.append(y)
                x_axis.append(x)
                index += 1

        x_all.extend(x_axis)
        y_all.extend(y_axis)
        colors.extend([soil]*len(x_axis))

    
    ax.scatter(x_all, y_all,c=colors, cmap='jet', s=30, norm=matplotlib.colors.LogNorm())

    var_alfan_df = pd.read_csv(var_alfan_path)
    val_x = []
    val_y = []
    max_err_x = []
    max_err_y = []
    min_err_x = []
    min_err_y = []

    for i in range(len(var_alfan_df)):
        q = var_alfan_df.iloc[i,1]
        k = var_alfan_df.iloc[i,2]
        d = var_alfan_df.iloc[i,3]
        n = var_alfan_df.iloc[i,5]
        alfa = var_alfan_df.iloc[i,4]
        theta_r = var_alfan_df.iloc[i,6]
        theta_s = var_alfan_df.iloc[i,7]
        t = var_alfan_df.iloc[i,8]
        x, y, xlabel, ylabel = plot5(q, k, d, n, alfa, theta_r, theta_s, t)
        val_x.append(x)
        val_y.append(y)

        q = var_alfan_df.iloc[i,9]
        k = var_alfan_df.iloc[i,10]
        d = var_alfan_df.iloc[i,11]
        n = var_alfan_df.iloc[i,13]
        alfa = var_alfan_df.iloc[i,12]
        theta_r = var_alfan_df.iloc[i,14]
        theta_s = var_alfan_df.iloc[i,15]
        t = var_alfan_df.iloc[i,16]
        x, y, xlabel, ylabel = plot5(q, k, d, n, alfa, theta_r, theta_s, t)
        max_err_x.append(x)
        max_err_y.append(y)
        if var_alfan_df.iloc[i,16] != np.float("nan"):
            q = var_alfan_df.iloc[i,17]
            k = var_alfan_df.iloc[i,18]
            d = var_alfan_df.iloc[i,19]
            n = var_alfan_df.iloc[i,21]
            alfa = var_alfan_df.iloc[i,20]
            theta_r = var_alfan_df.iloc[i,22]
            theta_s = var_alfan_df.iloc[i,23]
            t = var_alfan_df.iloc[i,24]
            x, y, xlabel, ylabel = plot5(q, k, d, n, alfa, theta_r, theta_s, t)
            min_err_x.append(x)
            min_err_y.append(y)

    for i in range(len(var_alfan_df)):
        ax.plot([min_err_x[i], val_x[i], max_err_x[i]], [min_err_y[i], val_y[i], max_err_y[i]], marker='+', c="k", linewidth=2.0)#, label=f'Varied Input: {alfa} and {n}')

    ax.grid(True)
    ax.xscale("log")
    ax.yscale("log")
    ax.xlabel(f"{xlabel}") # α
    ax.ylabel(f"{ylabel}")
    ax.colorbar().ax.set_ylabel('Ks (m/hr)')
    ax.title(title)
    ax.show()


sensitivity_alfa_n_k(sens_alfan=True)