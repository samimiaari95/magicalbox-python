from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
import math
import pandas as pd
import numpy as np
import SLOTH.sloth.toolBox
import SLOTH.sloth.IO
import SLOTH.sloth.PlotLib
import SLOTH.sloth.mapper
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import powerlaw

plt.rcParams.update({'font.size': 22})
######################### INFILTRATION INVENTORY ###########################

def plot1(q, k, d, n, alfa, theta_r, theta_s, t):
    x = q
    y = t/d
    xlabel = "q (m/hr)"
    ylabel = "t/d (hr/m)"
    return x, y, xlabel, ylabel

def plot2(q, k, d, n, alfa, theta_r, theta_s, t):
    x = q/k
    y = t*k/d
    xlabel = "q/k (-)"
    ylabel = "t*k/d (-)"
    return x, y, xlabel, ylabel

def plot3(q, k, d, n, alfa, theta_r, theta_s, t):
    x = q/k
    y = t*q/d
    xlabel = "q/k (-)"
    ylabel = "t*q/d (-)"
    return x, y, xlabel, ylabel

def plot4(q, k, d, n, alfa, theta_r, theta_s, t):
    x = q/(k*d*alfa)
    y = t*k*alfa/((theta_s-theta_r))
    xlabel = "q/(k*d*alpha) (-)"
    ylabel = "t*k*alpha/(theta_s - theta_r) (-)"
    return x, y, xlabel, ylabel

def plot5(q, k, d, n, alfa, theta_r, theta_s, t):
    x = (q/(k))
    y = (t*k/(d))**(1-1/n)
    xlabel = "q/k (-)"
    ylabel = "(t*k/(n*d))^(1-1/n) (-)"
    return x, y, xlabel, ylabel

inf_cases_path = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_appended.csv'

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

# Fit the function
params, covariance = curve_fit(powerlaw, x_all, y_all)
a_fit, b_fit = params
print(f"Fitted a: {a_fit} and b:{b_fit}")

# Generate data points for the fitted curve
y_fit = [powerlaw(x, a_fit, b_fit) for x in x_all]

# calculate fitting accuracy
R_square = r2_score(y_all, y_fit)
print(f"R2 = {R_square}")
MSE = np.square(np.subtract(y_all,y_fit)).mean()
print(f"MSE = {MSE*10**-9} x10⁹")
RMSE = math.sqrt(MSE)
print(f"RMSE = {RMSE}")
MAE = mean_absolute_error(y_all, y_fit)
print(f"MAE = {MAE}")

x_fit = [min(x_all), max(x_all)]
y_fit = [powerlaw(x, a_fit, b_fit) for x in x_fit]

plt.plot(x_fit, y_fit, color="k", label="fitted line", linewidth=5)
plt.annotate(f"R²={round(R_square, 2)}\nf(x)={round(a_fit, 2)}x^({round(b_fit, 2)})", xy=(min(x_all), min(y_all)), color="black")

ax.scatter(x_all, y_all,c=colors, cmap='jet', s=30, norm=matplotlib.colors.LogNorm())
ax.grid(True)
ax.xscale("log")
ax.yscale("log")
ax.xlabel(f"{xlabel}") # α
ax.ylabel(f"{ylabel}")
ax.colorbar().ax.set_ylabel('Ks (m/hr)')
ax.show()

    
    