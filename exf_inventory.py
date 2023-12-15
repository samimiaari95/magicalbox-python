from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing
import math
import pandas as pd
import numpy as np
from scipy.stats import powerlaw, kstest, ks_2samp, anderson
import SLOTH.sloth.toolBox
import SLOTH.sloth.IO
import SLOTH.sloth.PlotLib
import SLOTH.sloth.mapper
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import powerlaw_func, linear_law

plt.rcParams.update({'font.size': 22})
######################### INFILTRATION INVENTORY ###########################
def anderson_darling(data):
    result = anderson(data, dist='power law')
    print(f"Anderson-Darling Test Statistic: {result.statistic}")
    print(f"Critical Values: {result.critical_values}")
    print(f"Significance Levels: {result.significance_level}")
    print(f"Is the data drawn from the specified distribution? {result.statistic < result.critical_values[2]}")



def KS_fit(data, fitted):
    # Perform Kolmogorov-Smirnov test for goodness of fit
    #ks_statistic, ks_p_value = kstest(data, powerlaw_func, args=(alpha,b))
    ks_statistic, ks_p_value = ks_2samp(data, fitted)
    print(f"ks_statistic = {ks_statistic}")
    print(f"ks_p_value = {ks_p_value}")
    return


def plot1(q, k, d, n, alfa, theta_r, theta_s, inf_t, exf_t):
    x = alfa*d*n
    y = exf_t*k*alfa
    xlabel = "α*d (-)"
    ylabel = "t*k*α (-)"
    return x, y, xlabel, ylabel

def plot2(q, k, d, n, alfa, theta_r, theta_s, inf_t, exf_t):
    x = inf_t
    y = exf_t
    xlabel = "alfa*d (-)"
    ylabel = "t*k*alfa (-)"
    return x, y, xlabel, ylabel


exf_cases_path = '/p/project/cslts/miaari1/python_scripts/outputs/testcases_drainage.csv'

q_column = "q"
k_column = "k"
d_column = "d"
n_column = "n"
alfa_column = "alfa"
theta_r_column = "theta_r"
theta_s_column = "theta_s"
inf_t_column = "inf_time"
exf_t_column = "exf_time"
df = pd.read_csv(exf_cases_path)
    
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
        inf_t = df_soil[inf_t_column].iloc[i]
        exf_t = df_soil[exf_t_column].iloc[i]
        if k>=q and exf_t!=0 and (q/k)>=0.001:
            x, y, xlabel, ylabel = plot1(q, k, d, n, alfa, theta_r, theta_s, inf_t, exf_t)
            y_axis.append(y)
            x_axis.append(x)
            index += 1

    x_all.extend(x_axis)
    y_all.extend(y_axis)
    colors.extend([soil]*len(x_axis))

# linearize
y_lin = np.log(y_all)
x_lin = np.log(x_all)
# Fit the function
params, covariance = curve_fit(linear_law, x_lin, y_lin)
a_fit, b_fit = params
#a_fit = np.log(0.2475)
#b_fit = -0.963

# fitting accuracy
y_fit = [linear_law(x, a_fit, b_fit) for x in x_lin]
R_square = r2_score(y_lin, y_fit)
print(f"R2 = {R_square}")
r = np.corrcoef(y_lin, y_fit)
r2 = r[0][1]**2
print(f"pearson r2: {r2}")
#plt.plot(x_lin, y_fit, color="k", label="fitted line", linewidth=5)
#plt.scatter(x_lin, y_lin, color="r", label="line")
#plt.show()
#exit()
# back transform to power law
a_fit = np.exp(a_fit)
print(f"Fitted a: {a_fit} and b:{b_fit}")


#a_fit = 0.0814093265449305
#b_fit = -0.970811357451946

# check kolmogorov-smirnov
#anderson_darling(y_all)
#KS_fit(y_all, y_fit)

# calculate fitting accuracy ####### NOT CONSIDERED, SHOULD BE CALCULATED ON LINEARIZED DATA
#R_square=1-(np.sum((np.array(y_all)-np.array(y_fit))**2)/np.sum((np.array(y_all)-np.array(np.mean(y_all)))**2))
#R_square = r2_score(y_all, np.exp(y_fit))
#print(f"R2 = {R_square}")
#r = np.corrcoef(y_all, np.exp(y_fit))
#r2 = r**2
#print(f"pearson r2: {r2}")
#MSE = np.square(np.subtract(y_all,y_fit)).mean()
#print(f"MSE = {MSE*10**-9} x10⁹")
#RMSE = math.sqrt(MSE)
#print(f"RMSE = {RMSE}")
#MAE = mean_absolute_error(y_all, y_fit)
#print(f"MAE = {MAE}")

x_fit = [min(x_all), max(x_all)]
y_fit = [powerlaw_func(x, a_fit, b_fit) for x in x_fit]
plt.plot(x_fit, y_fit, color="k", label="fitted line", linewidth=5)
plt.annotate(f"r²={round(r2,2)}\nf(x)={round(a_fit, 2)}x^({round(b_fit, 2)})", xy=(min(x_all), max(y_all)/10), color="black")

ax.scatter(x_all, y_all,c=colors, cmap='jet', s=30, norm=matplotlib.colors.LogNorm())
ax.grid(True)
ax.xscale("log")
ax.yscale("log")
ax.xlabel(f"{xlabel}") # α
ax.ylabel(f"{ylabel}")
ax.colorbar().ax.set_ylabel('Ks (m/hr)')
ax.show()
