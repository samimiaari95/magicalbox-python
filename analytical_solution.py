import pandas as pd
import numpy as np
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


def plot_roots():
    L = 400

    # Define the function
    def f(x):
        return np.tan(x*L) + 2*x

    # Create an array of x-values
    x_values = np.linspace(0, 10, 1000000)

    # Compute the corresponding y-values
    y_values = f(x_values)

    # Plot the function
    plt.plot(x_values, y_values, label='f(x) = tan(x*L) + 2x', color="black")

    # Add x-axis and y-axis lines
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

def find_roots_from_plot(L):

    # Define the function
    def f(x):
        y = np.tan(x*L) + 2*x
        return y

    # Create an array of x-values
    x_values = np.linspace(0, 20, 1000000)
    print("calculating y values")
    # Compute the corresponding y-values
    y_values = f(x_values)

    zeros = [0] * len(x_values)
    line1 = LineString(np.column_stack((x_values, y_values)))
    line2 = LineString(np.column_stack((x_values, zeros)))
    print("looking for intersections")
    intersection = line1.intersection(line2)

    points = [p.x for p in intersection]
    roots = [point for point in points if abs(f(point))< 0.0001]


    return roots
    plt.plot(x_values, y_values, color="black")
    plt.plot(x_values, zeros, color="black")
    new_zeros = [0 for i in range(len(points))]
    plt.plot(points, new_zeros, "ro")
    plt.xlabel("λ")
    plt.ylabel("f(λ)")
    plt.show()


def newton_raphson_func(x0, L):
    #L = 1
    tolerance = 1e-8  # Set the desired tolerance for convergence
    max_iterations = 1000  # Set a maximum number of iterations

    # Define the function f(x) and its derivative f'(x)
    def f(x):
        return np.tan(x*L) + 2*x

    def df(x):
        return L*(1/np.cos(x*L)**2) + 2

    # Implement the Newton-Raphson method
    def newton_raphson(x0):
        x_n = x0
        for _ in range(max_iterations):
            f_value = f(x_n)
            df_value = df(x_n)
            x_n1 = x_n - f_value / df_value

            if abs(x_n1 - x_n) < tolerance:
                return x_n1
            x_n = x_n1

        return None  # If the method does not converge within the maximum iterations

    # Initial guess for the root
    #x0 = 100

    # Find the root using the Newton-Raphson method
    root = newton_raphson(x0)

    if root is not None:
        #print("Approximate root:", root)
        return root
    else:
        #print("The Newton-Raphson method did not converge to a root.")
        return None

def residue(lambdaa, L, t, z):
    residue_value = (np.sin(lambdaa*z))*np.sin(lambdaa*L)*(np.exp(-t*(lambdaa*lambdaa)))/(1+(L/2)+(2*L*lambdaa*lambdaa))
    return residue_value

def sum_residue(L, t, z, roots):
    sum_values = []
    #plot_list = []
    
    for root in roots:
        resid = residue(root, L, t, z)
        sum_values.append(resid)
        
        #if z>0:
        #    plot_list.append(np.sum(sum_values))
    #if z >0:
    #    plt.plot(roots, plot_list, color="black")
    #    plt.xlabel("λ")
    #    plt.ylabel("Sum of residue")
    #    plt.show()
    #    return
    
    residues = np.sum(sum_values)

    return residues

def analytical_pressure():
    # define variables in cm and hr
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    # alfa = 0.0262 perfect for t=1091
    alfa = 0.025616931634881337 # NOT 1/cm but fitted based on K from Van Genuchten
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0 #cm/hr
    #qb_star = 0.9 #cm/hr
    qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    data = {}
    data = {"z": [*range(0, L_star+1, 1)]}
    # calculate for timesteps
    timesteps = range(1, 2002, 1)
    #timesteps = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    timesteps = [10,100,1091]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            #plotit = True
            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            #if z>0:
            #    plotit = False
            residues = sum_residue(L, t, z, roots)
            #if not plotit:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
            
    
    df = pd.DataFrame(data)
    print(df)
    df = df/100
    df.to_csv("/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure_modalfa.csv", index=False)
    return
    count = -1
    timesteps = [1091]
    for time in timesteps:
        x_axis = df[f"time={time}"].to_list()
        y_axis = df["z"].to_list()
        plt.plot(x_axis, y_axis, label=f"t={time}")
        plt.annotate(f"{time}", xy=(x_axis[count], y_axis[count]))
        count -= 8

    plt.xlabel("pressure head (cm)")
    plt.ylabel("z (cm)")
    plt.legend()
    plt.show()

def plot_analytical_numerical():
    plt.rcParams.update({'font.size': 22})
    # read from numerical solution
    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure.csv"
    numerical_solution = pd.read_csv(numerical_path)
    y_axis = numerical_solution["z"].to_list()
    y_axis = [4-x for x in y_axis]
    x_axis10 = numerical_solution["time=10"].to_list()
    x_axis100 = numerical_solution["time=100"].to_list()
    x_axis1091 = numerical_solution["time=1091"].to_list()
    plt.plot(x_axis10, y_axis, color="red")
    plt.annotate(f"t=10", xy=(x_axis10[-1], y_axis[-1]), color="red")
    plt.plot(x_axis100, y_axis, color="red")
    plt.annotate(f"t=100", xy=(x_axis100[-7], y_axis[-7]+0.2), color="red")
    plt.plot(x_axis1091, y_axis, color="red")
    plt.annotate(f"t=1091", xy=(x_axis1091[-1], y_axis[-1]), color="red")
    
    

    # read from analytical solution
    analytical_path = "/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure_modalfa.csv"
    analytical_solution = pd.read_csv(analytical_path)
    y_axis = analytical_solution["z"].to_list()
    y_axis = [4-x for x in y_axis]
    x_axis10 = analytical_solution["time=10"].to_list()
    x_axis100 = analytical_solution["time=100"].to_list()
    x_axis1091 = analytical_solution["time=1091"].to_list()
    plt.plot(x_axis10, y_axis, color="blue")
    plt.annotate(f"t=10", xy=(x_axis10[-2]-0.2, y_axis[-2]), color="blue")
    plt.plot(x_axis100, y_axis, color="blue")
    plt.annotate(f"t=100", xy=(x_axis100[-150]-0.3, y_axis[-150]), color="blue")
    plt.plot(x_axis1091, y_axis, color="blue")
    plt.annotate(f"t=1091", xy=(x_axis1091[-4], y_axis[-4]+0.3), color="blue")

    plt.plot([-4,0],[0,4],color="black")
    plt.annotate(f"t=0", xy=(-3.9, 0.1), color="black")

    plt.gca().invert_yaxis()
    plt.xlim([-4, 0.2])
    plt.xlabel("Pressure (m)")
    plt.ylabel("Soil depth (m)")
    line_up, = plt.plot([], label='Analytical solution', color="blue")
    line_down, = plt.plot([], label='Numerical solution', color="red")
    plt.legend(handles=[line_up, line_down])
    plt.show()
    #plt.savefig("/p/project/cslts/miaari1/python_scripts/analyticalvsnumerical.png")

def optimize_alfa(alfa):
    # define variables in cm and hr
    #L_star = 100 #cm
    L_star = 400 #cm
    Ks = 1 #cm/hr
    #alfa = 0.1 # 1/cm
    # alfa = 0.0262 perfect for t=1091
    #alfa = 0.020143 # NOT 1/cm but fitted based on K from Van Genuchten
    #Ss = 0.4
    Ss = 1
    #Sr = 0.06
    Sr = 0.2
    #qa_star = 0.1 #cm/hr
    qa_star = 0 #cm/hr
    #qb_star = 0.9 #cm/hr
    qb_star = 0.1 #cm/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    data = {}
    data = {"z": [*range(0, L_star, 10)]}
    # calculate for timesteps
    timesteps = range(1, 2002, 1)
    #timesteps = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    #timesteps = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    timesteps = [100]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (alfa*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:
            #plotit = True
            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = alfa*z_star
            #if z>0:
            #    plotit = False
            residues = sum_residue(L, t, z, roots)
            #if not plotit:
            #    print(residues)
            #    return
            #print(f"for t_star={t_star} and z={z_star} the residue is: {residues}")
            
            K = qb - (qb - np.exp(alfa*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/alfa
            data[f"time={t_star}"].append(pressure)
            
    
    df = pd.DataFrame(data)
    df = df/100
    return df

def find_alfa():
    from sklearn.metrics import mean_squared_error 

    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure.csv"
    numerical_solution = pd.read_csv(numerical_path)
    y_axis = numerical_solution["z"].to_list()
    x_num = numerical_solution["time=100"].to_list()
    best_mse = 10
    best_alfa = 0.02
    for alfa in range(1000, 3000, 1):
        print(f"we're in iteration {alfa}")
        alfa = alfa/100000
        df = optimize_alfa(alfa)
        x_ana = df["time=100"].to_list()
        mse = mean_squared_error(x_num,x_ana)
        if abs(mse)<abs(best_mse):
            best_mse = mse
            best_alfa = alfa
    print(f"best alfa: {best_alfa}")
    print(f"best mse: {best_mse}")

analytical_pressure()
plot_analytical_numerical()