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


def graphical_roots(L):
    # Define the function
    def equation(x, L):
        return np.tan(x * L) + 2 * x

    
    # Define the range of x values to plot
    x_values = np.linspace(0, 100, 1000000)

    # Calculate corresponding y values
    y_values = equation(x_values, L)

    # Plot the function
    #plt.plot(x_values, y_values, label='f(x) = tan(xL) + 2x')

    # Add a horizontal line at y=0 for reference
    #plt.axhline(0, color='red', linestyle='--', linewidth=1)

    # Set the y-axis limits for better visualization
    #plt.ylim(-10, 10)

    # Find where the function crosses the x-axis (y=0)
    zero_crossings = np.where(np.diff(np.sign(y_values)))[0]

    # Get approximate roots
    roots = [x_values[i] for i in zero_crossings]

    # Print the approximate roots
    print("Approximate Roots:", roots)
    return roots


def rootscalar(L, myroot):
    from scipy.optimize import root_scalar

    # Define the function
    def equation(x, L):
        return np.tan(x*L) + 2*x

    # Set the constant L
#    L = 1.0  # Replace with your desired value

    # Use the root_scalar function to find a root
    print(myroot)
    result = root_scalar(equation, args=(L,), bracket=[myroot, 20])
    print(result.root)
    # The root will be in result.root
    #print("Root:", result.root)
    return result.root


def newton_raphson_func(x0, L):
    #L = 1
    tolerance = 1e-12  # Set the desired tolerance for convergence
    max_iterations = 10000  # Set a maximum number of iterations

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
    
def plot_roots():
    L = 4

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
    print("applying newton-raphson")
    roots = [p.x for p in intersection]
    #roots = [rootscalar(L, x) for x in roots]
    #roots = graphical_roots(L)
    #roots = [newton_raphson_func(x, L) for x in roots]
    #roots = [root for root in roots if root]
    roots = [x for x in roots if abs(f(x))< 1e-5]
    print(len(roots))
    return roots
    plt.plot(x_values, y_values, color="black")
    plt.plot(x_values, zeros, color="black")
    new_zeros = [0 for i in range(len(roots))]
    plt.plot(roots, new_zeros, "ro")
    plt.xlabel("λ")
    plt.ylabel("f(λ)")
    plt.show()

def residue(lambdaa, L, t, z):
    residue_value = (np.sin(lambdaa*z))*np.sin(lambdaa*L)*(np.exp(-t*(lambdaa*lambdaa)))/(1+(L/2)+(2*L*lambdaa*lambdaa))
    return residue_value

def sum_residue(L, t, z, roots):
    sum_values = []
    plot_list = []
    
    for root in roots:
        resid = residue(root, L, t, z)
        sum_values.append(resid)
    
    residues = np.sum(sum_values)

    return residues

def plot_sum_residue(alfa):
    L_star = 4 #m
    Ks = 0.01 #m/hr
    Ss = 1
    Sr = 0.2
    qa_star = 0 #m/hr
    qb_star = 0.001 #m/hr
    pressure_0 = 0 #cm

    # define dimensionless parameters
    L = alfa*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    data = {}
    depth = 0.5
    data = {"z": [4-depth]}
    t_star = 100
    data[f"time={t_star}"] = []
    t = (alfa*Ks*t_star)/(Ss-Sr)
    z_star = data["z"][0]
    # for z_star=0 bottom layer, z_star=L_star top layer
    z = alfa*z_star
    
    sum_values = []
    plot_list = []
    roots.sort()
    for root in roots:
        resid = residue(root, L, t, z)
        sum_values.append(resid)
        
        if z>0:
            plot_list.append(np.sum(sum_values))
    if z >0:
        plt.plot(roots, plot_list, color="black")
        plt.xlabel("λ")
        plt.ylabel("Sum of residue")
        plt.show()
        return
    

def analytical_pressure():
    plt.rcParams.update({'font.size': 22})
    # define variables in m and hr
    L_star = 1 #m
    Ks = 0.01 #m/hr
    c = 10 #1/m 
    Ss = 0.4
    Sr = 0.06
    qa_star = 0.000 #m/hr
    qb_star = 0.001 #m/hr
    pressure_0 = 0 #m

    # define dimensionless parameters
    L = c*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    z_values = [x/100 for x in range(5, L_star*100-4, 1)]

    data = {}
    data = {"z": z_values}
    print(z_values)
    # calculate for timesteps
    #timesteps = range(1, 2002, 1)
    timesteps = [138]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (c*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = c*z_star
            
            residues = sum_residue(L, t, z, roots)
            
            K = qb - (qb - np.exp(c*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/c
            data[f"time={t_star}"].append(pressure)
            
    
    df = pd.DataFrame(data)
    y_axis = df["z"].to_list()
    timesteps = [138]
    for t in timesteps:
        x_axis = df[f"time={t}"].to_list()
        plt.plot(x_axis, y_axis, "k")
        plt.annotate(f"t=0", xy=(x_axis[-1], y_axis[-1]), color="black")



    # define variables in m and hr
    L_star = 1 #m
    Ks = 0.01 #m/hr
    c = 10 #1/m 
    Ss = 0.4
    Sr = 0.06
    qa_star = 0.001 #m/hr
    qb_star = 0.009 #m/hr
    pressure_0 = 0 #m

    # define dimensionless parameters
    L = c*L_star
    qa = qa_star/Ks
    qb = qb_star/Ks

    # get the roots for this L
    roots = find_roots_from_plot(L)

    # define layers
    z_values = [x/100 for x in range(5, L_star*100-4, 1)]

    data = {}
    data = {"z": z_values}
    print(z_values)
    # calculate for timesteps
    #timesteps = range(1, 2002, 1)
    timesteps = [0, 1, 3, 5, 10, 15, 20, 30, 50, 75]
    for t_star in timesteps:
        data[f"time={t_star}"] = []
        t = (c*Ks*t_star)/(Ss-Sr)
        for z_star in data["z"]:            
            # for z_star=0 bottom layer, z_star=L_star top layer
            z = c*z_star
            
            residues = sum_residue(L, t, z, roots)
            
            K = qb - (qb - np.exp(c*pressure_0))*np.exp(-z) - 4*(qb - qa)*(np.exp((L-z)/2)*np.exp(-t/4))*residues

            K_star = K*Ks
            pressure = (np.log(K_star/Ks))/c
            data[f"time={t_star}"].append(pressure)
            
    
    df = pd.DataFrame(data)
    #print(df)
    df.to_csv("/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure_example.csv", index=False)
    return

def plot_analytical_numerical():
    # TODO log fit of VGM and Haverkamp and Gardner parameters
    plt.rcParams.update({'font.size': 22})
    
    # read from numerical solution
    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure_example.csv"
    numerical_solution = pd.read_csv(numerical_path)
    y_axis = numerical_solution["z"].to_list()
    #y_axis = [1-x for x in y_axis]
    timesteps = [0, 1, 3, 10]#, 15, 20, 30, 50, 75]

    index = 0
    for t in timesteps:
        index -= 1
        x_axis = numerical_solution[f"time={t}"].to_list()
        #x_axis = [x*-1 for x in x_axis]
        #x_axis = [van_genuchten_maulem_k(alfa, abs(h), n, m) for h in x_axis]
        plt.plot(x_axis, y_axis, "k--")
        plt.annotate(f"{t}", xy=(x_axis[index], y_axis[index]), color="black")
    

    # read from analytical solution
    analytical_path = "/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure_example.csv"
    analytical_solution = pd.read_csv(analytical_path)
    y_axis = analytical_solution["z"].to_list()
    #y_axis = [1-x for x in y_axis]
    index = 0
    timesteps = [1, 3, 10]#, 15, 20, 30, 50, 75]
    for t in timesteps:
        index -= 1
        x_axis = analytical_solution[f"time={t}"].to_list()
        #x_axis = [x*-1 for x in x_axis]
        #x_axis = [gardner(abs(h), c) for h in x_axis]
        plt.plot(x_axis, y_axis, "k")
        plt.annotate(f"{t}", xy=(x_axis[index], y_axis[index]), color="black")
    

    # read from numerical solution
    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure_example_havekamp.csv"
    numerical_solution = pd.read_csv(numerical_path)
    y_axis = numerical_solution["z"].to_list()
    #y_axis = [1-x for x in y_axis]
    timesteps = [0, 1, 3, 10]#, 15, 20, 30, 50, 75]

    index = 0
    for t in timesteps:
        index -= 1
        x_axis = numerical_solution[f"time={t}"].to_list()
        #x_axis = [x*-1 for x in x_axis]
        #x_axis = [van_genuchten_maulem_k(alfa, abs(h), n, m) for h in x_axis]
        plt.plot(x_axis, y_axis, "k:")
        plt.annotate(f"{t}", xy=(x_axis[index], y_axis[index]), color="black")
    
    
    #plt.gca().invert_yaxis()
    plt.xlim([-0.3, 0])
    plt.ylim([0, 1])
    plt.xlabel("Pressure (m)")
    plt.ylabel("Soil depth (m)")
    #line_up, = plt.plot([], label='Analytical solution', color="blue")
    line_up, = plt.plot([], 'k', label='Analytical (Gardner)')
    line_mid, = plt.plot([], 'k:', label='Numerical (Haverkamp)')
    line_down, = plt.plot([], 'k--', label='Numerical (Van Genuchten-Mualem)')
    plt.legend(handles=[line_up, line_down, line_mid])
    plt.show()
    #plt.savefig("/p/project/cslts/miaari1/python_scripts/analyticalvsnumerical.png")

def fit_exp(h_values, y_data, theta_r, theta_s, n,m):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    plt.rcParams.update({'font.size': 22})

    # Define the gardner model function for pressure (h) and soil water content (θ) or relative permeability Kr
    def gardner_theta(h, a, b):
        theta = a*(h**(-b))
        return theta
    def gardner_k(h, c):
        k = np.exp(-c*h)
        return k
    
    # Fit the data to the exponential function
    params, covariance = curve_fit(gardner_k, h_values, y_data)
    # Extract the fitted parameters
    a_fit = params
    
    # Print the results
    print(f"Fitted a:{a_fit}")

    # Generate data points for the fitted curve
    y_fit = [gardner_k(x,a_fit) for x in h_values]

    # Plot the original data and the fitted curve
    plt.plot(h_values, y_data, label='Van Genuchten-Mualem Model')
    plt.plot(h_values, y_fit, 'r', label='Gardner model')
    plt.ylabel('Relative hydraulic conductivity Kr (-)')
    #plt.ylabel('Water content θ (m³/m³)')
    plt.xlabel('Pressure (m)')
    plt.xscale("log")
    plt.yscale("log")
    #plt.gca().invert_xaxis()
    plt.ylim([1e-12, 1])
    plt.title('Van Genuchten-Mualem vs Gardner models')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_vangenuchten():
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the van Genuchten-Mualem model function for pressure head (h) and soil water content (θ) or relative permeability Kr
    def van_genuchten_maulem_theta(theta_r, theta_s, alfa, n, m, h):
        theta = theta_r + (theta_s-theta_r)/((1+((alfa*h)**n))**m)
        return theta

    def van_genuchten_maulem_k(alfa, h, n, m):
        k = ((1-((alfa*h)**(n-1))*((1+(alfa*h)**(n))**(-m)))**(2))/((1+(alfa*h)**(n))**(m/2))
        return k
    # Define the parameter values for the model
    θs = 1     # Saturated water content
    θr = 0.2    # Residual water content
    alfa = 1     # Inverse of air entry pressure (1/m)α
    n = 2       # Van Genuchten parameter
    m = 1.0 - 1.0 / n  # Mualem parameter
    theta_r = θr
    theta_s = θs

    # Calculate pressure head (h) values using the van Genuchten-Mualem model
    h_values = [h/100 for h in range(1, 1000000)]
    y_values = [van_genuchten_maulem_k(alfa, h, n, m) for h in h_values]
    #y_values = [van_genuchten_maulem_theta(theta_r, theta_s, alfa, n, m, h) for h in h_values]
    fit_exp(h_values, y_values, theta_r, theta_s, n, m)
    return


def fit_vangenuchten():
    # TODO log fit of VGM and Haverkamp and Gardner parameters
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import math
    plt.rcParams.update({'font.size': 22})

    def van_genuchten_maulem_k(h, alfa, n):
        m = 1-1/n
        k = ((1-((alfa*h)**(n-1))*((1+(alfa*h)**(n))**(-m)))**(2))/((1+(alfa*h)**(n))**(m/2))
        return k
    def gardner_k(h, c):
        k = np.exp(-c*h)
        return k
    def haverkamp_k(h, A, gamma):
        k = A/(A + (h**gamma))
        return k
    
    c = 10 #/m
    h_values = [h/100 for h in range(1, 100000)]
    y_values = [gardner_k(h, c) for h in h_values]

    params, covariance = curve_fit(haverkamp_k, h_values, y_values)
    A, gamma = params
    print(f"Fitted A: {A} and gamma: {gamma}")
    y_haverkamp = [haverkamp_k(x, A, gamma) for x in h_values]

    params, covariance = curve_fit(van_genuchten_maulem_k, h_values, y_values)
    alfa, n = params
    print(f"Fitted alfa: {alfa} and n: {n}")
    y_VG = [van_genuchten_maulem_k(x, alfa, n) for x in h_values]

    # Plot the original data and the fitted curve
    plt.plot(h_values, y_haverkamp, label='Haverkamp')
    plt.plot(h_values, y_VG, label='Van Genuchten-Mualem')
    plt.plot(h_values, y_values, label='Gardner')
    plt.ylabel('Relative hydraulic conductivity Kr (-)')
    #plt.ylabel('Water content θ (m³/m³)')
    plt.xlabel('Pressure (m)')
    plt.xscale("log")
    plt.yscale("log")
    #plt.gca().invert_xaxis()
    plt.ylim([1e-12, 1])
    plt.title('Van Genuchten-Mualem vs Gardner models')
    plt.legend()
    plt.grid(True)
    plt.show()



def pressure_vs_Kr():
    def van_genuchten_maulem_k(alfa, h, n, m):
        # relative permeability by van genuchten & mualem
        k = ((1-((alfa*h)**(n-1))*((1+(alfa*h)**(n))**(-m)))**(2))/((1+(alfa*h)**(n))**(m/2))
        return k
    def gardner(h, c):
        # relative permeability by gardner
        k = np.exp(-c*h)
        return k
    
    #numerical solution
    numerical_path = "/p/project/cslts/miaari1/python_scripts/outputs/numerical_pressure.csv"
    numerical_solution = pd.read_csv(numerical_path)
    alfa = 1 #1/m
    n = 2
    m = 1-1/n

    x_axis = numerical_solution[f"time=100"].to_list()
    x_axis = [x*-1 for x in x_axis]
    y_axis = [van_genuchten_maulem_k(alfa, abs(h), n, m) for h in x_axis]
    
    plt.plot(x_axis, y_axis, color="red")

    # analytical solution
    analytical_path = "/p/project/cslts/miaari1/python_scripts/outputs/analytical_pressure.csv"
    analytical_solution = pd.read_csv(analytical_path)
    c = 2.46491113

    x_axis = analytical_solution[f"time=100"].to_list()
    x_axis = [x*-1 for x in x_axis]
    y_axis = [gardner(abs(h), c) for h in x_axis]
    
    plt.plot(x_axis, y_axis, color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


#analytical_pressure()
#plot_analytical_numerical()
#c = 2.46491113
#plot_sum_residue(c)
fit_vangenuchten()