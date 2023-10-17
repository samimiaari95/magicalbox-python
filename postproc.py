# Functions to extract Parflow output
# NAME:
#    functions_output.py
# VERSION:
#    03/25/2019 v0.5
# STATUS:
#    under development
# LANGUAGE:
#    Python 3.6x
# OWNER:
#    Thomas Pomeon - t.pomeon@fz-juelich.de - TP
# BASED ON WORK BY:
#    Niklas Wagner
#    Yueling Ma
# PURPOSE:
#    Reads Parflow .pfb outputs and exports discharge timeseries for a certain location #    and as an np array for the entire domain.
# REQUIRES:
#     sys, os, numpy, pf_read

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as plt
#import pandas as pd

#Define path to pf_read
#import pf_read as pfr
import SLOTH.sloth.IO

def save_steady_state(values, title):
    steady_state = pd.read_csv('/p/project/cslts/miaari1/python_scripts/steady_state.csv')
    steady_state[f'{title}'] = data[:,0,0]
    steady_state.to_csv('/p/project/cslts/miaari1/python_scripts/steady_state.csv', index=False)
    return

name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/srivastavaandyeh_0-1_2ndinf/exfiltration'
z = list(range(1,11))
nt=2000

for t in range (nt+1):
    #t = 193
    print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
    #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
    data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')

    plt.plot(data[:,0,0],z)
    

# NOTE comment it when testing
#save_steady_state(data[:,0,0], '001_10_193')
print(data[:,0,0])

plt.show()
print(data.shape)
print(data[0,0,0])
print(data[-1,0,0])