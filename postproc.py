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


name='/p/project/cslts/miaari1/python_scripts/parflow/comparewithanalytical/exfiltration/exfiltration'
nt=2000
pressures = {}
pressures["z"] = [z/100 for z in range(5,400, 10)]
timesteps = [1]#, 20, 30, 50, 75, 100]

index = 9
for t in timesteps:
    #t = 193
    print(name + '.out.satur.'+('{:05d}'.format(t))+'.pfb')
    #data = pfr.read(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')
    data = SLOTH.sloth.IO.read_pfb(name + '.out.press.'+ ('{:05d}'.format(t)) + '.pfb')

    plt.plot(data[:,0,0],pressures["z"], color="black")
    if index<0:
        index = 0
    plt.annotate(f"t={t}", xy=(data[:,0,0][index], pressures["z"][index]), color="black")
    index -= 1

    pressures[f"time={t}"] = data[:,0,0]

print(data[:,5,5].shape)
print(len(pressures["z"]))
df = pd.DataFrame(pressures)
#df.to_csv('/p/project/cslts/miaari1/python_scripts/outputs/clay_soil.csv', index=False)

plt.show()
