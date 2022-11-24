"""
Created on Tue Nov 22 09:58:42 2022

@author: Marta
"""

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import scipy.optimize as so
import matplotlib.pyplot as pl

x_position, y_position = np.loadtxt("data7.txt", delimiter="\t", unpack=True)

uXposition = 0.003 * x_position #μm
uYposition = 0.003 * y_position #μm
x_position = x_position * 0.12048 #μm
y_position = y_position * 0.12048 #μm
x0 = x_position[0]
y0 =y_position[0]

uXdisplacement = np.sqrt(2) * uXposition
uYdisplacement = np.sqrt(2) * uYposition
x_displacement = np.arange(0,120,1)
y_displacement = np.arange(0,120,1)
x_displacement_squared = np.arange(0,120,1)
y_displacement_squared = np.arange(0,120,1)
for i in np.arange(0,len(x_position),1):
    x_displacement_squared[i] = x_position[i]-x0
    x_displacement_squared[i] = (x_position[i]-x0)**2
uX_displacement_squared = x_displacement_squared * 2 * (uXdisplacement / x_displacement)    

for i in np.arange(0,len(y_position),1):
    y_displacement_squared[i] = y_position[i]-y0
    y_displacement_squared[i] = (y_position[i]-y0)**2
uY_displacement_squared = y_displacement_squared * 2 * (uYdisplacement / y_displacement)

uR_squared = np.sqrt(((uX_displacement_squared)**2) + ((uY_displacement_squared)**2))
    
t = np.arange(0,60,0.5)
uT = 0.05 #s

r_squared = y_displacement_squared + x_displacement_squared

def model(x, m):
    return x * m

popt, pcov = so.curve_fit(model, t, r_squared)

pl.cla()
pl.plot(t, model(t,popt[0]))
pl.errorbar(t, r_squared, xerr=uT, yerr = uR_squared, fmt=" ")
pl.plot(t, r_squared, ".")
pl.ylabel("Mean Displacment Squared (μm)")
pl.xlabel("Time (s)")
pl.legend(["Linear fit","Measured Data"])
#pl.savefig("DATASET 2 - R2 over t")

eta = 10e-8