# -*- coding: u\tf-8 -*-
"""
Created on Tue Nov 22 09:58:42 2022

@author: Marta
"""

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import scipy.optimize as so
import matplotlib.pyplot as plt

x_position, y_position = np.loadtxt("data1.txt", delimiter="\t", unpack=True)

xo= x_position[0]
yo=y_position[0]
 
xd=np.arange(0,120,1)
yd=np.arange(0,120,1)
x_displacement_squared = np.arange(0,120,1)
y_displacement_squared = np.arange(0,120,1)

for i in np.arange(0,len(x_position),1):
    x_displacement_squared[i] = ((x_position[i]-xo)*0.1208)**2


for i in np.arange(0,len(y_position),1):
    y_displacement_squared[i] = ((y_position[i]-yo)*0.1208)**2
    
time=np.arange(0,60,0.5)
x_err=x_position*0.003

r_squared = y_displacement_squared + x_displacement_squared

plt. plot(time, r_squared, marker=".", linestyle="None")
plt.ylabel("Mean Displacment Squared (μm)")
plt.xlabel("Time (s)")
plt.title("Mean Displacment vs Time")



    