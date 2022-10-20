#NICHOLAS GARCIA & MARTA GONCZAR
"""OSCILLATING MASS ON A SPRING"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl

m = 200.7; #g
T = 5.11 / 7; #seconds / oscillations = period

xt, x, ux = np.loadtxt("x.txt", unpack=True)

#converting from cm to m
x = x / 100 
ux = ux / 100

pl.cla()
pl.plot(xt,x)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Distance over Time")

total_time = xt[-1] - xt[0]
number_of_oscillations_in_plot = 7
T_fromPlot = total_time / number_of_oscillations_in_plot
print("\nFrom the plotted data, the period of the oscillation is ", T_fromPlot," seconds.")