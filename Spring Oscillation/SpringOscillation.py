#NICHOLAS GARCIA & MARTA GONCZAR
"""OSCILLATING MASS ON A SPRING"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from math import log10
from math import floor
from numpy import log

#function to round to a number of significant digits
def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1);

bareMass = 200.7 #g

bareMass = bareMass / 1000; #kg
barePeriod = 5.11 / 7; #seconds / oscillations = period

xt, x, ux = np.loadtxt("bare_x.txt", unpack=True)

#converting from cm to m
x = x / 100 
ux = ux / 100

pl.cla()
pl.plot(xt,x)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Distance over Time for Bare Oscillation")

total_time = xt[-1] - xt[0]
number_of_oscillations_in_plot = 7
T_fromPlot = total_time / number_of_oscillations_in_plot
print("\nFrom the plotted data, the period of the bare oscillation is ", T_fromPlot," seconds.")

#________________________________________________________

#mass with "bob" attached for damping
dampedMass = 217.6 #g
dampedMass = dampedMass / 1000; #kg

dampedPeriod = ((5682 - 5155)/100) / 7 #Difference in sample numbers at 100 samples per second divided by the number of oscillations in the time interval gives Period

dampedXt, dampedX, uDampedX = np.loadtxt("dampedx.txt", unpack=True)

#converting from cm to m
dampedX = dampedX / 100 
uDampedX = uDampedX / 100

pl.cla()
pl.plot(dampedXt,dampedX)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Distance over Time for Damped Oscillation")

total_time = xt[-1] - xt[0]
number_of_oscillations_in_plot = 7
T_fromPlot = total_time / number_of_oscillations_in_plot
print("\nFrom the plotted data, the period of the oscillation is ", T_fromPlot," seconds.")

def find_Amplitude(index,position_array):
    net_displacement = abs(max(position_array[index:index+8]) - min(position_array[index:index+8]))
    A = net_displacement / 2
    return A;

#determining Gamma coefficient
initialDampedAmplitude = find_Amplitude(0,dampedX);
dampedAmplitude1 = find_Amplitude(1200,dampedX);
dampedAmplitude2 = find_Amplitude(2000,dampedX);
dampedAmplitude3 = find_Amplitude(4000,dampedX);
dampedAmplitude4 = find_Amplitude(8000,dampedX);
dampedAmplitude5 = find_Amplitude(10000,dampedX);
dampedAmplitude6 = find_Amplitude(12000,dampedX);
uDampedAmplitude = round_it(((uDampedX + uDampedX)/2)[1],1)    

print(round(round_it(dampedAmplitude1, 6),3))
print(round(round_it(dampedAmplitude2, 6),3))
print(round(round_it(dampedAmplitude3, 6),3))
print(round(round_it(dampedAmplitude4, 6),3))
print(round(round_it(dampedAmplitude5, 6),3))
print(round(round_it(dampedAmplitude6, 6),3))

gamma = []
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(1200*0.01)))
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(2000*0.01)))
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(4000*0.01)))
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(8000*0.01)))
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(10000*0.01)))
gamma.append(-(log(abs(dampedAmplitude1/initialDampedAmplitude))/(12000*0.01)))
gamma = np.mean(gamma)