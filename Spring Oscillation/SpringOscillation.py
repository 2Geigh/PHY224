#NICHOLAS GARCIA & MARTA GONCZAR
"""OSCILLATING MASS ON A SPRING"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from math import log10
from math import floor
from numpy import log
from numpy import zeros

#function to round to a number of significant digits
def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1);

def find_Amplitude(position_array):
    """Finds the amplitude of a sinusoidal function based on extrema on a given time interval"""
    net_displacement = abs(max(position_array) - min(position_array))
    A = net_displacement / 2
    return A;

bareXt, bareX, uBareX = np.loadtxt("bare_x.txt", unpack=True)
bareVt, bareV = np.loadtxt("bare_v.txt", unpack=True)

#converting from cm to m
bareX = bareX / 100 
uBareX = uBareX / 100
bareV = bareV / 100 

sampleRate = 100 #samples per second
dt = 1/sampleRate

uBareXt = 0.005
bareMass = 200.7 #g
bareMass = bareMass / 1000; #kg
barePeriod = (bareXt[-1] - bareXt[0]) / 7; #seconds / oscillations = period
uBarePeriod = 0.01 #Because 0.01 is the distance between data points, and we do not know if the recorded maxima are actually at the respective times, as the true maximas could occur between data points
bareFrequency = 1/barePeriod
uBareFrequency = round_it((bareFrequency * uBarePeriod / barePeriod),1)
bareOmega0 = (2*np.pi)/barePeriod #1/s
uBareOmega0 = round_it((2 * np.pi * uBareFrequency),1)
bareK = (bareOmega0**2)*bareMass
uBareK = round_it(((2**(1/2))*bareOmega0*uBareOmega0),1)

#Using Forward Euler Method to get Plottable Data
bareY = zeros(len(bareXt/dt))
bareVy = zeros(len(bareXt/dt))
bareV0 = 0
bareY0 = find_Amplitude(bareX)
uBareY0 = 0.0001 #m, because 0.0001 is the difference between different values that could be the amplitude
bareY[0]=bareY0
bareVy[0] = bareV0
t0 = 0
for i in np.arange(0,len(bareY)-1,1):
    bareY[i+1] = bareY[i] + dt*bareVy[i]
    bareVy[i+1] = bareVy[i] - dt*(bareOmega0**2)*bareY[i]
    
#Calculating energy over time
bareEnergy = zeros(len(bareXt/dt))
for i in np.arange(0,len(bareY),1):
    bareEnergy[i] = (1/2)*bareMass*(bareV[i]**2)+(1/2)*bareK*(bareY[i]**2)

pl.cla()
pl.plot(bareXt,bareX, ".")
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - MEASURED - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(bareXt,bareY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - FORWARD - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(bareXt,bareVy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - FORWARD -  Velocity over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(bareVy,bareY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - FORWARD - Velocity over Position for Non-Damped Oscillation")
pl.cla()
pl.plot(bareXt,bareEnergy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - FORWARD - Energy over Time for Non-Damped Oscillation")

#Using Symplectic Euler Method to get Plottable Data
bareY = zeros(len(bareXt/dt))
bareVy = zeros(len(bareXt/dt))
bareV0 = 0
bareY0 = find_Amplitude(bareX)
bareY[0]=bareY0
bareVy[0] = bareV0
t0 = 0
for i in np.arange(0,len(bareY)-1,1):
    bareY[i+1] = bareY[i] + dt*bareVy[i]
    bareVy[i+1] = bareVy[i] - dt*(bareK/bareMass)*bareY[i+1]
    
pl.cla()
pl.plot(bareXt,bareY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - SYMPLECTIC - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(bareXt,bareVy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - SYMPLECTIC -  Velocity over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(bareVy,bareY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 1 - SYMPLECTIC - Velocity over Position for Non-Damped Oscillation")

print("For the BARE MASS:")
print("Angular frequency is ",round(bareOmega0,1)," +- ",uBareOmega0," s^-1")
print("Amplitude is ",round(bareY0,4)," +- ", uBareY0, " m")
print("Frequency is  ",round(bareFrequency,1)," +- ",uBareOmega0," Hz")
print("Spring constant is ",round(bareK,0)," +- ",uBareK, " kg/s/s")

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

total_time = bareXt[-1] - bareXt[0]
number_of_oscillations_in_plot = 7
T_fromPlot = total_time / number_of_oscillations_in_plot
print("\nFrom the plotted data, the period of the oscillation is ", T_fromPlot," seconds.")

def find_Amplitude(index,position_array):
    """Finds the amplitude of a sinusoidal function based on extrema on a given time interval"""
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