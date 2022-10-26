#NICHOLAS GARCIA & MARTA GONCZAR
"""OSCILLATING MASS ON A SPRING"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from math import log10
from math import floor
from numpy import log
from numpy import zeros
import scipy.optimize as so
from numpy import sin

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
    bareEnergy[i] = (1/2)*bareMass*(bareVy[i]**2)+(1/2)*bareK*(bareY[i]**2)

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

dampedXt, dampedX, uDampedX = np.loadtxt("dampedx.txt", unpack=True)

udampedXt = 0.005
dampedMass = 217.6 #g
dampedMass = dampedMass / 1000; #kg
dampedPeriod = (dampedXt[-1] - dampedXt[0]) / 7; #seconds / oscillations = period
uDampedPeriod = 0.01 #Because 0.01 is the distance between data points, and we do not know if the recorded maxima are actually at the respective times, as the true maximas could occur between data points
dampedFrequency = 1/dampedPeriod
uDampedFrequency = round_it((dampedFrequency * uDampedPeriod / dampedPeriod),1)
dampedOmega0 = (2*np.pi)/dampedPeriod #1/s
uDampedOmega0 = round_it((2 * np.pi * uDampedFrequency),1)
dampedK = (dampedOmega0**2)*dampedMass
uDampedK = round_it(((2**(1/2))*dampedOmega0*uDampedOmega0),1)

#Using Forward Euler Method to get Plottable Data
dampedY = zeros(len(dampedXt/dt))
dampedVy = zeros(len(dampedXt/dt))
dampedV0 = 0
dampedY0 = find_Amplitude(dampedX)
uDampedY0 = 0.0001 #m, because 0.0001 is the difference between different values that could be the amplitude
dampedY[0]=dampedY0
dampedVy[0] = dampedV0
t0 = 0
for i in np.arange(0,len(dampedY)-1,1):
    dampedY[i+1] = dampedY[i] + dt*dampedVy[i]
    dampedVy[i+1] = dampedVy[i] - dt*(dampedOmega0**2)*dampedY[i]
    
#Calculating energy over time
dampedEnergy = zeros(len(dampedXt/dt))
for i in np.arange(0,len(dampedY),1):
    dampedEnergy[i] = (1/2)*dampedMass*(dampedVy[i]**2)+(1/2)*dampedK*(dampedY[i]**2)
    
pl.cla()
pl.plot(dampedXt,dampedX, ".")
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - MEASURED - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedXt,dampedY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - FORWARD - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedXt,dampedVy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - FORWARD -  Velocity over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedVy,dampedY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - FORWARD - Velocity over Position for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedXt,dampedEnergy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - FORWARD - Energy over Time for Non-Damped Oscillation")

dampedPeriod = ((5682 - 5155)/100) / 7 #Difference in sample numbers at 100 samples per second divided by the number of oscillations in the time interval gives Period

#converting from cm to m
dampedX = dampedX / 100 
uDampedX = uDampedX / 100

#Using Symplectic Euler Method to get Plottable Data
dampedY = zeros(len(dampedXt/dt))
dampedVy = zeros(len(dampedXt/dt))
dampedV0 = 0
dampedY0 = find_Amplitude(dampedX)
dampedY[0]=dampedY0
dampedVy[0] = dampedV0
t0 = 0
for i in np.arange(0,len(dampedY)-1,1):
    dampedY[i+1] = dampedY[i] + dt*dampedVy[i]
    dampedVy[i+1] = dampedVy[i] - dt*(dampedK/dampedMass)*dampedY[i+1]

pl.cla()
pl.plot(dampedXt,dampedY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - SYMPLECTIC - Position over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedXt,dampedVy)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - SYMPLECTIC -  Velocity over Time for Non-Damped Oscillation")
pl.cla()
pl.plot(dampedVy,dampedY)
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("Session 2 - SYMPLECTIC - Velocity over Position for Non-Damped Oscillation")

def find_Amplitude(index,position_array):
    """Finds the amplitude of a sinusoidal function based on extrema on a given time interval"""
    net_displacement = abs(max(position_array[index:index+150]) - min(position_array[index:index+150]))
    A = net_displacement / 2
    return A;

def find_Amplitude_time(index,position_array):
    """Finds the time of an aimpplitude on an interval"""
    peak = abs(max(position_array[index:index+10]))
    return peak;

#determining Gamma coefficient
initialDampedAmplitude = find_Amplitude(0,dampedX);
dampedAmplitude1 = find_Amplitude(100,dampedX);
dampedAmplitude2 = find_Amplitude(2000,dampedX);
dampedAmplitude3 = find_Amplitude(4000,dampedX);
dampedAmplitude4 = find_Amplitude(8000,dampedX);
dampedAmplitude5 = find_Amplitude(10000,dampedX);
dampedAmplitude6 = find_Amplitude(12000,dampedX);
uDampedAmplitude = round_it(((uDampedX + uDampedX)/2)[1],1)  

initialDampedAmplitude = find_Amplitude(0,dampedX);
dampedAmplitude1 = find_Amplitude_time(100,dampedX);
dampedAmplitude2 = find_Amplitude_time(2000,dampedX);
dampedAmplitude3 = find_Amplitude_time(4000,dampedX);
dampedAmplitude4 = find_Amplitude_time(8000,dampedX);
dampedAmplitude5 = find_Amplitude_time(10000,dampedX);
dampedAmplitude6 = find_Amplitude_time(12000,dampedX);
uDampedAmplitude = round_it(((uDampedX + uDampedX)/2)[1],1)    

print(round(round_it(dampedAmplitude1, 6),30))
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



print("For the BARE MASS:")
print("Angular frequency is ",round(bareOmega0,1)," +- ",uBareOmega0," s^-1")
print("Amplitude is ",round(bareY0,4)," +- ", uBareY0, " m")
print("Frequency is  ",round(bareFrequency,1)," +- ",uBareOmega0," Hz")
print("Spring constant is ",round(bareK,0)," +- ",uBareK, " kg/s/s")
#rint("Gamma constant is ",round(gamma,20)," +- ",uGamma)

Amplitudes = zeros(len(dampedXt))
for i in np.arange(0,12000,1):
    Amplitudes[i] = find_Amplitude(i,dampedX)
    
def gammaModel(t, gamma):
    return Amplitudes[0]*np.exp(t * -gamma)
poptLin, pcovLin = so.curve_fit(gammaModel,dampedXt,Amplitudes)

pl.cla()
#pl.plot(dampedXt,dampedX, ".")
pl.plot(dampedXt[0:11900],Amplitudes[0:11900]) #0.26625
pl.plot(dampedXt,gammaModel(dampedXt,poptLin[0]))
pl.xlabel("Time (s)")
pl.ylabel("Distance (m)")
pl.savefig("GAMMA DUMMY")

print("Gamma coefficient is ", poptLin[0])