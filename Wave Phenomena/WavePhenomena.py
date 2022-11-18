"NICK GARCIA & MARTA GONCZAR"
"""WAVE PHENOMENA"""
#2022/11/15

import numpy as np
from numpy import sin
from numpy import tan
from numpy import pi
import matplotlib.pyplot as pl
import scipy.optimize as so
from scipy.optimize import curve_fit

"WAVE SPEED"
frequency= [5,10,12,15,20]
avg_wavelength=[4.117,1.9,1.59,1.27,0.972]
uavg_wavelength=[0.4117,0.19,0.2385,0.1905,0.1458]
pl.cla()
pl.plot(frequency,avg_wavelength,".")
pl.errorbar(frequency,avg_wavelength, yerr=uavg_wavelength, fmt="none");
pl.xlabel("Frequency (Hz)");
pl.ylabel("Average Wavelength (cm)")
pl.savefig("ACTIVITY 2")




"DIFRACTION"
f = 20 #Hz
###################
slitSeperation = np.array([3.0,1.0,0.50,0.01])
angularSpread = np.array([34.21,41.72,50.47,58.86])
uAngularSpread = 0.15 * angularSpread
wavelength = np.array([0.972,0.972,0.972,0.972])
uWavelength = 0.15 * wavelength
##################
angularSpread = angularSpread * pi / 180
uAngularSpread = angularSpread * pi / 180
#################
a = slitSeperation
uA = 0.1 #cm
th = angularSpread
uTh = uAngularSpread
uSinth = sin(th)*uAngularSpread * (1/tan(th))
print(uSinth)
################
def model(x, slope, b):
    return (slope * x)+b;
prpv, pcov = so.curve_fit(model, sin(th), wavelength)
#Computing Chi-Squared
X2 = 0
i = 0
while i < len(sin(th)):
    X2 = X2 + ((((wavelength[i]-model(sin(th),prpv[0],prpv[1])[i])/uWavelength[i])**2)/(model(sin(th),prpv[0],prpv[1])[i]))
    i = i + 1
print("Chi-Squared is ",X2," for the Non-Linear Regression Model.")
if X2 < 1:
    print("Thus, this model is over-fit.\n")
if X2 > 10:
    print("Thus, the model is a poor fit.\n")

if X2 > 1 and X2 < 5:
    print("Thus, the model is an incomplete fit.\n")
################
pl.cla()
pl.plot(sin(th),model(sin(th),prpv[0],prpv[1]))
pl.errorbar(sin(th), wavelength, xerr=uSinth, yerr=uWavelength, fmt=" ")
pl.plot(sin(th),wavelength, ".")
pl.ylabel("Wavelength (cm)")
pl.xlabel("Sine of Angular spread")
pl.legend(['Measured values with uncertainty','Prediction from Regression'])
pl.savefig("Experiment 4");

print("slope is ",prpv[0])
print("chisquared is ", X2)

#a sin = l

#sint = l/a



"INTERFERENCE"
"""TRIAL 1"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
######################
k = 2 * pi / wavelength
uK = k * uWavelength / wavelength
d= 6.08
uD = 1
th = np.array([0.15, 11.65, 21.37, 31.29])
uTh = th * 0.15
###############
m = np.array([1,2,3,4])
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
print(uY[-1])
pl.cla()
pl.errorbar(m,y, yerr=uY, fmt=" ")
pl.plot(m,y, "o")
#pl.savefig("Experiment 5 -- TRIAL 1 - 608 cm")

"INTERFERENCE"
"""TRIAL 2"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
######################
k = 2 * pi / wavelength
uK = k * uWavelength / wavelength
####################
d= 12.58
uD = 1
th = np.array([2.86, 3.39, 7.81, 13.58])
uTh = th * 0.15
###############
m = np.array([1,2,3,4])
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
print(uY[-1])
pl.errorbar(m,y, yerr=uY, fmt=" ")
pl.plot(m,y, "o")

"INTERFERENCE"
"""TRIAL 3"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
######################
k = 2 * pi / wavelength
uK = k * uWavelength / wavelength
####################
d= 6.25
uD = 1
th = np.array([5.41, 15.64, 26.43, 38.09])
uTh = th * 0.15
###############
m = np.array([1,2,3,4])
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
print(uY[-1])
pl.errorbar(m,y, yerr=uY, fmt=" ")
pl.plot(m,y, "o")

pl.xlabel("Maxima order")
pl.ylabel("sin(θ) * d * k")
pl.legend(['d = 6.08 cm','d = 12.58 cm','d = 6.25 cm'])
pl.savefig("Activity 5")