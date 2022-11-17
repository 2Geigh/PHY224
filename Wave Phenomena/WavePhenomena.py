"NICK GARCIA & MARTA GONCZAR"
"""WAVE PHENOMENA"""
#2022/11/15

import numpy as np
from numpy import sin
from numpy import tan
from numpy import pi
import matplotlib.pyplot as pl

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
xslitSeperation=1/slitSeperation
angularSpread = np.array([34.21,41.72,50.47,58.86])
angularSpreadwidth= sin(angularSpread)
uAngularSpread = 0.15 * angularSpreadwidth
uslitsep=0.1
##################
#angularSpread = angularSpread * pi / 180
#uAngularSpread = 10
#################
a = slitSeperation
uA = 0.1 #cm
th = angularSpread
uSinth = sin(th)*uAngularSpread * (1/tan(th))
################
pl.cla()
pl.errorbar(xslitSeperation, angularSpreadwidth, xerr=uAngularSpread, yerr=uslitsep, fmt=" ")
pl.plot(xslitSeperation, angularSpreadwidth, ".")
pl.ylabel("Sin of the Angular Spread")
pl.xlabel("Slit seperation (cm)")
pl.savefig("Experiment 4 -- Slit sepration over Sine of Angular Spread");

#difraction 2.0





"INTERFERENCE"
"""TRIAL 1"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
k = 2 * pi / wavelength
uK = k * uWavelength / wavelength
th = np.array([0.15,11.65,21.37,31.29])
th = th * pi / 180 
uTh = th * 0.15
d= 6.08
uD = 1
###############
m = np.array([1,2,3,4])
x = np.arange(th[0],th[-1],0.01)
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
pl.cla()
pl.plot(x,k*d*sin(x))
pl.errorbar(th,k*d*sin(th)/2, xerr=uTh, yerr=uY, fmt=" ")
pl.plot(th,y, ".")
pl.xlabel("Angle of direction from inter-slit midpoint (radians)")
pl.ylabel("Sine of angle times separation times wave number")
pl.savefig("Experiment 5 -- TRIAL 1 - 608 cm")

"""TRIAL 2"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
k = 2 * pi / wavelength
######################
uK = k * uWavelength / wavelength
th = np.array([2.86,3.39,7.81,13.58])
th = th * pi / 180 
uTh = th * 0.15
d= 12.58
uD = 1
###############
m = np.array([1,2,3,4])
x = np.arange(th[0],th[-1],0.01)
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
pl.cla()
pl.plot(x,k*d*sin(x))
pl.errorbar(th,k*d*sin(th)/2, xerr=uTh, yerr=uY, fmt=" ")
pl.plot(th,y, ".")
pl.xlabel("Angle of direction from inter-slit midpoint (radians)")
pl.ylabel("Sine of angle times separation times wave number")
pl.savefig("Experiment 5 -- TRIAL 2 - 12 cm")

"""TRIAL 3"""
f = 20 #Hz
uF = 0.05
wavelength = 0.972
uWavelength = wavelength * 0.15
v = f*wavelength
uV = np.sqrt((uWavelength/wavelength)**2+(uF/f)**2)
k = 2 * pi / wavelength
uK = k * uWavelength / wavelength
######################
th = np.array([5.41,15.64,26.43,38.09])
th = th * pi / 180 
uTh = th * 0.15
d= 6.08
uD = 1
###############
m = np.array([1,2,3,4])
x = np.arange(th[0],th[-1],0.01)
uKD2 = np.sqrt(((uK/k)**2+(uD/d)**2))/2
uSinth = uTh * (sin(th))*(1/tan(th))
y = k * d * sin(th) / 2
uY = np.sqrt(((uKD2/(k*d/2))**2+(uSinth/(sin(th)))**2))
#####################
pl.cla()
pl.plot(x,k*d*sin(x))
pl.errorbar(th,k*d*sin(th)/2, xerr=uTh, yerr=uY, fmt=" ")
pl.plot(th,y, ".")
pl.xlabel("Angle of direction from inter-slit midpoint (radians)")
pl.ylabel("Sine of angle times separation times wave number")
pl.savefig("Experiment 5 -- TRIAL 3 - 625 cm")