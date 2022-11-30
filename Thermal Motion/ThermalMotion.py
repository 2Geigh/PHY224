"""Nicholas Garcia & Marta Gonczar"""
"""THERMAL MOTION EXPERIMENT"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def model(x, m):
    return m * x

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    return np.sum( (y_measure - y_predict)**2 / errors**2 )/(y_measure.size - number_of_parameters)

print("ACTIVITY 1")

beadDiameter = 1.9e-6 #m
uBeadDiameter = 0.1e-6 #m
viscosity = 1e-3 #kg/m-s
temperature = 296.5 #K
uTemperature = 0.5 #K
gasConstant = 8.31446261815324
micrometresPerPixel = 0.12048

x_position, y_position = np.loadtxt("data7.txt", delimiter="\t", unpack=True)
uPosition = 0.005
uPositionMicrometresPerPixel = 0.003

x_displacement = x_position - x_position[0]
y_displacement = y_position - y_position[0]
uX_displacement = np.sqrt(((uPosition/x_position)**2) + ((uPosition/x_position[0])**2))
uY_displacement = np.sqrt(((uPosition/y_position)**2) + ((uPosition/y_position[0])**2))

uX_displacement = np.sqrt(((uX_displacement/x_displacement)**2) + ((uPositionMicrometresPerPixel/micrometresPerPixel)**2))
uY_displacement = np.sqrt(((uY_displacement/y_displacement)**2) + ((uPositionMicrometresPerPixel/micrometresPerPixel)**2))
x_displacement = x_displacement * micrometresPerPixel
y_displacement = y_displacement * micrometresPerPixel

x_displacement_squared = (x_displacement)**2
y_displacement_squared = (y_displacement)**2
uX_displacement_squared = x_displacement * 2 * uX_displacement
uY_displacement_squared = y_displacement * 2 * uY_displacement

r2 = x_displacement_squared + y_displacement_squared
r2err = np.sqrt(((uX_displacement_squared)**2) + ((uY_displacement_squared)**2))
r2err[0] = r2err[1]
#r2err = np.array([30 for i in r2])

   
t = np.arange(0,60,0.5)

popt, pcov = curve_fit(model, t, r2)
pstd = np.sqrt(np.diag(pcov))

print(popt,pstd)

print("chi2r=",chi2reduced(r2, model(t, *popt), r2err, popt.size))

plt.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Time [s]", fontsize=15)
ax.set_ylabel(r"Mean Square Displacement [$\mu$m$^2$]", fontsize=15)
#ax.set_xlim()
ax.tick_params(axis='both', which='both',labelsize=15)

ax.errorbar(t, r2, yerr=r2err, marker='.', ms=6, ls='', lw=1, capsize=2, color='b', label="Data")
x = np.linspace(0, 65, 1000)
ax.plot(x, model(x, *popt), 'b-', label="Fit")
ax.legend(prop={'size':15})

fig.tight_layout()
plt.savefig("Activity 1 - Line Plot.png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=600)
plt.show()

D = popt[0]/4
D = D * (10**(-12))
uD = (pstd[0]/4)*(10**(-12))
print("D = ", D," +- ", pstd[0]/4)

gamma = 6 * np.pi * viscosity * (beadDiameter / 2)
avogadro = temperature * gasConstant / (D * gamma)

uGamma = 3 * np.pi * viscosity * uBeadDiameter
uOneOverGamma = uGamma / (gamma**2)
uOneOverD = uD / (D**2)
uOneOverDGamma = np.sqrt(((uOneOverD/(1/D))**2)+((uOneOverGamma/(1/gamma))**2))
uAvogadro = gasConstant * (np.sqrt(((uTemperature/temperature)**2)+((uOneOverDGamma/((1/D)*(1/gamma)))**2)))

print(avogadro," +- ",uAvogadro," number of molecules/mole")
print("______")
print("ACTIVITY 2")
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

import scipy.stats as ss

steps = []

temp_x, temp_y = np.loadtxt("data1.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data2.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data3.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data4.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data5.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data6.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data7.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data8.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data9.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

temp_x, temp_y = np.loadtxt("data10.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data11A.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data11B.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data12.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data13.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data14.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data15.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data16.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data17.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data18.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data19.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)
temp_x, temp_y = np.loadtxt("data20.txt", delimiter="\t", unpack=True)
temp_step = np.array([np.sqrt((temp_x[i+1] - temp_x[i])**2 + (temp_y[i+1] - temp_y[i])**2) for i in range(len(temp_x)-1)])

steps.append(temp_step)

uMeanSteps = np.mean(steps) / np.sqrt(len(steps))

mu = np.mean(steps) - 2.5
uMu = uMeanSteps

c = np.sqrt(mu) - 0.5
uC = np.sqrt(mu) * uMu / mu

values, bins = np.histogram(steps, bins=40, density=True)
bins_c = (bins[1:] + bins[:1])/2
widths = bins[1] - bins[0]

plt.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel(r"Step-size [$\mu$m]", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
#ax.set_xlim()
ax.tick_params(axis='both',which='both',labelsize=15)

ax.bar(bins_c, values, width=0.4*widths, label="Step-size Data", color='b')
x = np.linspace(bins[0]-widths, bins[-1]+widths,1000)
ax.plot(x,ss.norm.pdf(x,mu,c),'b-',label="Gaussian fit")
ax.legend(prop={'size':15})
plt.savefig("Activity 1 - Histogram.png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=600)
plt.show()

fig.tight_layout()
#plt.savefig("Img2.png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=600)

print(c**2)

# 2c^2 = 4Dt
# D = c^2/2t
# t = 0.5

D = c**2
uD = (c**2) * 2 * uC / c
D = D * (10**(-12))
uD = uD * (10**(-12))

gamma = 6 * np.pi * viscosity * (beadDiameter / 2)
avogadro = temperature * gasConstant / (D * gamma)

uGamma = 3 * np.pi * viscosity * uBeadDiameter
uOneOverGamma = uGamma / (gamma**2)
uOneOverD = uD / (D**2)
uOneOverDGamma = np.sqrt(((uOneOverD/(1/D))**2)+((uOneOverGamma/(1/gamma))**2))
uAvogadro = gasConstant * (np.sqrt(((uTemperature/temperature)**2)+((uOneOverDGamma/((1/D)*(1/gamma)))**2)))

print(avogadro," +- ",uAvogadro," number of molecules/mole")
