"""NICHOLAS GARICA & MARTA GONCZARIS"""
#_________________________________________________
"""Ohm's Law"""
#2022 / 9 / 22

import numpy as np
import scipy as sp
import scipy.optimize as so
import matplotlib.pyplot as pl

V1, uV1, I1, uI1, R1, uR1 = np.loadtxt("Ohm DATA 1.csv", delimiter=",", unpack=True);

#Creating a line of best fit
def model(x,a,b):
    return (a*x) + b;

def R_reciprocal(x):
    return (9.52727262 * x) + 0.01237408;

so.curve_fit(model, V1, I1)
def residue(y,x):
    return y - R_reciprocal(x)

y0 = 0

#First, we plot V over I with uncertainty
pl.subplot(2, 1, 2);
pl.plot(V1, R_reciprocal(V1))
pl.plot(V1,I1, ".");
pl.errorbar(V1, I1, xerr=uV1, yerr=uI1, fmt="none");
pl.xlabel("Voltage (V)");
pl.ylabel("Current (mA)")

pl.subplot(2,1,1)
pl.plot(V1, residue(I1,V1), ".")
pl.ylabel("Residuals")
"""pl.title("Current as a function of Voltage of a circuit with ~100 Ohms of Resistance");"""
pl.errorbar(V1, residue(I1,V1), xerr=uV1, yerr=uI1, fmt="none");

pl.savefig("OHM Figure 1",dpi=300,bbox_inches="tight")
pl.show();

#DATA 1
#_____________________________________________________________________________
#DATA 2

V2, uV2, I2, uI2, R2, uR2 = np.loadtxt("Ohm DATA 2.csv", delimiter=",", unpack=True);

#Creating a line of best fit
def model(x,a,b):
    return (a*x) + b;

def R_reciprocal(x):
    return (2.40337662e+01 * x) + -5.68657996e-03;

so.curve_fit(model, V2, I2)
def residue(y,x):
    return y - R_reciprocal(x)


sum = 0
i = 0
while i<len(V2):
    sum = sum + ((residue(I2[i],V2[i]))**2)/(R_reciprocal(I2[i]))
    i = i + 1
X2 = sum
print("#############################")
print(sum)


#First, we plot V over I with uncertainty
pl.subplot(2, 1, 2);
pl.plot(V2, R_reciprocal(V2))
pl.plot(V2,I2, ".");
pl.errorbar(V2, I2, xerr=uV2, yerr=uI2, fmt="none");
pl.xlabel("Voltage (V)");
pl.ylabel("Current (mA)")

pl.subplot(2,1,1)
pl.plot(V2, residue(I2,V2), ".")
pl.ylabel("Residuals")
"""pl.title("Current as a function of Voltage of a circuit with ~100 Ohms of Resistance");"""
pl.errorbar(V2, residue(I2,V2), xerr=uV2, yerr=uI2, fmt="none");

pl.savefig("OHM Figure 2",dpi=300,bbox_inches="tight")
pl.show();