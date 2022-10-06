"""NICHOLAS GARICA & MARTA GONCZARIS"""
#_________________________________________________
"""Power Law"""
#2022 / 9 / 27

import numpy as np
import scipy as sp
import scipy.optimize as so
import matplotlib.pyplot as pl
from pylab import yscale
from pylab import xscale
from pylab import loglog
import pylab as pyl


V, uV, I, uI = np.loadtxt("data3.csv", delimiter=",", unpack=True);

logV = np.log(V[1:]);
loguV = uV / V
#loguV = loguV[1:];
logI = np.log(I[1:]);
loguI = uI / I
#loguI = loguI[1:];
b_theoretical = 3/5;
a_theoretical = 7; #chosen arbitrarily to roughly fit with experimental data

#Creating a line of best fit
def linear(x,a,b):
    return (x*a) + b;
def power(x,a,b):
    return a*(x**b);

popt_lin, pcov_lin = so.curve_fit(linear, logV, logI)
popt_nonlin, pcov_nonlin = so.curve_fit(power, V, I)

def powerlaw_relation_linear(x):
    return linear(x,popt_lin[0],popt_lin[1]);

def powerlaw_relation_nonlinear(x):
    return power(x, popt_nonlin[0], popt_nonlin[1]);

def residue_lin(y,x):
    return y - powerlaw_relation_linear(x)

def residue_nonlin(y,x):
    return y - powerlaw_relation_nonlinear(x)

pl.figure(1)
pl.subplot(2, 1, 2);
pl.plot(logV,logI,"x")
pl.plot(logV, powerlaw_relation_linear(logV));
pl.errorbar(np.log(V), np.log(I), xerr=loguV, yerr=loguI, fmt="none");
pl.plot(np.log(V), np.log(power(V,a_theoretical,b_theoretical))) #theoretical
pl.xlabel("Log of Voltage (V)");
pl.ylabel("Log of Current (mA)")
pl.legend(['Measured values','Linear Regression','Theoretical Curve'])

pl.subplot(2, 1, 1);
pl.plot(logV,residue_lin(logI, logV),"x")
pl.errorbar(logV, residue_lin(logI,logV), yerr=1, fmt=" ");
pl.ylabel("Residue")

pl.savefig("powerplot LIN",dpi=300,bbox_inches="tight")
pl.plot()
pl.show();


pl.figure(2)
pl.subplot(2,1,2)
pl.plot(V,I,"x")
pl.plot(V, powerlaw_relation_nonlinear(V))
pl.errorbar(V, I, xerr=uV, yerr=uI, fmt="none");
pl.plot(V, power(V,a_theoretical,b_theoretical)) #theoretical
pl.ylabel("Current (mA)")
pl.xlabel("Voltage (V)")
pl.legend(['Measured values','Non-Linear Regression', "Theoretical Curve"])

pl.subplot(2, 1, 1);
pl.plot(V,residue_nonlin(I, V),"x")
pl.errorbar(V, residue_nonlin(I,V), yerr=uI, fmt=" ");
pl.ylabel("Residue")

pl.tight_layout()
pl.savefig("powerplot NONLIN",dpi=300,bbox_inches="tight")
pl.show();



pl.figure(3)
pl.plot(np.log(V),np.log(I),".") #raw data
pl.errorbar(np.log(V), np.log(I), xerr=loguV, yerr=loguI, fmt=" ") #error
pl.plot(np.log(V), np.log(power(V, popt_nonlin[0], popt_nonlin[1]))) #power fit
pl.plot(np.log(V), linear(np.log(V), popt_lin[0], popt_lin[1])) #linear fit
pl.plot(np.log(V), np.log(power(V,a_theoretical,b_theoretical))) #theoretical

#pl.errorbar(V, I, xerr=uV, yerr=uI, fmt="none");
pl.ylabel("Log of Current (mA)")
pl.xlabel("Log of Voltage (V)")
pl.legend(['Measured values with uncertainty','Non-Linear Regression','Linear Regression','Theoretical Prediction'])

pl.tight_layout()
pl.savefig("powerplot FINAL",dpi=300,bbox_inches="tight")
pl.show();

lin_sigma_a = (pcov_lin[0,0])**(1/2);
lin_sigma_b = (pcov_lin[1,1])**(1/2);

nonlin_sigma_a = (pcov_nonlin[0,0])**(1/2)
nonlin_sigma_b = (pcov_nonlin[1,1])**(1/2)

sigma_a = (lin_sigma_a + nonlin_sigma_a) / 2;
sigma_b = (lin_sigma_b + nonlin_sigma_b) / 2;

def expected(x):
    return powerlaw_relation_linear(x);


i = 0
sum = 0
for i in range(1,len(V),1):
    sum = sum + ((((np.log(I[i])) - linear(np.log(V[i]), popt_lin[0], popt_lin[1]))**2)/linear(np.log(V[i]), popt_lin[0], popt_lin[1]))
    
print("#############################")
print("#############################")
print("#############################")
print("The Chi-Squared value is ",sum);
    
    
    
    