#RADIOACTIVE DECAY
#NICHOLAS GARCIA AND MARTA GONCZAR
#2022 / 9 / 29


from numpy import exp
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import scipy.optimize as so
import matplotlib.pyplot as pl
from numpy import log

machineData = np.loadtxt("Barium.txt", delimiter="\t");
dt = 20 #seconds
sampleNumber = machineData[:,0]; #x
totalCount = machineData[:,1]; #y
uTotalCount = (totalCount)**(1/2);

backgroundData = np.loadtxt("Background.txt", delimiter="\t");
backgroundSample = backgroundData[:,0];
backgroundCount = backgroundData[:,1];

backgroundMean = 0
for i in backgroundCount:
    backgroundMean = backgroundMean + i/len(backgroundSample);
uBackgroundMean = np.std(backgroundCount) / ((len(backgroundSample))**(1/2))

sampleCount = totalCount - backgroundMean;
uSampleCount = (((uTotalCount)**2)+((uBackgroundMean)**2))**(1/2)
sampleRate = sampleCount / dt
uSampleRate = ((sampleCount)**(1/2))/dt

#Instructions say Barium has half-life of 2.6 minutes

t = np.arange(20, 1220, 20) #Each count was a 20 second interval up to 20 minutes

def model_I_nonlin(x,A,TH): #t,I0,t_half
    return A * ((1/2)**(x/TH));

def model_I_lin(x,a,th): #t,I0,t_half
    #ln(I) =
    return (x*th) + a;

def uncertainty_nonlin(A, uA):
    A = np.array(A)
    uA = np.array(uA)
    return abs((uA)/(A))

poptLin, pcovLin = so.curve_fit(model_I_lin, t, log(abs(sampleRate)))
poptNonLin, pcovNonLin = so.curve_fit(model_I_nonlin, t, sampleRate)

linHalfLife = (abs(poptLin[1]))**(-1);
nonLinHalfLife = poptNonLin[1];
theoreticalHalfLife = 2.6 * 60;

def I_nonlin(x):
    return poptNonLin[0] * ((1/2)**(x/poptNonLin[1]))

def I_lin(x):
    return (x * poptLin[1]) + poptLin[0]

def I_nonlin_theoretical(x):
    return poptNonLin[0] * ((1/2)**(x/theoreticalHalfLife))

def I_lin_theoretical(x):
    return (x * (-1/theoreticalHalfLife)*log(2)) + poptLin[0]

pl.cla()
pl.figure(1)
pl.plot(t, sampleRate, ".")
pl.plot(t, I_nonlin(t))
pl.plot(t, I_nonlin_theoretical(t))
pl.xlabel("Time (s)")
pl.ylabel("Decay rate (1/s)")
pl.errorbar(t, sampleRate, yerr=uSampleRate, fmt=" ")
pl.legend(['Measured values with uncertainty','Prediction from Regression', 'Theoretical Curve'])
pl.savefig("Decay Rate over Time with Uncertainty and Non-Linear Fit",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();

pl.figure(2)
pl.cla()
pl.plot(t, log(sampleRate), ".")
pl.errorbar(t, log(sampleRate), yerr=uSampleRate/sampleRate, fmt=" ")
pl.plot(t, I_lin(t))
pl.plot(t, I_lin_theoretical(t))
pl.xlabel("Time (s)")
pl.ylabel("Log of Decay Rate (1/s)")
pl.legend(['Measured values with uncertainty','Prediction from Regression', 'Theoretical Curve'])
pl.savefig("Decay Rate over Time with Uncertainty and Linear Fit",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();

print("#####################################")
print("#####################################")
print("The expected half-life is 2.6 minutes, or 160 seconds.")
print("The linear regression method predicts the half-life to be ", linHalfLife, "seconds.");
print("The non-linear regression method predicts the half-life to be ", nonLinHalfLife, "seconds.");
print("#####################################")
print("The difference between the expected and linearly-regressed half-life is ", abs(linHalfLife-(2.6*60)),"seconds.")
print("The difference between the expected and nonlinearly-regressed half-life is ", abs(nonLinHalfLife-(2.6*60)), "seconds.")
print("#####################################")
print("#####################################")

#Error in lon-linear fit is smaller, so that's what we'll calculate chi-squared for

sigma_nonLinHalfLife = pcovNonLin[1,1];
uHalfLife = (sigma_nonLinHalfLife)**(1/2)
print("The uncertainty of the half-life calculated using the non-linear regression model is ", uHalfLife, "seconds.")

#Computing Chi-Squared

X2 = 0
i = 0
while i < len(t):
    X2 = X2 + (((sampleRate[i]-I_nonlin(t[i]))/uHalfLife)**2)
    i = i + 1
parameters = 2 #the number of parameters in nu = N - n
X2 = X2 / ((len(sampleCount))-parameters)
print("Chi-Squared is ",X2)
if X2 < 1:
    print("Thus, this model is over-fit.\n")