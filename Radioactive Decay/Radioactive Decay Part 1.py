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
from math import log10 , floor

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
uSampleRate = ((abs(sampleCount))**(1/2))/dt

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

from math import log10 , floor
def round_it(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

uLinHalfLife = (pcovLin[1,1])**(1/2)
uLinHalfLife = round_it(uLinHalfLife, 1)
uNonLinHalfLife = (pcovNonLin[1,1])**(1/2)
uNonLinHalfLife = uNonLinHalfLife / (nonLinHalfLife**2)
uNonLinHalfLife = round_it(uNonLinHalfLife, 1)

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
pl.subplot(2, 1, 2)
pl.plot(t, sampleRate, ".")
pl.plot(t, I_nonlin(t))
pl.plot(t, I_nonlin_theoretical(t))
pl.xlabel("Time (s)")
pl.ylabel("Decay Rate (1/s)")
pl.errorbar(t, sampleRate, yerr=uSampleRate, fmt=" ")
pl.legend(['Measured values with uncertainty','Prediction from Regression', 'Theoretical Curve'])

pl.subplot(2, 1, 1)
pl.plot(t, sampleRate-I_nonlin_theoretical(t), ".")
pl.errorbar(t, sampleRate-I_nonlin_theoretical(t), yerr=uSampleRate, fmt=" ")
pl.ylabel("Residue")

pl.savefig("Decay Rate over Time with Uncertainty and Non-Linear Fit",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();

pl.figure(2)
pl.cla()
pl.subplot(2, 1, 2)
pl.plot(t, log(sampleRate), ".")
pl.errorbar(t, log(sampleRate), yerr=uSampleRate/sampleRate, fmt=" ")
pl.plot(t, I_lin(t))
pl.plot(t, I_lin_theoretical(t))
pl.xlabel("Log of Time (s)")
pl.ylabel("Log of Decay Rate (1/s)")
pl.legend(['Measured values with uncertainty','Prediction from Regression', 'Theoretical Curve'])

pl.subplot(2, 1, 1)
pl.plot(t, (log(sampleRate))-I_lin_theoretical(t), ".")
pl.errorbar(t, (log(sampleRate))-I_lin_theoretical(t), yerr=uSampleRate/sampleRate, fmt=" ")
pl.ylabel("Residue")

pl.savefig("Decay Rate over Time with Uncertainty and Linear Fit",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();

print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("#####################################")
print("_________________________")
print("The expected half-life is 2.6 minutes, or 160 seconds.")
print("The linear regression method predicts the half-life to be ", linHalfLife, "+- ",uLinHalfLife," seconds.");
print("The non-linear regression method predicts the half-life to be ", nonLinHalfLife, "+- ",uNonLinHalfLife, " seconds.");
print("_________________________")
print("The difference between the expected and linearly-regressed half-life is ", abs(linHalfLife-(2.6*60))," +- ",((uLinHalfLife**2))**(1/2),"seconds.")
print("The difference between the expected and non-linearly-regressed half-life is ", abs(nonLinHalfLife-(2.6*60))," +- ",((uNonLinHalfLife**2))**(1/2), "seconds.")
print("_________________________")

#Error in lon-linear fit is smaller, so that's what we'll calculate chi-squared for

"""sigma_nonLinHalfLife = pcovNonLin[1,1];
uHalfLife = (sigma_nonLinHalfLife)**(1/2)
print("The uncertainty of the half-life calculated using the non-linear regression model is ", uHalfLife, "seconds.")"""

#Computing Chi-Squared
X2 = 0
i = 0
while i < len(t):
    if i == 52:
        i = 53;
    X2 = X2 + (((sampleRate[i]-I_nonlin(t[i]))/uSampleRate[i])**2)
    i = i + 1
parameters = 2 #the number of parameters in nu = N - n
X2 = X2 / ((len(sampleCount))-parameters)


print("Chi-Squared is ",X2," for the Non-Linear Regression Model.")
if X2 < 1:
    print("Thus, this model is over-fit.\n")
if X2 > 10:
    print("Thus, the model is a poor fit.\n")

if X2 > 1 and X2 < 5:
    print("Thus, the model is an incomplete fit.\n")