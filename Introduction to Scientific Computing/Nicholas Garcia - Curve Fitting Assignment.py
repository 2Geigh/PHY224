#Nicholas Garcia
#2022 / 9 / 20

#Introduction to Scientific Computing

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl




"1: READING AND PLOTTING DATA"
rocketData = np.loadtxt("rocket.csv",delimiter=',')
r = rocketData[:,1]; #vertical position of rocket in km
t = rocketData[:,0]; #elapsed time since launch in hours
ur = rocketData[:,2]; #uncertainty of position of rocket in km

#plotting the data in a scatter plot
pl.cla()
pl.plot(t,r,"g.");
pl.errorbar(t, r, yerr=ur,fmt=" ");
pl.title("Vertical Position of Saturn V Rocket from time of Launch");
pl.xlabel("Time (hours)");
pl.ylabel("Position (km)");
pl.savefig("Saturn V Position-Time Graph with Measured Values",dpi=300,bbox_inches="tight")
pl.show()




"2: ESTIMATING THE SPEED"
#Calculating the mean of the speed
#Mean of speed = sum of speed values / number of speed values

v = r / t;
v = v[1:] #done to remove null value v[0]

v_total = 0;
for i in range(1,23,1):
    v_total = v_total + v[i]; #getting the sum of total speed values
    
#FINAL ANSWERS
v_mean = v_total / len(v)
v_standard_error = np.std(v) #standard deviation of the speed values




"3: LINEAR REGRESSION"
rocketData = np.loadtxt("rocket.csv", delimiter=",")
t = rocketData[:,0]
d = rocketData[:,1]

t_ave = 0
for i in range(0,24,1):
    t_ave = t_ave + t[i]
t_ave = t_ave / len(t) #t_bar

d_ave = 0;
for i in range(0,24,1):
    d_ave = d_ave + d[i]
d_ave = d_ave / len(d) #d_bar

#Determining the numerator and denominators respectively of the u_bar equation
num_sum = 0;
for i in range(1,24,1):
    num_sum = num_sum + ((t[i]-t_ave)*(d[i]-d_ave))
den_sum = 0;
for i in range(1,24,1):
    den_sum = den_sum + ((t[i]-t_ave)**2)

#FINAL ANSWERS
u_best_est = num_sum / den_sum; #u_hat
d0_best_est = d_ave - (u_best_est * t_ave) #d_0_hat




"4: PLOTTING THE PREDICTION"
def d(t,d0,u):
    return d0 + (u*t);

pl.cla()
measured = pl.plot(t,r,"g.");
uncertainty = pl.errorbar(t, r, yerr=ur,fmt=" ");
predicted = pl.plot(t,d(t,d0_best_est,u_best_est),"b-")
pl.title("Vertical Position of Saturn V Rocket from time of Launch");
pl.xlabel("Time (hours)");
pl.ylabel("Position (km)");
pl.plot([1, 2])
pl.legend(['Measured values with uncertainty','Predicted values'])
pl.savefig("Saturn V Position-Time Graph with Measured and Predicted Values",dpi=300,bbox_inches="tight")
pl.show()




"5: CHARACTERIZING THE FIT"
def X_2_r(d0,u):
    X2r_sum = 0
    for i in range(0,len(t), 1):
        X2r_sum = X2r_sum + ((r[i]-d(t[i],d0,u))**2)/((ur[i])**2)
    X2r = (1/(len(t)-2)) * X2r_sum;
    return X2r;

print(X_2_r(d0_best_est,u_best_est));
print("X2r ABOVE");
#USE COEFFICEITNS AND SUCH FROM BOTH LINEARIZATIO NAND FROM "PLOTTING THE PREDICTION" EXERCISE




"6: CURVE FITTING"
from scipy.optimize import curve_fit

def model(x,v,v0):
    return v*x + v0;

popt, pcov = curve_fit(model,t,r);
pstd = np.sqrt(np.diag(pcov));

v0_best_est = popt[0];
r0_best_est = popt[1];
uv0_best_est = pstd[0];
ur0_best_est = pstd[1];

print(v0_best_est);
print(r0_best_est);
print(uv0_best_est);
print(ur0_best_est);

X2r_sum2 = 0
for i in range(0,len(t), 1):
    X2r_sum2 = X2r_sum2 + ((r[i]-d(t[i],r0_best_est,v0_best_est))**2)/((ur0_best_est)**2)
X2r_2 = (1/(len(t)-2)) * X2r_sum2;

print(X2r_2);
print("X2r_2 ABOVE")

pl.cla()
measured = pl.plot(t,r,"g.");
uncertainty = pl.errorbar(t, r, yerr=ur,fmt=" ");
pl.errorbar(t, model(t,popt[0],popt[1]), yerr=ur,fmt=" ");
predicted = pl.plot(t,model(t,popt[0],popt[1]),"r-")
pl.title("Vertical Position of Saturn V Rocket from time of Launch");
pl.xlabel("Time (hours)");
pl.ylabel("Position (km)");
pl.plot([1, 2])
pl.legend(['Measured values with uncertainty','Predicted values from Linear Regression with Uncertainty'])
pl.savefig("Saturn V Linear Regression Model figure",dpi=300,bbox_inches="tight")
pl.show()




#______________________________________________________________________________




"""FEATHER DROP EXPERIMENT"""




featherData = np.loadtxt("feather.csv",delimiter=',')
s = featherData[:,1];
t = featherData[:,0];
us = featherData[:,2];

def whatis_s (t,s0,v,a):
    return s0 + (v*t) + ((t**2)*a/2);

popt, pcov = curve_fit(whatis_s,t,s,sigma=us,p0=[2,1,1.4]);
pstd = np.sqrt(np.diag(pcov));

s0_best_est = popt[0];
v_best_est = popt[1];
a_best_est = pstd[2];
us0_best_est = pstd[0];
uv_best_est = pstd[1];
ua_best_est = pstd[2];

print("The best estimate of the feather's initial height is ",s0_best_est," metres.");
print("The best estimate of the feather's speed is ",v_best_est," metres per second.");
print("The best estimate of the feather's acceleration is ",a_best_est," metres per square second.");

print("The best estimate of the uncertainty of the feather's initial height is ",us0_best_est," metres.");
print("The best estimate of the uncertainty of the feather's speed is ",uv_best_est," metres per second.");
print("The best estimate of the uncertainty of the feather's acceleration is ",ua_best_est," metres per square second.");

pl.cla()
measured = pl.plot(t,s,"g.");
uncertainty = pl.errorbar(t, s, yerr=us,fmt=" ");
predicted = pl.plot(t,whatis_s(t,popt[0],popt[1],popt[2]),"r-")
pl.title("Feather trajectory");
pl.xlabel("Time (s)");
pl.ylabel("Height (m)");
pl.plot([1, 2])
pl.legend(['Measured values with uncertainty','Predicted values from Regression with Uncertainty'])
pl.savefig("Feather trajectory figure",dpi=300,bbox_inches="tight")
pl.show()