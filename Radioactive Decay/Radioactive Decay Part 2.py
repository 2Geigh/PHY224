#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy.stats import poisson
from scipy.stats import norm
import matplotlib.pyplot as pl
from numpy import e
from numpy import pi
from numpy import sqrt

machineData = np.loadtxt("Fiesta.txt", delimiter="\t");
dt = 3 #seconds
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

sampleCountMean = 0
for i in sampleCount:
    sampleCountMean = sampleCountMean + i/len(sampleCount);
m = sampleCountMean

#mu = mean number of successful mmeasurements of radioactive decay per time interval
#stdev mu = mu ** (1/2)

def Poisson(n,M):
    return ((np.e)**(-M))*(M**n)/(np.math.factorial(n))
poissonArray = []
x = np.arange(80)
for i in x:
    poissonArray.append(Poisson(i,m))
    
gaussianMean = m
gaussianStdv = m**(1/2)

def Gaussian(A):
    return (1/(gaussianStdv*(sqrt(2*pi))))*(e**(-0.5*(((A-gaussianMean)/gaussianStdv)**2)))
    
pl.cla()
pl.hist(sampleCount, bins="auto", density=(True))
pl.plot(x, poissonArray)
pl.plot(x, Gaussian(x))
pl.ylabel("Frequency")
pl.xlabel("Sample Count")
pl.legend(["Poisson probability mass function","Gaussian distribution"])

pl.savefig("Histogram of Fiesta Plate Sample Count",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();


#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################
#########################

m = backgroundMean
poissonArray = []
x = np.arange(10)
for i in x:
    poissonArray.append(Poisson(i,m))

gaussianMean = m
gaussianStdv = m**(1/2)

def Gaussian(A):
    return (1/(gaussianStdv*(sqrt(2*pi))))*(e**(-0.5*(((A-gaussianMean)/gaussianStdv)**2)))

pl.cla()
pl.hist(backgroundCount, bins=10, density=(True))
pl.plot(x, poissonArray)
pl.plot(x, Gaussian(x))
pl.ylabel("Frequency")
pl.xlabel("Sample Count")
pl.legend(["Poisson probability mass function","Gaussian distribution"])

pl.savefig("Histogram of Background Sample Count",dpi=300,bbox_inches="tight")
pl.tight_layout();
pl.show();

"""
handle=open("Barium.txt")
handle2=open("Background.txt")

data1=handle.read()
data2=handle2.read()
NCbar= np.loadtxt("Barium.txt",usecols=1)
NCback=np.loadtxt("Background.txt",usecols=1)"""


"""n = len(NCback)
def average(data,N):
    avg=(np.sum(data))/N
    return avg

average_back= average(NCback,n)
data= NCbar-average_back


# In[48]:


from scipy.stats import poisson
x=poisson(data)

#sp.stats.norm


# In[24]:


np.histogram(data)
plt.hist(data, bins=50)
plt.xlabel("Number count")
plt.ylabel("data points")


# In[35]:


#Add the Poisson probability mass function to qualitatively fit the data. The most
#appropriate value for ?? can by taking the average value of all of your count data. This
#is the Maximum Likelihood Estimation of the parameter.
x=sp.stats.poisson.math:(data)



# In[33]:


sp.stats.norm


# In[ ]:




"""