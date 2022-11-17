# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:14:13 2022

@author: Marta
"""

frequency= [5,10,12,15,20]
avg_wavelength=[4.117,1.9,1.59,1.27,0.972]
uavg_wavelength=[0.4117,0.19,0.2385,0.1905,0.1458]
import  matplotlib.pyplot as plt
plt.plot(frequency,avg_wavelength,".")
plt.errorbar(frequency,avg_wavelength, yerr=uavg_wavelength, fmt="none");
plt.xlabel("Frequency (Hz)");
plt.ylabel("Average Wavelength (cm)")