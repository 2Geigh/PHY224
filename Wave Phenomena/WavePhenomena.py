"NICK GARCIA & MARTA GONCZAR"
"""WAVE PHENOMENA"""
#2022/11/15

import numpy as np
from numpy import sin
import matplotlib.pyplot as pl

"DIFRACTION"

f = 20 #Hz

slitSeperation = np.array([3.0,1.0,0.50,0.01])
angularSpread = np.array([34.21,41.72,50.47,58.86])
uAngularSpread = 15/100 * angularSpread

a = slitSeperation
uA = uAngularSpread
th = angularSpread

pl.cla()
pl.plot(sin(th),a, ".")
pl.errorbar(sin(th), a, yerr=uA, fmt=" ")
pl.xlabel("Sin of the Angular Spread")
pl.ylabel("Slit seperation (cm)")
pl.savefig("Slit sepration over Sine of Angular Spread");