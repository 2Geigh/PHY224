import numpy as np
import matplotlib.pyplot as plt
OCV1, uOCV1, R1, uR1, TV1, uTV1, I1, uI1, OCV2, uOCV2, R2, uR2, TV2, uTV2, I2, uI2 = np.loadtxt("PSdata2.csv", delimiter=",", unpack=True)
#part 1
#(a)Plot V vs I and use the data to determine the output resistance of the battery Rb.
plt.plot(TV1,I1, marker="o",linestyle="None")
plt.xlabel("Battery Terminal Voltage (V)")
plt.ylabel("Current (I)")
plt.title("The Position Of A Rocket over Time")
#(b) Estimate all uncertainties.

plt.errorbar(time, position, yerr=up,fmt="o",ecolor="blue")


#wefsgtsdgdgfs