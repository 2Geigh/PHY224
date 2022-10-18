import numpy as np
import matplotlib.pyplot as plt
Battery_OpenCircuitVoltage, uBattery_OpenCircuitVoltage, Battery_Resistance, uBattery_Resistance, Battery_TerminalVoltage, uBattery_TerminalVoltage, Battery_Current, uBattery_Current = np.loadtxt("BATTERYdata.csv", delimiter=",", unpack=True)
DCSupply_OpenCircuitVoltage, uDCSupply_OpenCircuitVoltage, DCSupply_Resistance, DCSupply_Resistance, DCSupply_TerminalVoltage, uDCSupply_TerminalVoltage, DCSupply_Current, uDCSupply_Current = np.loadtxt("BATTERYdata.csv", delimiter=",", unpack=True)
#part 1
#(a)Plot V vs I and use the data to determine the output resistance of the battery Rb.
plt.plot(TV1,I1, marker="o",linestyle="None")
plt.xlabel("Battery Terminal Voltage (V)")
plt.ylabel("Current (I)")
plt.title("The Position Of A Rocket over Time")
#(b) Estimate all uncertainties.

plt.errorbar(time, position, yerr=up,fmt="o",ecolor="blue")


#wefsgtsdgdgfs