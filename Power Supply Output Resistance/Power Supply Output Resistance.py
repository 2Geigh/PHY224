import numpy as np
import matplotlib.pyplot as plt
Battery_OpenCircuitVoltage, uBattery_OpenCircuitVoltage, Battery_Resistance, uBattery_Resistance, Battery_TerminalVoltage, uBattery_TerminalVoltage, Battery_Current, uBattery_Current = np.loadtxt("BATTERYdata.csv", delimiter=",", unpack=True)
DCSupply_OpenCircuitVoltage, uDCSupply_OpenCircuitVoltage, DCSupply_Resistance, uDCSupply_Resistance, DCSupply_TerminalVoltage, uDCSupply_TerminalVoltage, DCSupply_Current, uDCSupply_Current = np.loadtxt("DCSUPPLYdata.csv", delimiter=",", unpack=True)
#6.5V
65V_OpenCircuitVoltage, u65V_OpenCircuitVoltage, 65V_Resistance, u65V_Resistance, 65V_TerminalVoltage, u65V_TerminalVoltage, 65V_Current, u65V_Current = np.loadtxt("6.5V.csv", delimiter=",", unpack=True)
#10v
10V_OpenCircuitVoltage, u10V_OpenCircuitVoltage, 10V_Resistance, u10V_Resistance, 10V_TerminalVoltage, u10V_TerminalVoltage, 10V_Current, u10V_Current = np.loadtxt("10V.csv", delimiter=",", unpack=True)
#15V
15V_OpenCircuitVoltage, u15V_OpenCircuitVoltage, 15V_Resistance, u15V_Resistance, 15V_TerminalVoltage, u15V_TerminalVoltage, 15V_Current, u15V_Current = np.loadtxt("15V.csv", delimiter=",", unpack=True)
#20v
20V_OpenCircuitVoltage, u20V_OpenCircuitVoltage, 20V_Resistance, u20V_Resistance, 20V_TerminalVoltage, u20V_TerminalVoltage, 20V_Current, u20V_Current = np.loadtxt("20V.csv", delimiter=",", unpack=True)

#part 1
#(a)Plot V vs I and use the data to determine the output resistance of the battery Rb.
plt.plot(Battery_Current,Battery_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("Battery Terminal Voltage (V)")
plt.xlabel("Current (I)")
plt.title("Battery Terminal voltage vs. Current")
#(b) Estimate all uncertainties.
plt.errorbar(Battery_Current,Battery_TerminalVoltage, xerr=uBattery_Current,yerr=uBattery_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
#part2
#(b) Plot voltage V as a function of current I, calculate the output resistance of
# thepower supply Rps

plt.plot(DCSupply_Current,DCSupply_TerminalVoltage, marker="o",linestyle="None")
plt.ylabel("Battery Terminal Voltage (V)")
plt.xlabel("Current (I)")
plt.title("Battery Terminal voltage vs. Current")
#(c) Estimate all uncertainties.
plt.errorbar(DCSupply_Current,DCSupply_TerminalVoltage, xerr=uDCSupply_Current, yerr=uDCSupply_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
#plot 6.5

plt.plot(65V_Current,65V_TerminalVoltage, marker="o",linestyle="None")
plt.ylabel("Battery Terminal Voltage (V)")
plt.xlabel("Current (I)")
plt.title("Battery Terminal voltage vs. Current")
#(c) Estimate all uncertainties.
plt.errorbar(65V_Current,65V_TerminalVoltage, xerr=u65V_Current, yerr=u65V_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
