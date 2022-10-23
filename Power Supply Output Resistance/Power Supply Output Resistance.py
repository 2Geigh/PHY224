import numpy as np
import matplotlib.pyplot as plt
Battery_OpenCircuitVoltage, uBattery_OpenCircuitVoltage, Battery_Resistance, uBattery_Resistance, Battery_TerminalVoltage, uBattery_TerminalVoltage, Battery_Current, uBattery_Current = np.loadtxt("BATTERYdata.csv", delimiter=",", unpack=True)
DCSupply_OpenCircuitVoltage, uDCSupply_OpenCircuitVoltage, DCSupply_Resistance, uDCSupply_Resistance, DCSupply_TerminalVoltage, uDCSupply_TerminalVoltage, DCSupply_Current, uDCSupply_Current = np.loadtxt("DCSUPPLYdata.csv", delimiter=",", unpack=True)
#6.5V
V65_OpenCircuitVoltage, u65V_OpenCircuitVoltage, V65_Resistance, u65V_Resistance, V65_TerminalVoltage, u65V_TerminalVoltage, V65_Current, u65V_Current = np.loadtxt("6.5V.csv", delimiter=",", unpack=True)
#10v
V10_OpenCircuitVoltage, u10V_OpenCircuitVoltage, V10_Resistance, u10V_Resistance, V10_TerminalVoltage, u10V_TerminalVoltage, V10_Current, u10V_Current = np.loadtxt("10V.csv", delimiter=",", unpack=True)
#15V
V15_OpenCircuitVoltage, u15V_OpenCircuitVoltage, V15_Resistance, u15V_Resistance, V15_TerminalVoltage, u15V_TerminalVoltage, V15_Current, u15V_Current = np.loadtxt("15V.csv", delimiter=",", unpack=True)
#20v
V20_OpenCircuitVoltage, u20V_OpenCircuitVoltage, V20_Resistance, u20V_Resistance, V20_TerminalVoltage, u20V_TerminalVoltage, V20_Current, u20V_Current = np.loadtxt("20V.csv", delimiter=",", unpack=True)


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
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (I)")
plt.title("DC Power SUpply Voltage vs. Current")
#(c) Estimate all uncertainties.
plt.errorbar(DCSupply_Current,DCSupply_TerminalVoltage, xerr=uDCSupply_Current, yerr=uDCSupply_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
#plot 6.5
plt.plot(V65_Current,V65_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (I)")
plt.title("DC Power SUpply Voltage 6.5V vs. Current")
plt.errorbar(V65_Current,V65_TerminalVoltage, xerr=u65V_Current, yerr=u65V_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
#plot 10
plt.plot(V10_Current,V10_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (I)")
plt.title("DC Power SUpply Voltage 10V vs. Current")
plt.errorbar(V10_Current,V10_TerminalVoltage, xerr=u10V_Current, yerr=u10V_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
# PLOT 15
plt.plot(V15_Current,V15_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (I)")
plt.title("DC Power SUpply Voltage 15V vs. Current")
plt.errorbar(V15_Current,V15_TerminalVoltage, xerr=u15V_Current, yerr=u15V_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()
#PLOT 20
plt.plot(V20_Current,V20_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (I)")
plt.title("DC Power SUpply Voltage 20V vs. Current")
plt.errorbar(V20_Current,V20_TerminalVoltage, xerr=u20V_Current, yerr=u20V_TerminalVoltage,fmt="o",ecolor="blue")
plt.show()

