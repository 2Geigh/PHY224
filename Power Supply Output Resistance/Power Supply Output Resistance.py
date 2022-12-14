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
plt.xlabel("Current (A)")
plt.title("Battery Terminal voltage vs. Current")
#(b) Estimate all uncertainties.
plt.errorbar(Battery_Current,Battery_TerminalVoltage, xerr=uBattery_Current,yerr=uBattery_TerminalVoltage,fmt="o",ecolor="blue")
plt.savefig("1-BatteryTerminal-VxI");
plt.show()
#part2
#(b) Plot voltage V as a function of current I, calculate the output resistance of
# thepower supply Rps

plt.subplot()
plt.plot(DCSupply_Current,DCSupply_TerminalVoltage, marker="o",linestyle="None")
plt.ylabel("DC Power SUpply Voltage (V)")
plt.xlabel("Current (mA)")
plt.title("DC Power SUpply Voltage vs. Current")
#(c) Estimate all uncertainties.
plt.errorbar(DCSupply_Current,DCSupply_TerminalVoltage, xerr=uDCSupply_Current, yerr=uDCSupply_TerminalVoltage,fmt="o",ecolor="blue")
plt.savefig("2-BatteryTerminal-VxI");
plt.show()
#plot 6.5
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(V65_Current,V65_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("Voltage (V)")
plt.xlabel("Current (mA)")
plt.title("6.5V")
plt.errorbar(V65_Current,V65_TerminalVoltage, xerr=u65V_Current, yerr=u65V_TerminalVoltage,fmt="o",ecolor="blue")
#plt.savefig("6,5V-BatteryTerminal-VxI");
#plt.show()
#plot 10
plt.subplot(2,2,2)
plt.plot(V10_Current,V10_TerminalVoltage, marker="o",linestyle="-")
plt.xlabel("Current (mA)")
plt.title("10V")
plt.errorbar(V10_Current,V10_TerminalVoltage, xerr=u10V_Current, yerr=u10V_TerminalVoltage,fmt="o",ecolor="blue")
#plt.savefig("10V-BatteryTerminal-VxI");
#plt.show()
# PLOT 15
plt.subplot(2,2,3)
plt.plot(V15_Current,V15_TerminalVoltage, marker="o",linestyle="-")
plt.ylabel("Voltage (V)")
plt.xlabel("Current (mA)")
plt.title("15V")
plt.errorbar(V15_Current,V15_TerminalVoltage, xerr=u15V_Current, yerr=u15V_TerminalVoltage,fmt="o",ecolor="blue")
#plt.savefig("15V-BatteryTerminal-VxI");
#plt.show()
#PLOT 20
plt.subplot(2,2,4)
plt.plot(V20_Current,V20_TerminalVoltage, marker="o",linestyle="-")
#plt.ylabel("DC Power Supply Voltage (V)")
plt.xlabel("Current (mA)")
plt.title("20V")
plt.errorbar(V20_Current,V20_TerminalVoltage, xerr=u20V_Current, yerr=u20V_TerminalVoltage,fmt="o",ecolor="blue")
plt.tight_layout(h_pad=2);
plt.savefig("20V-BatteryTerminal-VxI",dpi=300,bbox_inches="tight");
plt.show()

output_resitance=[0.676, 1.092, 1.704,7.619]
uoutput_resitance=[0.17, 0.36, 0.78, 6.15]
plt.plot(Battery_Resistance, output_resitance,marker="o",linestyle="-")
plt.xlabel("Battery Resistance (ohms)")
plt.ylabel("Output resitance (ohms)")
plt.title("Battery Resitance Vs Output Resistance")
plt.errorbar(Battery_Resistance, output_resitance, xerr=uBattery_Resistance, yerr=uoutput_resitance,fmt="o",ecolor="blue")
plt.show()

output_res2=[0.0955261901, 1.835168502, 1.794430089,1.661129568,1.83286735,1.822521154, 1.819623944, 1.888319396,1.844660194,1.8287373,1.838006231,1.798237727,1.848563969,1.832061069,1.799065421, 1.618122977]
uoutput_res2=[0.167139204,0.3495309685,
0.7529019955,
4.360213809,
0.1441300205,
0.3028944476,
0.6523361729,
3.7758079,
0.1309176374,
0.2749073124,
0.5910486334,
3.415989566,
0.1244775061,
0.2608595657,
0.5599005086,
3.23560266]
plt.plot(DCSupply_Resistance, output_res2,marker="o",linestyle="none")
plt.xlabel("DC Power Supply Resistance (ohms)")
plt.ylabel("Output resitance (ohms)")
plt.title("DC Power Supply Resitance Vs Output Resistance")
plt.errorbar(DCSupply_Resistance, output_res2, xerr=uDCSupply_Resistance, yerr=uoutput_res2,fmt="o",ecolor="blue")
plt.show()
#######