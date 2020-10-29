import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time_step = "t=0_01"
problem = "4-body"

filename = "./result/" + time_step + "/" + problem + "/error_KDK.txt"
data = pd.read_csv(filename, header=None)
KDK_err = data[0]

filename = "./result/" + time_step + "/" + problem + "/error_DKD.txt"
data = pd.read_csv(filename, header=None)
DKD_err = data[0]

filename = "./result/" + time_step + "/" + problem + "/error_RK4.txt"
data = pd.read_csv(filename, header=None)
RK4_err = data[0]

filename = "./result/" + time_step + "/" + problem + "/time.txt"
data = pd.read_csv(filename, header=None)
time = data[0]

# For SunEarthMoon like system error
# plt.plot(time, KDK_err, label="KDK", linewidth=0.5)
# plt.plot(time, DKD_err, label="DKD", linewidth=0.5)
# plt.plot(time, RK4_err, label="RK4", linewidth=0.5)
# plt.yscale('symlog', linthreshy=0.01)
# plt.legend(loc="upper right")
# plt.xlabel("time", fontsize=14)
# plt.title("Error Log Plot of Sun Earth Moon like System", fontsize=16)
# plt.show()

# For 3-body system
# plt.plot(time, KDK_err, label="KDK")
# plt.plot(time, DKD_err, label="DKD")
# plt.plot(time, RK4_err, label="RK4")
# plt.legend(loc="upper right")
# plt.xlabel("time", fontsize=14)
# plt.title("Error Plot of 3-Body System", fontsize=16)
# plt.show()

# For the original problem
# plt.plot(time, KDK_err, label="KDK")
# plt.plot(time, DKD_err, label="DKD")
# plt.plot(time, RK4_err, label="RK4")
# plt.legend(loc="upper right")
# plt.xlabel("time", fontsize=14)
# plt.title("Error Plot of the Original Problem (2-body)", fontsize=16)
# plt.show()

#
plt.plot(time, KDK_err, label="KDK", linewidth=0.5)
plt.plot(time, DKD_err, label="DKD", linewidth=0.5)
plt.plot(time, RK4_err, label="RK4", linewidth=0.5)
plt.legend(loc="upper right")
plt.xlabel("time", fontsize=14)
plt.title("Error Plot of the 4-body", fontsize=16)
plt.show()
