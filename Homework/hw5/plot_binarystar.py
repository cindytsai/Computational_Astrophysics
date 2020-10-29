import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot errors in different algorithm
# #######################################################
time_step = "t=0_01"
error_type = "energy"

filename = "./result/BinaryStar/" + time_step + "_" + error_type + "/error_KDK.txt"
data = pd.read_csv(filename, header=None)
KDK_err = data[0]

filename = "./result/BinaryStar/" + time_step + "_" + error_type + "/error_DKD.txt"
data = pd.read_csv(filename, header=None)
DKD_err = data[0]

filename = "./result/BinaryStar/" + time_step + "_" + error_type + "/error_RK4.txt"
data = pd.read_csv(filename, header=None)
RK4_err = data[0]

filename = "./result/BinaryStar/" + time_step + "_" + error_type + "/time.txt"
data = pd.read_csv(filename, header=None)
time = data[0]

#
plt.plot(time, KDK_err, label="KDK")
plt.plot(time, DKD_err, label="DKD")
plt.plot(time, RK4_err, label="RK4")
plt.legend(loc="upper right")
plt.xlabel("time", fontsize=14)
plt.title("Error Plot (%s) of Binary Star" % error_type, fontsize=16)
plt.show()

# Plot errors in same algorithm but different time step
# ######################################################

error_type = "position"
algorithm_name = "DKD"

filename = "./result/BinaryStar/" + "t=0_02" + "_" + error_type + "/error_" + algorithm_name + ".txt"
data = pd.read_csv(filename, header=None)
err1 = data[0]

filename = "./result/BinaryStar/" + "t=0_02" + "_" + error_type + "/time.txt"
data = pd.read_csv(filename, header=None)
time1 = data[0]

filename = "./result/BinaryStar/" + "t=0_01" + "_" + error_type + "/error_" + algorithm_name + ".txt"
data = pd.read_csv(filename, header=None)
err2 = data[0]

filename = "./result/BinaryStar/" + "t=0_01" + "_" + error_type + "/time.txt"
data = pd.read_csv(filename, header=None)
time2 = data[0]

filename = "./result/BinaryStar/" + "t=0_005" + "_" + error_type + "/error_" + algorithm_name + ".txt"
data = pd.read_csv(filename, header=None)
err3 = data[0]

filename = "./result/BinaryStar/" + "t=0_005" + "_" + error_type + "/time.txt"
data = pd.read_csv(filename, header=None)
time3 = data[0]


plt.plot(time1, err1, label=algorithm_name + " 0.02")
plt.plot(time2, err2, label=algorithm_name + " 0.01")
plt.plot(time3, err3, label=algorithm_name + " 0.005")
plt.legend(loc="upper right")
plt.xlabel("time", fontsize=14)
plt.title("Error Plot (%s) of Binary Star of Different " % error_type + r'$\Delta$' + "t", fontsize=16)
plt.show()
