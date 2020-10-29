import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings  
# (N kinds of GridSize, N kinds of NThread)
loop_shape = (7, 26) # TODO

# Read data from file
#####################
filename = "./result/result.txt"
data = pd.read_csv(filename, sep=' ', header=None)

Error 		= np.asarray(data[0])
GridSize 	= np.asarray(data[1])
Iteration	= np.asarray(data[2])
TimeUsed	= np.asarray(data[3])
NThread		= np.asarray(data[4])

# Reshape data,
################
# row index -> Grid size N x N
# col index -> Number of threads
TimeUsedGrid = TimeUsed.reshape(loop_shape)

ErrorGrid = Error.reshape(loop_shape)
ErrorArr = ErrorGrid[:,0]

GridSize = GridSize.reshape(loop_shape)
GridSize = GridSize[:,0]

NThread = NThread.reshape(loop_shape)
NThread = NThread[0,:]

# Further calculate data
########################
# SpeedUp = time_spent / (time_spent with NThread = 1)
SpeedUpGrid = np.copy(TimeUsedGrid)
for i in range(SpeedUpGrid.shape[0]):
    SpeedUpGrid[i,:] = SpeedUpGrid[i,0] / SpeedUpGrid[i,:]

# Efficiency = SpeedUp / NThread
EfficiencyGrid = np.copy(SpeedUpGrid)
for i in range(EfficiencyGrid.shape[0]):
    # EfficiencyGrid[i,:] = (EfficiencyGrid[i,:] - 1.0) / NThread
    EfficiencyGrid[i,:] = EfficiencyGrid[i,:] / NThread
    

# Plot as figure
################
# Plot SpeedUp
for i in range(SpeedUpGrid.shape[0]):
    plt.plot(NThread, SpeedUpGrid[i,:], '.-', label="GridSize = " + str(GridSize[i]))
plt.plot(NThread, np.ones(len(NThread)), '-.', color=(0.5, 0.5, 0.5), linewidth=2.0)
plt.title("Speed Up Rate Compare with NThread = 1", fontsize=16)
plt.xlabel("Number of Threads", fontsize=14)
plt.legend(loc="upper left")
plt.xlim(0.0, NThread.max() + 1)
plt.show()

# # Plot Efficiency-NThread
plt.plot(NThread, np.zeros(len(NThread)), '-.', color=(0.5, 0.5, 0.5), linewidth=2.0)
for i in range(SpeedUpGrid.shape[0]):
    plt.plot(NThread, EfficiencyGrid[i,:], '.-', label="GridSize = " + str(GridSize[i]))
plt.title("Speed Up Efficiency", fontsize=16)
plt.xlabel("Number of Threads", fontsize=14)
plt.yticks(np.linspace(100, 0, 11, endpoint=True) * 0.01, 
		   ["100 %", "90 %", "80 %", "70 %", "60 %", "50 %", "40 %", "30 %", "20 %", "10 %", "0 %"])
plt.legend()
plt.show()

# Plot Efficiency(each NThread)-GridSize
for i in range(SpeedUpGrid.shape[1]):
    plt.plot(GridSize, EfficiencyGrid[:,i], '.-', label="NThread = " + str(NThread[i]))
plt.title("Speed Up Efficiency", fontsize=16)
plt.xlabel("Grid Size", fontsize=14)
plt.yticks(np.linspace(100, 0, 11, endpoint=True) * 0.01, 
		   ["100 %", "90 %", "80 %", "70 %", "60 %", "50 %", "40 %", "30 %", "20 %", "10 %", "0 %"])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Plot Error - GridSize
plt.plot(GridSize, ErrorArr, 'b.-')
plt.title("Error vs Grid Size", fontsize=16)
plt.xlabel("Grid Size N x N", fontsize=14)
plt.show()
