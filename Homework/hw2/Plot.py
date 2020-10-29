import yt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data from Enzo, and get x-axis data
data = yt.load("DD0001/data0001")
my_ray = data.ortho_ray(0, (0, 0))

# Get density, x-velocity, pressure
ray_sort = np.argsort(my_ray["x"])
Enzo_Rho = my_ray["density"][ray_sort]
Enzo_Vx = my_ray["x-velocity"][ray_sort]
Enzo_Pres = my_ray["pressure"][ray_sort]
Enzo_x = my_ray["x"][ray_sort]
# Change them to numpy array, even though we can do without it.
# Create the x coordinate
Enzo_Rho = np.asarray(Enzo_Rho)
Enzo_Vx = np.asarray(Enzo_Vx)
Enzo_Pres = np.asarray(Enzo_Pres)
Enzo_x = np.asarray(Enzo_x)

# Read from the strong-shock.txt ( analytic solution ).
headerList = ["r", "Rho", "Vx", "Vy", "Vz", "Pres"]
data = pd.read_csv('strong-shock.txt', skiprows=(0, 1, 2, 3, 4, 5), comment='#', names=headerList, sep='\s{2,}', engine='python')
r = data["r"]
Rho = data["Rho"]
Vx = data["Vx"]
Pres = data["Pres"]

# create figure
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, dpi=140)
fig.subplots_adjust(hspace=0.1, wspace=0.0)

ax[0].set_title('Enzo with AMR Level = 4, t = 0.4')
# Plot the result from Enzo
line_d, = ax[0].plot(Enzo_x, Enzo_Rho, 'ro', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.1, markersize=2)
line_u, = ax[1].plot(Enzo_x, Enzo_Vx, 'go', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.1, markersize=2)
line_p, = ax[2].plot(Enzo_x, Enzo_Pres, 'bo', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.1, markersize=2)
# Plot the analytic solution.
exact_d = ax[0].plot(r, Rho, 'k-', linewidth=0.5)
exact_u = ax[1].plot(r, Vx, 'k-', linewidth=0.5)
exact_P = ax[2].plot(r, Pres, 'k-', linewidth=0.5)

ax[2].set_xlabel('x')
ax[0].set_ylabel('Density')
ax[1].set_ylabel('Velocity')
ax[2].set_ylabel('Pressure')
ax[0].set_xlim(0.0, 1.0)
ax[0].set_ylim(+0.0, 1300.0)
ax[1].set_ylim(-0.2, 1.0)
ax[2].set_ylim(-50.0, 550.0)

plt.show()
