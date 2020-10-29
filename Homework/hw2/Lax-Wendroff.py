
#--------------------------------------------------------------------
# Test the Sod's shock tube problem with the Lax-Wendroff scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L = 1.0       # 1-D computational domain size
N_In = 128       # number of computing cells
cfl = 1.0       # Courant factor
nghost = 1         # number of ghost zones
gamma = 5.0 / 3.0   # ratio of specific heats
end_time = 0.4       # simulation time

# derived constants
N = N_In + 2 * nghost    # total number of cells including ghost zones
dx = L / N_In             # spatial resolution

# plotting parameters
nstep_per_image = 1     # plotting frequency


# -------------------------------------------------------------------
# define initial condition
# -------------------------------------------------------------------
def InitialCondition(x):
   #  Sod shock tube
   if (x < 0.5 * L):
      d = 1.25e3  # density
      u = 0.0     # velocity x
      P = 5.0e2   # pressure
      E = P / (gamma - 1.0) + 0.5 * d * u**2.0    # energy density
   else:
      d = 1.25e2
      u = 0.0
      P = 5.0
      E = P / (gamma - 1.0) + 0.5 * d * u**2.0

#  conserved variables [0/1/2] <--> [density/momentum x/energy]
   return np.array([d, d * u, E])


# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition(U):
   #  outflow
   U[0:nghost] = U[nghost]
   U[N - nghost:N] = U[N - nghost - 1]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure(d, px, e):
   P = (gamma - 1.0) * (e - 0.5 * px**2.0 / d)
   return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep(U):
   P = ComputePressure(U[:, 0], U[:, 1], U[:, 2])
   a = (gamma * P / U[:, 0])**0.5
   u = np.abs(U[:, 1] / U[:, 0])

   max_info_speed = np.amax(u + a)
   dt_cfl = cfl * dx / max_info_speed
   dt_end = end_time - t

   return min(dt_cfl, dt_end)


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
def Conserved2Flux(U):
   flux = np.empty(3)

   P = ComputePressure(U[0], U[1], U[2])
   u = U[1] / U[0]

   flux[0] = U[1]
   flux[1] = u * U[1] + P
   flux[2] = u * (U[2] + P)

   return flux


# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------
def init():
   line_d.set_xdata(x)
   line_u.set_xdata(x)
   line_p.set_xdata(x)
   return line_d, line_u, line_p


# -------------------------------------------------------------------
# update animation
# -------------------------------------------------------------------
def update(frame):
   global t, U

#  for frame==0, just plot the initial condition
   if frame > 0:
      for step in range(nstep_per_image):

         #        set the boundary conditions
         BoundaryCondition(U)

#        estimate time-step from the CFL condition
         dt = ComputeTimestep(U)
         print("t = %13.7e --> %13.7e, dt = %13.7e" % (t, t + dt, dt))

#        compute fluxes
         flux = np.empty((N, 3))
         for j in range(nghost, N - nghost + 1):
            #           flux[j] is defined at j-1/2
            U_half = 0.5 * (U[j] + U[j - 1]) - 0.5 * dt / dx * (Conserved2Flux(U[j]) - Conserved2Flux(U[j - 1]))
            flux[j] = Conserved2Flux(U_half)

#        update the volume-averaged input variables by dt
         U[nghost:N - nghost] -= dt / dx * (flux[nghost + 1:N - nghost + 1] - flux[nghost:N - nghost])

#        update time
         t = t + dt
         if (t >= end_time):
            anim.event_source.stop()
            break

#  plot
   d = U[nghost:N - nghost, 0]
   u = U[nghost:N - nghost, 1] / U[nghost:N - nghost, 0]
   P = ComputePressure(U[nghost:N - nghost, 0], U[nghost:N - nghost, 1], U[nghost:N - nghost, 2])
   print('t = %6.3f, density = (%.2f, %.2f), velocity = (%.2f, %.2f), pressure = (%.2f, %.2f)' % (t, np.min(d), np.max(d), np.min(u), np.max(u), np.min(P), np.max(P)))
   line_d.set_ydata(d)
   line_u.set_ydata(u)
   line_p.set_ydata(P)
 # ax[0].legend( loc='upper right', fontsize=12 )
 # ax[1].legend( loc='upper right', fontsize=12 )
 # ax[2].legend( loc='upper right', fontsize=12 )
   ax[0].set_title('Lax-Wendroff , t = %6.3f' % (t))

   return line_d, line_u, line_p


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
# set initial condition
t = 0.0
x = np.empty(N_In)
U = np.empty((N, 3))
for j in range(N_In):
   x[j] = (j + 0.5) * dx    # cell-centered coordinates
   U[j + nghost] = InitialCondition(x[j])

# create figure
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, dpi=140)
fig.subplots_adjust(hspace=0.1, wspace=0.0)
# fig.set_size_inches( 6.4, 12.8 )
line_d, = ax[0].plot([], [], 'r-o', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.5, markersize=2)
line_u, = ax[1].plot([], [], 'g-o', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.5, markersize=2)
line_p, = ax[2].plot([], [], 'b-o', ls='-', linewidth=1, markeredgecolor='k', markeredgewidth=0.5, markersize=2)

# Read from the strong-shock.txt.
headerList = ["r", "Rho", "Vx", "Vy", "Vz", "Pres"]
data = pd.read_csv('strong-shock.txt', skiprows=(0, 1, 2, 3, 4, 5), comment='#', names=headerList, sep='\s{2,}', engine='python')
r = data["r"]
Rho = data["Rho"]
Vx = data["Vx"]
Pres = data["Pres"]

# Plot the exact solution.
exact_d = ax[0].plot(r, Rho, 'k-', linewidth=0.5)
exact_u = ax[1].plot(r, Vx, 'k-', linewidth=0.5)
exact_P = ax[2].plot(r, Pres, 'k-', linewidth=0.5)

ax[2].set_xlabel('x')
ax[0].set_ylabel('Density')
ax[1].set_ylabel('Velocity')
ax[2].set_ylabel('Pressure')
ax[0].set_xlim(0.0, L)
ax[0].set_ylim(+0.0, 1300.0)
ax[1].set_ylim(-0.2, 1.0)
ax[2].set_ylim(-50.0, 550.0)

# create movie
nframe = 99999999  # arbitrarily large
anim = animation.FuncAnimation(fig, func=update, init_func=init,
                               frames=nframe, interval=10, repeat=False)
plt.show()
