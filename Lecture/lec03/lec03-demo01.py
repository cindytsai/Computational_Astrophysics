
#--------------------------------------------------------------------
# Simulate acoustic wave with the Lax-Friedrichs scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter


#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L        = 1.0       # 1-D computational domain size
N_In     = 64        # number of computing cells
cfl      = 0.8       # Courant factor
nghost   = 1         # number of ghost zones
cs       = 1.0       # sound speed
d_amp    = 1.0e-6    # density perturbation amplitude
d0       = 1.0       # density background
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 5.0       # simulation time

# derived constants
N  = N_In + 2*nghost    # total number of cells including ghost zones
dx = L/N_In             # spatial resolution

# plotting parameters
nstep_per_image = 1     # plotting frequency


# -------------------------------------------------------------------
# define the reference solution
# -------------------------------------------------------------------
def ref_func( x, t ):
   WaveK = 2.0*np.pi/L        # wavenumber
   WaveW = 2.0*np.pi/(L/cs)   # angular frequency
   Phase = WaveK*x - WaveW*t  # wave phase

   v1 = cs*d_amp/d0        # velocity perturbation
   P0 = cs**2.0*d0/gamma   # background pressure
   P1 = cs**2.0*d_amp      # pressure perturbation

#  d/u/P/e = density/velocity/pressure/total energy
   d = d0 + d_amp*np.cos(Phase)
   u = v1*np.cos(Phase)
   P = P0 + P1*np.cos(Phase)
   E = P/(gamma-1.0) + 0.5*d*u**2.0

#  conserved variables [0/1/2] <--> [density/momentum x/energy]
   return np.array( [d, d*u, E] )


# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition( U ):
#  periodic
   U[0:nghost]   = U[N_In:nghost+N_In]
   U[N-nghost:N] = U[N-nghost-N_In:N-N_In]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure( d, px, e ):
   P = (gamma-1.0)*( e - 0.5*px**2.0/d )
   return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
   P = ComputePressure( U[:,0], U[:,1], U[:,2] )
   a = ( gamma*P/U[:,0] )**0.5
   u = np.abs( U[:,1]/U[:,0] )

   max_info_speed = np.amax( u + a )
   dt_cfl         = cfl*dx/max_info_speed
   dt_end         = end_time - t

   return min( dt_cfl, dt_end )


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
def Conserved2Flux( U ):
   flux = np.empty( 3 )

   P = ComputePressure( U[0], U[1], U[2] )
   u = U[1] / U[0]

   flux[0] = U[1]
   flux[1] = u*U[1] + P
   flux[2] = u*( U[2] + P )

   return flux


# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------
def init():
   line_num.set_xdata( x )
   line_ref.set_xdata( x )
   return line_num, line_ref


# -------------------------------------------------------------------
# update animation
# -------------------------------------------------------------------
def update( frame ):
   global t, U

#  for frame==0, just plot the initial condition
   if frame > 0:
      for step in range( nstep_per_image ):

#        set the boundary conditions
         BoundaryCondition( U )

#        estimate time-step from the CFL condition
         dt = ComputeTimestep( U )
         print( "t = %13.7e --> %13.7e, dt = %13.7e" % (t,t+dt,dt) )

#        compute fluxes
         flux = np.empty( (N,3) )
         for j in range( nghost, N-nghost+1 ):
#           flux[j] is defined at j-1/2
            flux[j] = 0.5*(  Conserved2Flux( U[j] ) + Conserved2Flux( U[j-1] ) \
                            -dx/dt*( U[j] - U[j-1] )  )

#        update the volume-averaged input variables by dt
         U[nghost:N-nghost] -= dt/dx*( flux[nghost+1:N-nghost+1] - flux[nghost:N-nghost] )

#        update time
         t = t + dt
         if ( t >= end_time ):
            anim.event_source.stop()
            break

#  calculate the reference analytical solution and estimate errors
   for j in range( N_In ):
      U_ref[j+nghost] = ref_func( x[j], t )

   d     = U    [ nghost:N-nghost, 0 ] - d0
   d_ref = U_ref[ nghost:N-nghost, 0 ] - d0
   err   = np.abs( d_ref - d ).sum()/N_In

#  plot
   line_num.set_ydata( d )
   line_ref.set_ydata( d_ref )
   ax.legend( loc='upper right', fontsize=12 )
   ax.set_title( 't = %6.3f, error = %10.3e' % (t, err) )

   return line_num, line_ref


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
# set initial condition
t     = 0.0
x     = np.empty( N_In )
U     = np.empty( (N,3) )
U_ref = np.empty( (N,3) )
for j in range( N_In ):
   x[j] = (j+0.5)*dx    # cell-centered coordinates
   U[j+nghost] = ref_func( x[j], t )

# create figure
fig, ax = plt.subplots( 1, 1, dpi=140 )
fig.subplots_adjust( hspace=0.0, wspace=0.0 )
#fig.set_size_inches( 6.4, 12.8 )
line_num, = ax.plot( [], [], 'r', ls='-',  label='Numerical' )
line_ref, = ax.plot( [], [], 'b', ls='--', label='Reference' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'Density' )
ax.set_xlim( 0.0, L )
ax.set_ylim( -1.5*d_amp, +1.5*d_amp )
ax.yaxis.set_major_formatter( FormatStrFormatter('%8.1e') )

# create movie
nframe = 99999999 # arbitrarily large
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()


