
#--------------------------------------------------------------------
# Solve the advection equation with the Lax-Wendroff scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
L   = 1.0   # 1-D computational domain size
N   = 800   # number of computing cells
v   = 1.0   # advection velocity
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 0.8   # Courant condition factor

# derived constants
dx     = L/N      # spatial resolution
dt     = cfl*dx/v # time interval for data update
period = L/v      # time period

# define a reference analytical solution
def ref_func( x, t ):
   k = 2.0*np.pi/L   # wavenumber
   return u0 + amp*np.sin( k*(x-v*t) )

# initial condition
t = 0.0
x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates
u = ref_func( x, t )                   # initial density distribution

# plotting parameters
end_time        = 2.0*period  # simulation time
nstep_per_image = 1           # plotting frequency

# create figure
fig       = plt.figure( figsize=(6,6), dpi=140 )
ax        = plt.axes( xlim=(0.0,L), ylim=(u0-amp*1.5,u0+amp*1.5) )
line_num, = ax.plot( [], [], 'r', ls='-',  label='Numerical' )
line_ref, = ax.plot( [], [], 'b', ls='--', label='Reference' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'u' )
ax.tick_params( top=False, right=True, labeltop=False, labelright=True )

def init():
   line_num.set_xdata( x )
   line_ref.set_xdata( x )
   return line_num, line_ref

def update( frame ):
   global t, u

   for step in range( nstep_per_image ):
#     back up the input data
      u_in = u.copy()

#     calculate the half-timestep solution
      u_half = np.empty( N )
      for i in range( N ):
#        u_half[i] is defined at the left face of cell i
         im = (i-1+N) % N  # assuming periodic boundary condition
         u_half[i] = 0.5*( u_in[i] + u_in[im] ) - 0.5*dt*v*( u_in[i] - u_in[im] )/dx

#     update all cells
      for i in range( N ):
         ip = (i+1) % N    # assuming periodic boundary condition
         u[i] = u_in[i] - dt*v*( u_half[ip] - u_half[i] )/dx

#     update time
      t = t + dt
      if ( t >= end_time ):   break

#  calculate the reference analytical solution and estimate errors
   u_ref = ref_func( x, t )
   err   = np.abs( u_ref - u ).sum()/N

#  plot
   line_num.set_ydata( u )
   line_ref.set_ydata( u_ref )
   ax.legend( loc='upper right', fontsize=12 )
   ax.set_title( 't/T = %6.3f, error = %10.3e' % (t/period, err) )

   return line_num, line_ref


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()
