
#--------------------------------------------------------------------
# Solve the diffusion equation with the FTCS scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
L   = 1.0   # 1-D computational domain size
N   = 100   # number of equally spaced sampling points
D   = 1.0   # diffusion coefficient
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 0.8   # Courant condition factor

# derived constants
dx      = L/(N-1)                # spatial resolution
dt      = cfl*0.5*dx**2.0/D      # time interval for data update
t_scale = (0.5*L/np.pi)**2.0/D   # diffusion time scale across L

# define a reference analytical solution
def ref_func( x, t ):
   k = 2.0*np.pi/L   # wavenumber
   return u0 + amp*np.sin( k*x )*np.exp( -k**2.0*D*t )

# initial condition
t = 0.0
x = np.linspace( 0.0, L, N )  # coordinates including both ends
u = ref_func( x, t )          # initial density distribution

# plotting parameters
end_time        = 2.0*t_scale # simulation time
nstep_per_image = 10          # plotting frequency

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

#     update all **interior** cells with the FTCS scheme
      for i in range( 1, N-1 ):
         u[i] = u_in[i] + dt*D*( u_in[i+1] - 2*u_in[i] + u_in[i-1] )/dx**2.0

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
   ax.set_title( 't/T = %7.4f, error = %10.3e' % (t/t_scale, err) )

   return line_num, line_ref


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()
