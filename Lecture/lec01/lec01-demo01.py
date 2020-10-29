import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
G  = 1.0       # gravitational constant
M  = 2.0       # central point mass
dt = 1.0e-2    # time interval for data update

# initial condition
t     = 0.0
x     = 1.0
y     = 0.0
r     = ( x**2 + y**2 )**0.5
vx    = 0.0
vy    = ( G*M/r )**0.5
v_abs = ( vx**2 + vy**2 )**0.5
E0    = 0.5*v_abs**2 - G*M/r

# plotting parameters
period          = 2.0*np.pi*r/v_abs
end_time        = 1.0*period
nstep_per_image = 1

# create figure
fig   = plt.figure( figsize=(6,6), dpi=100 )
ax    = plt.axes( xlim=(-1.5,+1.5), ylim=(-1.5,+1.5) )
ball, = ax.plot( [], [], 'ro', ms=10 )
text  = ax.text( 0.0, 1.3, '', fontsize=16, color='black',
                 ha='center', va='center' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_aspect( 'equal' )
ax.tick_params( top=True, right=True, labeltop=True, labelright=True )
ax.add_artist( plt.Circle( (0.0, 0.0), r, color='b', fill=False ) )

def init():
   ball.set_data( [], [] )
   text.set( text='' )
   return ball, text

def update_orbit( i ):
   global t, x, y, vx, vy

   for step in range( nstep_per_image ):
#     calculate acceleration
      r     = ( x*x + y*y )**0.5
      a_abs = G*M/(r*r)
      ax    = -a_abs*x/r
      ay    = -a_abs*y/r

#     update orbit (Euler's method)
      x  = x + vx*dt
      y  = y + vy*dt
      vx = vx + ax*dt
      vy = vy + ay*dt

#     update time
      t = t + dt
      if ( t >= end_time ):   break

#  calculate energy error
   E   = 0.5*( vx**2 + vy**2 ) - G*M/r
   err = (E-E0)/E0

#  plot
   ball.set_data( x, y )
   text.set( text='t/T = %6.3f, error = %10.3e' % (t/period, err) )

   return ball, text


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update_orbit, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()
