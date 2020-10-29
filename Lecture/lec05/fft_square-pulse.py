
#--------------------------------------------------------------------
# Convolution with DFT
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# constants
L       = 1.0     # 1-D domain size
N       = 100     # number of equally spaced sampling points
lambda1 = L       # wavelength of component 1
lambda2 = L/N*3   # wavelength of component 2
amp1    = 1.0     # amplitude of component 1
amp2    = 1.0e-1  # amplitude of component 2
dx      = L/N     # spatial resolution

# set the input data
x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates
#u = amp1*np.sin( 2.0*np.pi/lambda1*x ) + \
#    amp2*np.sin( 2.0*np.pi/lambda2*x )

interval0 = [1 if (i<0.50) else 0 for i in x]
interval1 = [1 if (i>=0.50 and i<0.80) else 0 for i in x]
interval2 = [1 if (i>=0.80) else 0 for i in x]
u = interval0 + (1 - x)* interval1 + interval2 


# define a convolution filter
#f  = np.array( [1.0, 2.0, 4.0, 2.0, 1.0] )
f  = np.array( [-1.0, -1.0, 4.0, -1.0, -1.0] ) *0.5
#f /= f.sum()                  # normalization
f_pad0 = np.zeros( u.size )   # zero-padded filter
f_pad0[ 0:f.size ] = f
f_pad0 = np.roll( f_pad0, -(f.size//2) )  # f_pad0 = [0.4, 0.2, 0.1, 0.0, ..., 0.0, 0.1, 0.2]

# convolution
uk = np.fft.rfft( u )
fk = np.fft.rfft( f_pad0 )
u_con = np.fft.irfft( uk*fk )

# create figure
fig       = plt.figure( figsize=(6,6), dpi=140 )
ax        = plt.axes()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'u' )

ax.plot( x, u,     'r', ls='-', label='Before convolution' )
ax.plot( x, u_con, 'b', ls='-', label='After convolution' )

ax.legend( loc='lower right', fontsize=12 )

plt.show()
