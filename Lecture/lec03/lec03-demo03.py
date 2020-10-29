
#--------------------------------------------------------------------
# Simulate acoustic wave with the MUSCL-Hancock scheme
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
nghost   = 2         # number of ghost zones
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

   u1 = cs*d_amp/d0        # velocity perturbation
   P0 = cs**2.0*d0/gamma   # background pressure
   P1 = cs**2.0*d_amp      # pressure perturbation

#  d/u/v/w/P/e = density/velocity-x/velocity-y/velocity-z/pressure/total energy
   d = d0 + d_amp*np.cos(Phase)
   u = u1*np.cos(Phase)
   v = 0.0
   w = 0.0
   P = P0 + P1*np.cos(Phase)
   E = P/(gamma-1.0) + 0.5*d*( u**2.0 + v**2.0 + w**2.0 )

#  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy]
   return np.array( [d, d*u, d*v, d*w, E] )


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
def ComputePressure( d, px, py, pz, e ):
   P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d )
   return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
   P = ComputePressure( U[:,0], U[:,1], U[:,2], U[:,3], U[:,4] )
   a = ( gamma*P/U[:,0] )**0.5
   u = np.abs( U[:,1]/U[:,0] )
   v = np.abs( U[:,2]/U[:,0] )
   w = np.abs( U[:,3]/U[:,0] )

#  maximum information speed in 3D
   max_info_speed = np.amax( u + a )
   dt_cfl         = cfl*dx/max_info_speed
   dt_end         = end_time - t

   return min( dt_cfl, dt_end )


# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope( L, C, R ):
#  compute the left and right slopes
   slope_L = C - L
   slope_R = R - C

#  apply the van-Leer limiter
   slope_LR      = slope_L*slope_R
   slope_limited = np.where( slope_LR>0.0, 2.0*slope_LR/(slope_L+slope_R), 0.0 )

   return slope_limited


# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive( U ):
   W = np.empty( 5 )

   W[0] = U[0]
   W[1] = U[1]/U[0]
   W[2] = U[2]/U[0]
   W[3] = U[3]/U[0]
   W[4] = ComputePressure( U[0], U[1], U[2], U[3], U[4] )

   return W


# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved( W ):
   U = np.empty( 5 )

   U[0] = W[0]
   U[1] = W[0]*W[1]
   U[2] = W[0]*W[2]
   U[3] = W[0]*W[3]
   U[4] = W[4]/(gamma-1.0) + 0.5*W[0]*( W[1]**2.0 + W[2]**2.0 + W[3]**2.0 )

   return U


# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
def DataReconstruction_PLM( U ):

#  allocate memory
   W = np.empty( (N,5) )
   L = np.empty( (N,5) )
   R = np.empty( (N,5) )

#  conserved variables --> primitive variables
   for j in range( N ):
      W[j] = Conserved2Primitive( U[j] )

   for j in range( 1, N-1 ):
#     compute the left and right states of each cell
      slope_limited = ComputeLimitedSlope( W[j-1], W[j], W[j+1] )

#     get the face-centered variables
      L[j] = W[j] - 0.5*slope_limited
      R[j] = W[j] + 0.5*slope_limited

#     ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
      L[j] = np.maximum( L[j], np.minimum( W[j-1], W[j] ) )
      L[j] = np.minimum( L[j], np.maximum( W[j-1], W[j] ) )
      R[j] = 2.0*W[j] - L[j]

      R[j] = np.maximum( R[j], np.minimum( W[j+1], W[j] ) )
      R[j] = np.minimum( R[j], np.maximum( W[j+1], W[j] ) )
      L[j] = 2.0*W[j] - R[j]

#     primitive variables --> conserved variables
      L[j] = Primitive2Conserved( L[j] )
      R[j] = Primitive2Conserved( R[j] )

   return L, R


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
def Conserved2Flux( U ):
   flux = np.empty( 5 )

   P = ComputePressure( U[0], U[1], U[2], U[3], U[4] )
   u = U[1] / U[0]

   flux[0] = U[1]
   flux[1] = u*U[1] + P
   flux[2] = u*U[2]
   flux[3] = u*U[3]
   flux[4] = u*( U[4] + P )

   return flux


# -------------------------------------------------------------------
# Roe's Riemann solver
# -------------------------------------------------------------------
def Roe( L, R ):
#  compute the enthalpy of the left and right states: H = (E+P)/rho
   P_L = ComputePressure( L[0], L[1], L[2], L[3], L[4] )
   P_R = ComputePressure( R[0], R[1], R[2], R[3], R[4] )
   H_L = ( L[4] + P_L )/L[0]
   H_R = ( R[4] + P_R )/R[0]

#  compute Roe average values
   rhoL_sqrt = L[0]**0.5
   rhoR_sqrt = R[0]**0.5

   u  = ( L[1]/rhoL_sqrt + R[1]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   v  = ( L[2]/rhoL_sqrt + R[2]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   w  = ( L[3]/rhoL_sqrt + R[3]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   H  = ( rhoL_sqrt*H_L  + rhoR_sqrt*H_R  ) / ( rhoL_sqrt + rhoR_sqrt )
   V2 = u*u + v*v + w*w
#  check negative pressure
   assert H-0.5*V2 > 0.0, "negative pressure!"
   a  = ( (gamma-1.0)*(H - 0.5*V2) )**0.5

#  compute the amplitudes of different characteristic waves
   dU     = R - L
   amp    = np.empty( 5 )
   amp[2] = dU[2] - v*dU[0]
   amp[3] = dU[3] - w*dU[0]
   amp[1] = (gamma-1.0)/a**2.0 \
            *( dU[0]*(H-u**2.0) + u*dU[1] - dU[4] + v*amp[2] + w*amp[3] )
   amp[0] = 0.5/a*( dU[0]*(u+a) - dU[1] - a*amp[1] )
   amp[4] = dU[0] - amp[0] - amp[1]

#  compute the eigenvalues and right eigenvector matrix
   EigenValue    = np.array( [u-a, u, u, u, u+a] )
   EigenVector_R = np.array(  [ [1.0, u-a,   v,   w,  H-u*a],
                                [1.0,   u,   v,   w, 0.5*V2],
                                [0.0, 0.0, 1.0, 0.0,      v],
                                [0.0, 0.0, 0.0, 1.0,      w],
                                [1.0, u+a,   v,   w,  H+u*a] ]  )

#  compute the fluxes of the left and right states
   flux_L = Conserved2Flux( L )
   flux_R = Conserved2Flux( R )

#  compute the Roe flux
   amp *= np.abs( EigenValue )
   flux = 0.5*( flux_L + flux_R ) - 0.5*amp.dot( EigenVector_R )

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

#        data reconstruction
         L, R = DataReconstruction_PLM( U )

#        update the face-centered variables by 0.5*dt
         for j in range( 1, N-1 ):
            flux_L = Conserved2Flux( L[j] )
            flux_R = Conserved2Flux( R[j] )
            dflux  = 0.5*dt/dx*( flux_R - flux_L )
            L[j]  -= dflux
            R[j]  -= dflux

#        compute fluxes
         flux = np.empty( (N,5) )
         for j in range( nghost, N-nghost+1 ):
#           R[j-1] is the LEFT state at the j+1/2 inteface
            flux[j] = Roe( R[j-1], L[j]  )

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
U     = np.empty( (N,5) )
U_ref = np.empty( (N,5) )
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


