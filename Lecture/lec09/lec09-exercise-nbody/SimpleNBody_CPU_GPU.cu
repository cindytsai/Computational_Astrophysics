#include <cstdio>
#include <cstdlib>
#include <cmath>


/* -----------------------------------------------------------
   SimpleNBody_CPU_GPU: use CPU (or GPU) to evolve the direct N-Body system

   Assuming particle mass = 1 and G = 1
   -----------------------------------------------------------*/



// N-Body parameters
#define N                  8192              // number of particles (**must be a multiple of BLOCK_SIZE**)
#define BOX                1.0               // simulation box size
#define V_MAX              0.1               // maximum initial velocity
#define NSTEP              50                // total number of evolution steps
#define SOFTEN             1.0e-2            // soften length for calculating the gravitational acceleration

// GPU parameters
#define GPU_GLOBAL_SLOW    1                 // use the slower version of GPU global memory
#define GPU_GLOBAL_FAST    2                 // use the faster version of GPU global memory
#define GPU_SHARED         3                 // use the GPU shared memory
#define GPU                                  // use GPU (otherwise the CPU version will be used)
#define GPU_MODE           GPU_GLOBAL_SLOW   // set to GPU_GLOBAL_SLOW / GPU_GLOBAL_FAST / GPU_SHARED

#define BLOCK_SIZE         256
#define GRID_SIZE          ( N / BLOCK_SIZE )



// -----------------------------------------------------------
// CPU function for evolving particles
// -----------------------------------------------------------
void EvolveParticle_CPU( const int n, const float dt, float (*Pos)[3], float (*Vel)[3], const float (*Acc)[3] )
{

   for (int p=0; p<n; p++)
   for (int d=0; d<3; d++)
   {
//    first-order Euler integration (caution: advance Pos first)
      Pos[p][d] += Vel[p][d]*dt;
      Vel[p][d] += Acc[p][d]*dt;
   }

} // FUNCTION : EvolveParticle_CPU



#ifdef GPU
// -----------------------------------------------------------
// GPU function for evolving particles
// -----------------------------------------------------------
__global__
void EvolveParticle_GPU( const int n, const float dt, float (*Pos)[3], float (*Vel)[3], const float (*Acc)[3] )
{

// get the target particle index of each thread
   const int p = blockDim.x*blockIdx.x + threadIdx.x;


// if ( p < n ) --> no longer necessary since N = BLOCK_SIZE*GRID_SIZE (total number of threads)
   for (int d=0; d<3; d++)
   {
//    first-order Euler integration
      Pos[p][d] += Vel[p][d]*dt;
      Vel[p][d] += Acc[p][d]*dt;
   }

} // FUNCTION : EvolveParticle_GPU
#endif // #ifdef GPU



// -----------------------------------------------------------
// CPU function for calculating the pairwise acceleration
// -----------------------------------------------------------
void GetAcc_CPU( const int n, const float (*Pos)[3], float (*Acc)[3] )
{

   const float eps2 = SOFTEN*SOFTEN;   // prevent from large numerical errors during close encounters
   float dr[3], r, r3;

// calculate the acceleration from all "j" particles to each "i" particle
   for (int i=0; i<n; i++)
   {
//    must initialize to zero since we want to accumulate results later
      for (int d=0; d<3; d++)    Acc[i][d] = 0.0f;

//    loop over all j particles
      for (int j=0; j<n; j++)
      {
//       get distance
         for (int d=0; d<3; d++)    dr[d] = Pos[j][d] - Pos[i][d];

         r  = sqrtf( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2] + eps2 );
         r3 = r*r*r;

//       accumulate acceleration from all j particles (acc = r/|r|^3)
         for (int d=0; d<3; d++)    Acc[i][d] += dr[d]/r3;
      }
   }

} // FUNCTION : GetAcc_CPU



#ifdef GPU
// -----------------------------------------------------------
// GPU function for calculating the pairwise acceleration
// --> Using only the global memory
//
// ** Each thread calculates the acceleration from
//    all "j" particles to one "i" particle **
// -----------------------------------------------------------
__global__
void GetAcc_GPU_Global_Slow( const int n, const float (*g_Pos)[3], float (*g_Acc)[3] )
{

   const float eps2 = SOFTEN*SOFTEN;
   const uint  i    = blockDim.x*blockIdx.x + threadIdx.x;

   float r, r3, dr[3];


// initialize as zero
   g_Acc[i][0] = 0.0f;
   g_Acc[i][1] = 0.0f;
   g_Acc[i][2] = 0.0f;


// calculate the acceleration from all "j" particles
   for (int j=0; j<n; j++)
   {
//    OPTIMIZATION !?
      for (int d=0; d<3; d++)    dr[d] = g_Pos[j][d] - g_Pos[i][d];

      r  = sqrtf( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2] + eps2 );
      r3 = r*r*r;

//    OPTIMIZATION !?
      for (int d=0; d<3; d++)    g_Acc[i][d] += dr[d]/r3;
   }

} // FUNCTION : GetAcc_GPU_Global_Slow



__global__
void GetAcc_GPU_Global_Fast( const int n, const float (*g_Pos)[3], float (*g_Acc)[3] )
{

   const float eps2 = SOFTEN*SOFTEN;
   const uint  i    = blockDim.x*blockIdx.x + threadIdx.x;

   float r, r3, dr[3], I_Pos[3], Acc[3] = {0.0f, 0.0f, 0.0f};  // Acc only needs to be initialized once


// load data from global memory to the **per-thread** registers (much faster)
   for (int d=0; d<3; d++)    I_Pos[d] = g_Pos[i][d];


// calculate the acceleration from all "j" particles
   for (int j=0; j<n; j++)
   {
//    note that we don't need to access the positions of i particles
//    from the global memory anymore (use I_Pos instead of g_Pos)
//    for (int d=0; d<3; d++)    dr[d] = g_Pos[j][d] - g_Pos[i][d];
      for (int d=0; d<3; d++)    dr[d] = g_Pos[j][d] - I_Pos[d];

      r  = sqrtf( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2] + eps2 );
      r3 = r*r*r;

//    store temporary data in the per-thread variable "Acc" instead of g_Acc (much faster)
//    for (int d=0; d<3; d++)    g_Acc[i][d] += dr[d]/r3;
      for (int d=0; d<3; d++)    Acc[d] += dr[d]/r3;
   }

// store data back to the global memory (only needs to be done once)
   for (int d=0; d<3; d++)    g_Acc[i][d] = Acc[d];

} // FUNCTION : GetAcc_GPU_Global_Fast



// -----------------------------------------------------------
// GPU function for calculating the pairwise acceleration
// --> Taking advantage of the GPU shared memory
//
// ** Each thread calculates the acceleration from
//    all "j" particles to one "i" particle **
// -----------------------------------------------------------
__global__
void GetAcc_GPU_Shared( const int n, const float (*g_Pos)[3], float (*g_Acc)[3] )
{

   const float eps2 = SOFTEN*SOFTEN;
   const uint  tx   = threadIdx.x;
   const uint  i    = blockDim.x*blockIdx.x + tx;

// load data from global memory to the **per-thread** registers (much faster)
   const float I_Pos_x = g_Pos[i][0];
   const float I_Pos_y = g_Pos[i][1];
   const float I_Pos_z = g_Pos[i][2];

// declared shared-memory arrays
   __shared__ float sJ_Pos_x[BLOCK_SIZE];
   __shared__ float sJ_Pos_y[BLOCK_SIZE];
   __shared__ float sJ_Pos_z[BLOCK_SIZE];

   float Acc[3] = { 0.0f, 0.0f, 0.0f };
   float dx, dy, dz, r, r3;
   uint  j;


// calculate BLOCK_SIZE j particles at a time (because of the limited shared memory)
   for (int J_Base=0; J_Base<n; J_Base+=BLOCK_SIZE)
   {
      j = J_Base + tx;
      sJ_Pos_x[tx] = g_Pos[j][0];
      sJ_Pos_y[tx] = g_Pos[j][1];
      sJ_Pos_z[tx] = g_Pos[j][2];

//    synchronize all threads to ensure that all shared-memory data have been loaded
      __syncthreads();

//    k : kth particle in the currently loaded j particles
      for (int k=0; k<BLOCK_SIZE; k++)
      {
//       load data from the shared memory (for the j particle)
//       and the per-thread registers (for the i particles)
         dx = sJ_Pos_x[k] - I_Pos_x;
         dy = sJ_Pos_y[k] - I_Pos_y;
         dz = sJ_Pos_z[k] - I_Pos_z;

//       accumulate the gravitational acceleration
         r  = sqrtf( dx*dx + dy*dy + dz*dz + eps2 );
         r3 = r*r*r;

//       store temporary data in the per-thread variable "Acc" instead of g_Acc (much faster)
         Acc[0] += dx/r3;
         Acc[1] += dy/r3;
         Acc[2] += dz/r3;
      } // for (int k=0; k<GroupSize_J; k++)

//    synchronize all threads again before reloading data into the shared memory
      __syncthreads();
   } // for (int J_Base=0; J_Base<n; J_Base+=BLOCK_SIZE)

// store data back to the global memory
   g_Acc[i][0] = Acc[0];
   g_Acc[i][1] = Acc[1];
   g_Acc[i][2] = Acc[2];

} // FUNCTION : GetAcc_GPU_Shared
#endif // #ifdef GPU



// -----------------------------------------------------------
// output data
// -----------------------------------------------------------
void DumpData( const int Step, const float Pos[][3], const float Vel[][3], const float Acc[][3] )
{

   char FileName[100];
   sprintf( FileName, "Data_%d%d%d%d", Step/1000, (Step%1000)/100, (Step%100)/10, Step%10 );

   FILE *File = fopen( FileName, "w" );

   fprintf( File, "#%12s  %13s  %13s  %13s  %13s  %13s  %13s  %13s  %13s\n",
            "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az" );

   for (int p=0; p<N; p++)
      fprintf( File, "%13.6e  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e  %13.6e\n",
               Pos[p][0], Pos[p][1], Pos[p][2],
               Vel[p][0], Vel[p][1], Vel[p][2],
               Acc[p][0], Acc[p][1], Acc[p][2] );

   fclose( File );

} // FUNCTION : DumpData



// -----------------------------------------------------------
// estimate the evolution time-step by min( Acc^(-0.5) )
// --> very rough and should not be applied to real applications
// -----------------------------------------------------------
float GetTimeStep( const int n, const float Acc[][3] )
{

   const float Safety = 1.0e-2;        // safety factor
   float dt, a, dt_min = __FLT_MAX__;

   for (int p=0; p<n; p++)
   {
      a      = sqrtf( Acc[p][0]*Acc[p][0] + Acc[p][1]*Acc[p][1] + Acc[p][2]*Acc[p][2] );
      dt     = 1.0f/sqrt(a);
      dt_min = fminf( dt, dt_min );
   }

   return Safety*dt_min;

} // FUNCTION : GetTimeStep



// -----------------------------------------------------------
// main function
// -----------------------------------------------------------
int main( void )
{

// for simplicity, N must be a multiple of BLOCK_SIZE
   if ( N%BLOCK_SIZE != 0 )
   {
      fprintf( stderr, "N%%BLOCK_SIZE = %d != 0 !!\n", N%BLOCK_SIZE );
      exit( EXIT_FAILURE );
   }


// allocate host memory
   float (*h_Pos)[3], (*h_Vel)[3], (*h_Acc)[3];

   h_Pos = new float [N][3];
   h_Vel = new float [N][3];
   h_Acc = new float [N][3];


#  ifdef GPU
// allocate device memory
   float (*d_Pos)[3], (*d_Vel)[3], (*d_Acc)[3];

   cudaMalloc( &d_Pos, N*3*sizeof(float) );
   cudaMalloc( &d_Vel, N*3*sizeof(float) );
   cudaMalloc( &d_Acc, N*3*sizeof(float) );
#  endif


// initialize particles
   const uint RSeed = 1234;      // random seed
   const float MaxR = 0.5*BOX;   // maximum radius for the initial particle position
   float r;

   srand( RSeed );

   for (int p=0; p<N; p++)
   {
      r = MaxR + 1.0;

//    ensure r <= MaxR
      while ( r > MaxR )
      {
         for (int d=0; d<3; d++)
         h_Pos[p][d] = ( (float)rand()/RAND_MAX )*BOX - 0.5*BOX;

         r = sqrtf( h_Pos[p][0]*h_Pos[p][0] + h_Pos[p][1]*h_Pos[p][1] + h_Pos[p][2]*h_Pos[p][2] );
      }

//    -V_Max < v < V_Max
      for (int d=0; d<3; d++)
      h_Vel[p][d] = ( (float)rand()/RAND_MAX )*2*V_MAX - V_MAX;
   }


// calculate the initial acceleration
#  ifdef GPU
   cudaMemcpy( d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice );

#  if   ( GPU_MODE == GPU_GLOBAL_SLOW )
   GetAcc_GPU_Global_Slow <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#  elif ( GPU_MODE == GPU_GLOBAL_FAST )
   GetAcc_GPU_Global_Fast <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#  elif ( GPU_MODE == GPU_SHARED )
   GetAcc_GPU_Shared      <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#  else
#  error : unsupported GPU_MODE !!
#  endif

   cudaMemcpy( h_Acc, d_Acc, N*3*sizeof(float), cudaMemcpyDeviceToHost );
#  else

   GetAcc_CPU( N, h_Pos, h_Acc );
#  endif


// dump the initial data
   DumpData( 0, h_Pos, h_Vel, h_Acc );


// evolution loop
   for (int s=0; s<NSTEP; s++)
   {
//    estimate the evolution time-step
      const float dt = GetTimeStep( N, h_Acc );

      fprintf( stdout, "Step %4d --> %4d: t = %13.7e --> %13.7e (dt = %13.7e)\n", s, s+1, s*dt, (s+1)*dt, dt );


//    evolve particles
#     ifdef GPU
      cudaMemcpy( d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice );
      cudaMemcpy( d_Vel, h_Vel, N*3*sizeof(float), cudaMemcpyHostToDevice );
      cudaMemcpy( d_Acc, h_Acc, N*3*sizeof(float), cudaMemcpyHostToDevice );

      EvolveParticle_GPU <<< GRID_SIZE, BLOCK_SIZE >>> ( N, dt, d_Pos, d_Vel, d_Acc );

      cudaMemcpy( h_Pos, d_Pos, N*3*sizeof(float), cudaMemcpyDeviceToHost );
      cudaMemcpy( h_Vel, d_Vel, N*3*sizeof(float), cudaMemcpyDeviceToHost );

#     else

      EvolveParticle_CPU( N, dt, h_Pos, h_Vel, h_Acc );
#     endif


//    calculate acceleration for the next step
#     ifdef GPU
      cudaMemcpy( d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice );

#     if   ( GPU_MODE == GPU_GLOBAL_SLOW )
      GetAcc_GPU_Global_Slow <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#     elif ( GPU_MODE == GPU_GLOBAL_FAST )
      GetAcc_GPU_Global_Fast <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#     elif ( GPU_MODE == GPU_SHARED )
      GetAcc_GPU_Shared      <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_Pos, d_Acc );
#     else
#     error : unsupported GPU_MODE !!
#     endif

      cudaMemcpy( h_Acc, d_Acc, N*3*sizeof(float), cudaMemcpyDeviceToHost );
#     else

      GetAcc_CPU( N, h_Pos, h_Acc );
#     endif


//    dump data
      DumpData( s+1, h_Pos, h_Vel, h_Acc );
   } // for (int s=0; s<NSTEP; s++)


// free host and device memories
   delete [] h_Pos;
   delete [] h_Vel;
   delete [] h_Acc;

#  ifdef GPU
   cudaFree( d_Pos );
   cudaFree( d_Vel );
   cudaFree( d_Acc );
#  endif

} // FUNCTION : main
