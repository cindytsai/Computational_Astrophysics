#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/* -----------------------------------------------------------
   SimpleNBody_CPU: use CPU to evolve the direct N-Body system

   Assuming particle mass = 1 and G = 1
   -----------------------------------------------------------*/



// N-Body parameters
#define N                  8192              // number of particles (**must be a multiple of BLOCK_SIZE**)
#define BOX                1.0               // simulation box size
#define V_MAX              0.1               // maximum initial velocity
#define NSTEP              50                // total number of evolution steps
#define SOFTEN             1.0e-2            // soften length for calculating the gravitational acceleration



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

// -----------------------------------------------------------
// GPU function for evolving particles
// -----------------------------------------------------------
__global__ void EvolveParticle_GPU( const int n, const float dt, float (*Pos)[3], float (*Vel)[3], const float (*Acc)[3] )
{
    // One block compute one particle
    // Each thread compute 1 dim 

    int p = blockDim.x;
    int d = threadIdx.x;

    Pos[p][d] += Vel[p][d]*dt;
    Vel[p][d] += Acc[p][d]*dt;

}


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

// -----------------------------------------------------------
// GPU function for calculating the pairwise acceleration
// -----------------------------------------------------------
__global__ void GetAcc_GPU( const int n, const float (*Pos)[3], float (*Acc)[3] )
{
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    const float eps2 = SOFTEN * SOFTEN;
    float dr[3], r, r3;
    
    while(index < n){
        for(int j = 0; j < n; j = j+1){
            // Get distance
            for(int d = 0; d < 3; d = d+1){
                dr[d] = Pos[j][d] - Pos[index][d];
            }

            r = sqrtf( dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2] + eps2 );
            r3 = powf(r, 3);

            for(int d = 0; d < 3; d = d+1){
                Acc[index][d] = Acc[index][d] + dr[d] / r3;
            }
        }

        index = index + blockDim.x * gridDim.x;
    }
}

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

// for simplicity, we can assume N to be a multiple of BLOCK_SIZE
   /*
   if ( N%BLOCK_SIZE != 0 )
   {
      fprintf( stderr, "N%%BLOCK_SIZE = %d != 0 !!\n", N%BLOCK_SIZE );
      exit( EXIT_FAILURE );
   }
   */


// allocate host memory
   float (*h_Pos)[3], (*h_Vel)[3], (*h_Acc)[3];

   h_Pos = new float [N][3];
   h_Vel = new float [N][3];
   h_Acc = new float [N][3];


// ** for GPU: allocate device memory and set block size and grid size **
    int BlockSize = 3;
    int GridSize = N;
    float (*d_Pos)[3], (*d_Vel)[3], (*d_Acc)[3];
    cudaMalloc((void**) &d_Pos, N*3*sizeof(float));
    cudaMalloc((void**) &d_Vel, N*3*sizeof(float));
    cudaMalloc((void**) &d_Acc, N*3*sizeof(float));

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
   // GetAcc_CPU( N, h_Pos, h_Acc );

// ** for GPU: **
// 1. transfer Pos from CPU to GPU
   cudaMemcpy(d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice);
// 2. invoke the GPU kernel
   GetAcc_GPU <<<10, 128>>>( N, d_Pos, d_Acc );
// 3. transfer Acc from GPU to CPU
   cudaMemcpy(h_Acc, d_Acc, N*3*sizeof(float), cudaMemcpyDeviceToHost);

// dump the initial data
   DumpData( 0, h_Pos, h_Vel, h_Acc );


// evolution loop
   for (int s=0; s<NSTEP; s++)
   {
//    estimate the evolution time-step
      const float dt = GetTimeStep( N, h_Acc );

      fprintf( stdout, "Step %4d --> %4d: t = %13.7e --> %13.7e (dt = %13.7e)\n", s, s+1, s*dt, (s+1)*dt, dt );


//    evolve particles
      // EvolveParticle_CPU( N, dt, h_Pos, h_Vel, h_Acc );

//    ** for GPU: **
//    1. transfer Pos, Vel, Acc from CPU to GPU
      cudaMemcpy(d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_Vel, h_Vel, N*3*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_Acc, h_Acc, N*3*sizeof(float), cudaMemcpyHostToDevice);

//    2. invoke the GPU kernel
      EvolveParticle_GPU<<<GridSize, BlockSize>>>(N, dt, d_Pos, d_Vel, d_Acc);

//    3. transfer Pos and Vel from GPU to CPU
      cudaMemcpy(h_Pos, d_Pos, N*3*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_Vel, d_Vel, N*3*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_Acc, d_Acc, N*3*sizeof(float), cudaMemcpyDeviceToHost);

//    calculate acceleration for the next step
      // GetAcc_CPU( N, h_Pos, h_Acc );

//    ** for GPU: **
//    1. transfer Pos from CPU to GPU
      cudaMemcpy(d_Pos, h_Pos, N*3*sizeof(float), cudaMemcpyHostToDevice);
//    2. invoke the GPU kernel
      GetAcc_GPU <<<10, 128>>>( N, d_Pos, d_Acc );
//    3. transfer Acc from GPU to CPU
      cudaMemcpy(h_Acc, d_Acc, N*3*sizeof(float), cudaMemcpyDeviceToHost);

//    dump data
      DumpData( s+1, h_Pos, h_Vel, h_Acc );
   } // for (int s=0; s<NSTEP; s++)


// free host memory
   delete [] h_Pos;
   delete [] h_Vel;
   delete [] h_Acc;

// ** for GPU: free device memory **
   cudaFree(d_Pos);
   cudaFree(d_Vel);
   cudaFree(d_Acc);

} // FUNCTION : main
