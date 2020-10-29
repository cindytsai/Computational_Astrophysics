#include <cstdio>
#include <cstdlib>
#include <cmath>


/* -----------------------------------------------------------
   Smoothing: compute A[i] = 0.25*A[i-1] + 0.5*A[i] + 0.25*A[i+1]
   -----------------------------------------------------------*/


#define N            1024     // array size

#define BLOCK_SIZE   16
#define GRID_SIZE    ( (N+BLOCK_SIZE-1) / BLOCK_SIZE )



// -----------------------------------------------------------
// GPU kernel using global memory only
// -----------------------------------------------------------
__global__
void Smooth_Global( const int n, float *In, float *Out )
{

   int i, im, ip;

   i  = blockIdx.x*blockDim.x + threadIdx.x;
   im = (i+n-1)%n;   // assuming periodicity
   ip = (i  +1)%n;   // assuming periodicity

   if ( i < n )
   Out[i] = 0.25f*In[im] + 0.5f*In[i] + 0.25f*In[ip];

}



// -----------------------------------------------------------
// GPU kernel taking advantage of the shared memory
// -----------------------------------------------------------
__global__
void Smooth_Shared( const int n, float *In, float *Out )
{

   int i, im, ip;

// declare shared memory ("+2" for the ghost-zone data)
   __shared__ float s_In[BLOCK_SIZE+2];


// each thread loads only ONE element into the shared-memory array
   i = blockIdx.x*blockDim.x + threadIdx.x;
   s_In[threadIdx.x+1] = In[i];


// use 0-th thread to load the ghost-zone data
   if ( threadIdx.x == 0 )
   {
      im = (blockIdx.x*blockDim.x              + n - 1 ) % n;  // assuming periodicity
      ip = (blockIdx.x*blockDim.x + blockDim.x         ) % n;  // assuming periodicity

      s_In[           0] = In[im];
      s_In[BLOCK_SIZE+1] = In[ip];
   }


// ** synchronize all threads within the same thread block **
// --> ensure that all data have been loaded into the shared memory BEFORE doing any calculation
   __syncthreads();


// use the shared-memory array for the calculations
   if ( i < n )
   Out[i] = 0.25f*s_In[threadIdx.x] + 0.5f*s_In[threadIdx.x+1] + 0.25f*s_In[threadIdx.x+2];
}



// -----------------------------------------------------------
// main function
// -----------------------------------------------------------
int main( void )
{

   float *h_In, *h_Out;
   float *d_In, *d_Out;

// allocate host memory
   h_In  = (float*)malloc( N*sizeof(float) );
   h_Out = (float*)malloc( N*sizeof(float) );

// allocate device memory
   cudaMalloc( &d_In,  N*sizeof(float) );
   cudaMalloc( &d_Out, N*sizeof(float) );


// initialize the host array (--> sin(x) + perturbation)
   const uint  RSeed = 0;        // random seed
   const float Pert  = 1.0e-1;   // perturbation amplitude

   srand( RSeed );

   for (int i=0; i<N; i++)
      h_In[i] = sinf( 2.0*M_PI/N*i ) + ( (float)rand()/RAND_MAX )*2.0*Pert - Pert;


// transfer data from CPU to GPU
   cudaMemcpy( d_In, h_In, N*sizeof(float), cudaMemcpyHostToDevice );


// execute the GPU kernel
// Smooth_Global <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_In, d_Out );
   Smooth_Shared <<< GRID_SIZE, BLOCK_SIZE >>> ( N, d_In, d_Out );

// error handling
   cudaError_t ErrGPU = cudaGetLastError();
   if ( ErrGPU != cudaSuccess )
   {
      printf( "Kernel error: %s\n", cudaGetErrorString(ErrGPU) );
      exit( EXIT_FAILURE );
   }


// transfer data from GPU to CPU
   cudaMemcpy( h_Out, d_Out, N*sizeof(float), cudaMemcpyDeviceToHost );


// dump data
   FILE *File = fopen( "Data", "w" );

   fprintf( File, "#%12s   %14s   %14s\n", "x", "pre-smoothing", "post-smoothing" );
   for (int i=0; i<N; i++)
      fprintf( File, "%13.7e   %14.7e   %14.7e\n", 2.0*M_PI/N*i, h_In[i], h_Out[i] );

   fclose( File );


// free host and device memories
   free( h_In  );
   free( h_Out );

   cudaFree( d_In  );
   cudaFree( d_Out );

}
