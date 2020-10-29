#include <cstdio>
#include <cstdlib>


/* -----------------------------------------------------------
   SAXPY: compute Single-precision A*X + Y

   Reference: http://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/
              --> by Mark Harris
   -----------------------------------------------------------*/


// -----------------------------------------------------------
// GPU version of SAXPY
// -----------------------------------------------------------
__global__
void saxpy_GPU( const int n, const float a, float *x, float *y, float *z )
{
// "__global__" --> GPU kernel
// n, a: passed by value (transferred to the device implicitly)
// x, y, z: pointers to the device global memory (must be preallocated and transferred to the device explicitly)

// blockIdx.x : index of the thread block
// blockDim.x : size of the thread block
// threadIdx.x: thread index within a thread block
// --> i      : global index to distinguish all threads
   int i = blockIdx.x*blockDim.x + threadIdx.x;

// caution: i can be >= n if N%NTHREAD_PER_BLOCK != 0
   if ( i < n )   z[i] = a*x[i] + y[i];
}



// -----------------------------------------------------------
// CPU version of SAXPY
// -----------------------------------------------------------
void saxpy_CPU( const int n, const float a, float *x, float *y, float *z )
{
   for (int i=0; i<n; i++)
   z[i] = a*x[i] + y[i];
}



// -----------------------------------------------------------
// host main function
// -----------------------------------------------------------
int main( void )
{

   const float A = 2.3;
   const int   N = 1024;   // array size

// number of threads per thread block (also refer to as BLOCK_SIZE)
   const int NTHREAD_PER_BLOCK = 256;

// number of thread blocks (also refer to as GRID_SIZE)
// --> make sure that there are enough threads to access all N elements in the array
   const int NBLOCK = (N+NTHREAD_PER_BLOCK-1)/NTHREAD_PER_BLOCK;


// -----------------------------------------------------------
// STEP 1: Declare and allocate host and device memories
// -----------------------------------------------------------
// prefix h: pointer to the "host   (CPU)" memory
// prefix d: pointer to the "device (GPU)" memory
   float *h_x, *h_y, *h_z, *d_x, *d_y, *d_z;

// allocate host memory
   h_x = (float*)malloc( N*sizeof(float) );
   h_y = (float*)malloc( N*sizeof(float) );
   h_z = (float*)malloc( N*sizeof(float) );

// allocate device memory
   cudaMalloc( &d_x, N*sizeof(float) );
   cudaMalloc( &d_y, N*sizeof(float) );
   cudaMalloc( &d_z, N*sizeof(float) );


// -----------------------------------------------------------
// STEP 2: Initialize the host data
// -----------------------------------------------------------
   for (int i=0; i<N; i++)
   {
      h_x[i] = i + 1.0f;
      h_y[i] = i + 2.0f;
   }


// -----------------------------------------------------------
// STEP 3: Copy data from host to device (CPU -> GPU)
// -----------------------------------------------------------
   cudaMemcpy( d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice );


// -----------------------------------------------------------
// STEP 4: Execute the GPU kernel
// -----------------------------------------------------------
   saxpy_GPU <<< NBLOCK, NTHREAD_PER_BLOCK >>> ( N, A, d_x, d_y, d_z );

// CPU counterpart
   float *h_z_CPU = (float*)malloc( N*sizeof(float) );
   memcpy( h_z_CPU, h_z, N*sizeof(float) );

   saxpy_CPU( N, A, h_x, h_y, h_z_CPU );


// -----------------------------------------------------------
// STEP 5: Copy data from device to host (GPU -> CPU)
// -----------------------------------------------------------
   cudaMemcpy( h_z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost );


// compare CPU and GPU results
   float MaxErr=0.0f, Result_CPU, Result_GPU, Err;

   for (int i = 0; i < N; i++)
   {
      Err = fabsf(  ( h_z_CPU[i] - h_z[i] ) / h_z_CPU[i]  );

      if ( Err > MaxErr )
      {
         MaxErr     = Err;
         Result_CPU = h_z_CPU[i];
         Result_GPU = h_z    [i];
      }
   }

   printf( "Max error : %13.7e\n", MaxErr     );
   printf( "CPU result: %13.7e\n", Result_CPU );
   printf( "GPU result: %13.7e\n", Result_GPU );


// free host memory
   free( h_x );
   free( h_y );
   free( h_z );
   free( h_z_CPU );

// free device memory
   cudaFree( d_x );
   cudaFree( d_y );
   cudaFree( d_z );

}
