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
// for (int i=0; i<n; i++)
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   if ( i < n )
   z[i] = a*x[i] + y[i];
}



// -----------------------------------------------------------
// main function
// -----------------------------------------------------------
int main( void )
{

   const float A = 2.0;
   const int   N = 10;     // array size

   float *h_x, *h_y, *h_z;
   float *d_x, *d_y, *d_z;

// allocate host memory
   h_x = (float*)malloc( N*sizeof(float) );
   h_y = (float*)malloc( N*sizeof(float) );
   h_z = (float*)malloc( N*sizeof(float) );

// allocate device memory
   cudaMalloc( &d_x, N*sizeof(float) );
   cudaMalloc( &d_y, N*sizeof(float) );
   cudaMalloc( &d_z, N*sizeof(float) );


   for (int i=0; i<N; i++)
   {
      h_x[i] = i + 1.0f;
      h_y[i] = i + 2.0f;
   }


// transfer data from CPU to GPU
   cudaMemcpy( d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice );


// saxpy_CPU( N, A, h_x, h_y, h_z );
// execute the GPU kernel
   const int NThread_Per_Block = 128;
   const int NBlock            = ( N + NThread_Per_Block - 1 ) / NThread_Per_Block;

   saxpy_GPU <<< NBlock, NThread_Per_Block >>> ( N, A, d_x, d_y, d_z );


// transfer data from GPU to CPU
   cudaMemcpy( h_z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost );


   for (int i = 0; i < N; i++)
      printf( "%4.1f*%4.1f + %4.1f = %4.1f\n", A, h_x[i], h_y[i], h_z[i] );


   free( h_x );
   free( h_y );
   free( h_z );

   cudaFree( d_x );
   cudaFree( d_y );
   cudaFree( d_z );

}
