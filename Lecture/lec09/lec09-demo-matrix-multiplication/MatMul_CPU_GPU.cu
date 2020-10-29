#include <cstdio>


/* -----------------------------------------------------------
   Matrix Multiplication: C = A*B

   Reference: CUDA Toolkit Documentation
   -----------------------------------------------------------*/


// size of the thread block and thread grid
#define BLOCK_SIZE   16
#define GRID_SIZE_X  ( (C_WIDTH ) / (BLOCK_SIZE) )
#define GRID_SIZE_Y  ( (C_HEIGHT) / (BLOCK_SIZE) )



// *** matrix dimensions are assumed to be multiples of BLOCK_SIZE ***
#define A_HEIGHT     ( 2*(BLOCK_SIZE) )
#define A_WIDTH      ( 1*(BLOCK_SIZE) )

#define B_HEIGHT     ( A_WIDTH        )
#define B_WIDTH      ( 3*(BLOCK_SIZE) )

#define C_HEIGHT     ( A_HEIGHT       )
#define C_WIDTH      ( B_WIDTH        )



// -----------------------------------------------------------
// Matrix structure
//
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row*M.width + col)
// -----------------------------------------------------------
typedef struct {

   int    width;
   int    height;
   int    size;
   float* elements;

} Matrix;



// -----------------------------------------------------------
// CPU version of matrix multiplication
// -----------------------------------------------------------
void MatMul_CPU( Matrix A, Matrix B, Matrix C )
{
   for (int row=0; row<C.height; row++)
   for (int col=0; col<C.width;  col++)
   {
      float Cvalue = 0.0;

      for (int e=0; e<A.width; e++)
         Cvalue += A.elements[ row*A.width +   e ]
                 * B.elements[   e*B.width + col ];

      C.elements[ row*C.width + col ] = Cvalue;
   }
}



// -----------------------------------------------------------
// GPU version of matrix multiplication
// -----------------------------------------------------------
__global__
void MatMul_GPU( Matrix A, Matrix B, Matrix C )
{
// 2-D indices of thread blocks and threads
   int row = blockIdx.y*blockDim.y + threadIdx.y;
   int col = blockIdx.x*blockDim.x + threadIdx.x;

// each thread computes only one element of C
   float Cvalue = 0;

// compute the matrix element C(row,col)
   for (int e=0; e<A.width; e++)
      Cvalue += A.elements[ row*A.width +   e ]
              * B.elements[   e*B.width + col ];

// store data to the GPU "global" memory
   C.elements[ row*C.width + col ] = Cvalue;
}



// -----------------------------------------------------------
// host main function
// -----------------------------------------------------------
int main()
{

// -----------------------------------------------------------
// STEP 1: Declare and allocate host and device memories
// -----------------------------------------------------------
// prefix h: pointer to the "host   (CPU)" memory
// prefix d: pointer to the "device (GPU)" memory
   Matrix h_A, h_B, h_C;
   Matrix d_A, d_B, d_C;

// allocate host memory
   h_A.width  = A_WIDTH;
   h_B.width  = B_WIDTH;
   h_C.width  = C_WIDTH;

   h_A.height = A_HEIGHT;
   h_B.height = B_HEIGHT;
   h_C.height = C_HEIGHT;

   h_A.size   = h_A.height*h_A.width*sizeof(float);
   h_B.size   = h_B.height*h_B.width*sizeof(float);
   h_C.size   = h_C.height*h_C.width*sizeof(float);

   h_A.elements = (float*)malloc( h_A.size );
   h_B.elements = (float*)malloc( h_B.size );
   h_C.elements = (float*)malloc( h_C.size );

// allocate device memory
   d_A.width  = h_A.width;
   d_B.width  = h_B.width;
   d_C.width  = h_C.width;

   d_A.height = h_A.height;
   d_B.height = h_B.height;
   d_C.height = h_C.height;

   d_A.size   = h_A.size;
   d_B.size   = h_B.size;
   d_C.size   = h_C.size;

   cudaMalloc( &d_A.elements, d_A.size );
   cudaMalloc( &d_B.elements, d_B.size );
   cudaMalloc( &d_C.elements, d_C.size );


// -----------------------------------------------------------
// STEP 2: Initialize the host data
// -----------------------------------------------------------
   for (int row=0; row<h_A.height; row++)
   for (int col=0; col<h_A.width;  col++)
      h_A.elements[ row*h_A.width + col ] = row*h_A.width + col + 1.1;

   for (int row=0; row<h_B.height; row++)
   for (int col=0; col<h_B.width;  col++)
      h_B.elements[ row*h_B.width + col ] = row*h_B.width + col + 2.2;


// -----------------------------------------------------------
// STEP 3: Copy data from host to device (CPU -> GPU)
// -----------------------------------------------------------
   cudaMemcpy( d_A.elements, h_A.elements, h_A.size, cudaMemcpyHostToDevice );
   cudaMemcpy( d_B.elements, h_B.elements, h_B.size, cudaMemcpyHostToDevice );


// -----------------------------------------------------------
// STEP 4: Execute the GPU kernel
// -----------------------------------------------------------
   dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
   dim3 dimGrid( GRID_SIZE_X, GRID_SIZE_Y );

   MatMul_GPU <<< dimGrid, dimBlock >>> ( d_A, d_B, d_C );

// error handling
   cudaError_t ErrGPU = cudaGetLastError();
   if ( ErrGPU != cudaSuccess )
   {
      printf( "Kernel error: %s\n", cudaGetErrorString(ErrGPU) );
      exit( EXIT_FAILURE );
   }

// CPU counterpart
   Matrix h_C_CPU;
   h_C_CPU.width    = h_C.width;
   h_C_CPU.height   = h_C.height;
   h_C_CPU.size     = h_C.size;
   h_C_CPU.elements = (float*)malloc( h_C_CPU.size );

   MatMul_CPU( h_A, h_B, h_C_CPU );


// -----------------------------------------------------------
// STEP 5: Copy data from device to host (GPU -> CPU)
// -----------------------------------------------------------
   cudaMemcpy( h_C.elements, d_C.elements, h_C.size, cudaMemcpyDeviceToHost );



// compare the CPU and GPU results
   float Result_CPU, Result_GPU, Err, MaxErr=0.0f;

   for (int row=0; row<h_C.height; row++)
   for (int col=0; col<h_C.width;  col++)
   {
      Result_CPU = h_C_CPU.elements[ row*h_C.width + col ];
      Result_GPU = h_C    .elements[ row*h_C.width + col ];
      Err        = abs( (Result_CPU-Result_GPU) / Result_CPU );

      MaxErr = fmaxf( Err, MaxErr );

      /*
      printf( "C[%2d][%2d]: CPU = %14.7e, GPU = %14.7e, Err = %13.7e\n",
              row, col, Result_CPU, Result_GPU, Err );
              */
   }

   printf( "\nMaximum error = %13.7e\n\n", MaxErr );

   /*
// print A
   printf( "Matrix A\n" );
   for (int row=0; row<h_A.height; row++)
   {
      for (int col=0; col<h_A.width; col++)
         printf( "%5.1f ", h_A.elements[row*h_A.width+col] );

      printf( "\n" );
   }
   printf( "\n" );

// print B
   printf( "Matrix B\n" );
   for (int row=0; row<h_B.height; row++)
   {
      for (int col=0; col<h_B.width; col++)
         printf( "%5.1f ", h_B.elements[row*h_B.width+col] );

      printf( "\n" );
   }
   printf( "\n" );

// print C
   printf( "Matrix C = A*B\n" );
   for (int row=0; row<h_C.height; row++)
   {
      for (int col=0; col<h_C.width; col++)
         printf( "%5.1f ", h_C.elements[row*h_C.width+col] );

      printf( "\n" );
   }
   printf( "\n" );
   */


// free host memory
   free( h_A.elements );
   free( h_B.elements );
   free( h_C.elements );

// free device memory
   cudaFree( d_A.elements );
   cudaFree( d_B.elements );
   cudaFree( d_C.elements );
}
