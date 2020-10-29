#include <cstdio>
#include <cstdlib>


/* -----------------------------------------------------------
   SAXPY: compute Single-precision A*X + Y

   Reference: http://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/
              --> by Mark Harris
   -----------------------------------------------------------*/


// -----------------------------------------------------------
// CPU version of SAXPY
// -----------------------------------------------------------
void saxpy_CPU( const int n, const float a, float *x, float *y, float *z )
{
   for (int i=0; i<n; i++)
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

   h_x = (float*)malloc( N*sizeof(float) );
   h_y = (float*)malloc( N*sizeof(float) );
   h_z = (float*)malloc( N*sizeof(float) );


   for (int i=0; i<N; i++)
   {
      h_x[i] = i + 1.0f;
      h_y[i] = i + 2.0f;
   }


   saxpy_CPU( N, A, h_x, h_y, h_z );


   for (int i = 0; i < N; i++)
      printf( "%4.1f*%4.1f + %4.1f = %4.1f\n", A, h_x[i], h_y[i], h_z[i] );


   free( h_x );
   free( h_y );
   free( h_z );

}
