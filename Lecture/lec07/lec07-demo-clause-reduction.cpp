#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 4;
   omp_set_num_threads( NThread );

// initialize array
   const int N = 200;
   int a[N];
   for (int i=0; i<N; i++)    a[i] = i;

// example: sum
   int sum=0;  // must be initialized
#  pragma omp parallel for reduction( +:sum )
   for (int i=0; i<N; i++)
      sum += a[i];
   printf( "sum = %d\n", sum );

// example: max
   int maximum=-1;
#  pragma omp parallel for reduction( max:maximum )
   for (int i=0; i<N; i++)
      maximum = ( a[i] > maximum ) ? a[i] : maximum;
   printf( "maximum = %d\n", maximum );

   return EXIT_SUCCESS;
}
