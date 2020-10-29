#include <cstdio>
#include <cstdlib>
#include <omp.h>


void sum( const int N, const int N_min_for_parallel, int a[] )
{
   int  result   = 0;
   bool parallel = false;

#  pragma omp parallel if ( N >= N_min_for_parallel )
   {
#     pragma omp master
      parallel = omp_in_parallel();

#     pragma omp for reduction( +:result )
      for (int i=0; i<N; i++)
         result += a[i];
   }

   if ( parallel )
      printf( "N = %2d >= %2d --> summation is performed in parallel\n", N, N_min_for_parallel );
   else
      printf( "N = %2d <  %2d --> summation is performed serially\n", N, N_min_for_parallel );
}



int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 4;
   omp_set_num_threads( NThread );

// initialize array
   const int N1=5, N2=20, N_min_for_parallel=10;
   int a1[N1], a2[N2];
   for (int i=0; i<N1; i++)   a1[i] = i;
   for (int i=0; i<N2; i++)   a2[i] = i;

   sum( N1, N_min_for_parallel, a1 );
   sum( N2, N_min_for_parallel, a2 );

   return EXIT_SUCCESS;
}
