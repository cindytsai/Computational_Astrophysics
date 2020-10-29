#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <assert.h>

double cpu_time( void )
{
   clock_t current_clock = clock();
   double time = current_clock / (double)CLOCKS_PER_SEC;
   printf( "clock = %lu, time = %lf\n", current_clock, time );
   return time;
}

int main( int argc, char *argv[] )
{
   assert( argc > 1 );
   const long N = atol( argv[1] );

   double time1_omp, time2_omp, time1_clock, time2_clock;
   double *a = new double [N];
   for (long i=0; i<N; i++)    a[i] = i;

// get the starting time
   time1_clock = cpu_time();
#  ifdef _OPENMP 
   time1_omp   = omp_get_wtime();
#  endif

// perform some arbitrary operations
   omp_set_num_threads(atoi(argv[2]));
#  pragma omp parallel for
   for (long i=0; i<N; i++)   a[i] = pow( a[i], 2.0 );
// get the ending time
   time2_clock = cpu_time();
#  ifdef _OPENMP
   time2_omp   = omp_get_wtime();
#  endif
   
   printf( "Elapsed time by clock():         %lf sec\n", time2_clock-time1_clock );
   printf( "Elapsed time by omp_get_wtime(): %lf sec\n", time2_omp-time1_omp );
   
   delete [] a;
   
   return EXIT_SUCCESS;
}
