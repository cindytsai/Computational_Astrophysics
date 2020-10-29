#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// case 1: OMP_NUM_THREADS
// --> it will set the maximum number of threads available
#  pragma omp parallel
   {
      const int tid    = omp_get_thread_num();
      const int nt     = omp_get_num_threads();
      const int nt_max = omp_get_max_threads();

      if ( tid == 0 )
         printf( "Number of threads = %d/%d (via OMP_NUM_THREADS)\n", nt, nt_max );
#     pragma omp barrier

      printf( "Hello World from thread %d/%d\n", tid, nt );
   }


// case 2: omp_set_num_threads()
// --> it will overwrite the maximum number of threads available
   omp_set_num_threads( 8 );
#  pragma omp parallel
   {
      const int tid    = omp_get_thread_num();
      const int nt     = omp_get_num_threads();
      const int nt_max = omp_get_max_threads();

      if ( tid == 0 )
         printf( "Number of threads = %d/%d (via omp_set_num_threads)\n", nt, nt_max );
#     pragma omp barrier

      printf( "Hello World from thread %d/%d\n", tid, nt );
   }


// case 3: num_threads() clause
// --> it will NOT affect the maximum number of threads available
#  pragma omp parallel num_threads( 5 )
   {
      const int tid    = omp_get_thread_num();
      const int nt     = omp_get_num_threads();
      const int nt_max = omp_get_max_threads();

      if ( tid == 0 )
         printf( "Number of threads = %d/%d (via num_threads)\n", nt, nt_max );
#     pragma omp barrier

      printf( "Hello World from thread %d/%d\n", tid, nt );
   }

   return EXIT_SUCCESS;
}
