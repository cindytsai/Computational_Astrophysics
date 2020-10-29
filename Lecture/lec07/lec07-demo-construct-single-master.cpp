#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
   const int NThread = 4;
   omp_set_num_threads( NThread );

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      printf( "** Outside ** the single construct from thread %d/%d\n", tid, nt );

#     pragma omp single
      printf( "** Inside  ** the single construct from thread %d/%d\n", tid, nt );

#     pragma omp master
      printf( "** Inside  ** the master construct from thread %d/%d\n", tid, nt );
   }

   return EXIT_SUCCESS;
}
