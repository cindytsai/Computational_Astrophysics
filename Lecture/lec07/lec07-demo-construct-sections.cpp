#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 2;
   omp_set_num_threads( NThread );

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

#     pragma omp sections
      {
#        pragma omp section
         {
            printf( "section 0 is computed by thread %d/%d\n", tid, nt );
         }

#        pragma omp section
         {
            printf( "section 1 is computed by thread %d/%d\n", tid, nt );
         }

#        pragma omp section
         {
            printf( "section 2 is computed by thread %d/%d\n", tid, nt );
         }

#        pragma omp section
         {
            printf( "section 3 is computed by thread %d/%d\n", tid, nt );
         }

#        pragma omp section
         {
            printf( "section 4 is computed by thread %d/%d\n", tid, nt );
         }

#        pragma omp section
         {
            printf( "section 5 is computed by thread %d/%d\n", tid, nt );
         }
      }
   }

   return EXIT_SUCCESS;
}
