#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{

// set the number of threads
   const int NThread = 4;
   omp_set_num_threads( NThread );

// initialize array
   const int N = 8;
   int a[N];
   for (int i=0; i<N; i++)    a[i] = i;

// example of the combined constructs "parallel for"
#  pragma omp parallel for
   for (int i=0; i<N; i++)
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      a[i] = a[i] + 1;
      printf( "a[%2d] is computed by thread %d/%d\n", i, tid, nt );
   }

// example of the combined constructs "parallel sections"
#  pragma omp parallel sections
   {
#     pragma omp section
      {
         const int tid = omp_get_thread_num();
         const int nt  = omp_get_num_threads();
         printf( "section 0 is computed by thread %d/%d\n", tid, nt );
      }

#     pragma omp section
      {
         const int tid = omp_get_thread_num();
         const int nt  = omp_get_num_threads();
         printf( "section 1 is computed by thread %d/%d\n", tid, nt );
      }
   }

   return EXIT_SUCCESS;

}
