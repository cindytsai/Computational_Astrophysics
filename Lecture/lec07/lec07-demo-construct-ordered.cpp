#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 5;
   omp_set_num_threads( NThread );

// initialize array
   const int N = 10;
   int a[N];
   for (int i=0; i<N; i++)    a[i] = i;

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

//    with ordered
#     pragma omp single
      printf( "\n** with ordered **\n" );

#     pragma omp for ordered
      for (int i=0; i<N; i++)
      {
         a[i] = a[i] + 1;
#        pragma omp ordered
         printf( "a[%2d] is computed by thread %d/%d\n", i, tid, nt );
      }

//    without ordered
#     pragma omp single
      printf( "\n** without ordered **\n" );

#     pragma omp for
      for (int i=0; i<N; i++)
      {
         a[i] = a[i] + 1;
         printf( "a[%2d] is computed by thread %d/%d\n", i, tid, nt );
      }
   }

   return EXIT_SUCCESS;
}
