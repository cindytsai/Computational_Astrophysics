#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 2;
   omp_set_num_threads( NThread );

// initialize array
   const int N = 10;
   int a[N], b[N];
   for (int i=0; i<N; i++)
   {
      a[i] = 1*i;
      b[i] = 2*i;
   }

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

//    compute a[]=a[]+1 in parallel
//    --> use the nowait clause to remove the implied barrier at the end of the loop construct
#     pragma omp for nowait
      for (int i=0; i<N; i++)
      {
         a[i] = a[i] + 1;
         printf( "a[%2d] is computed by thread %d/%d\n", i, tid, nt );
      }

//    compute b[]=b[]+1 in parallel
#     pragma omp for nowait
      for (int i=0; i<N; i++)
      {
         b[i] = b[i] + 1;
         printf( "b[%2d] is computed by thread %d/%d\n", i, tid, nt );
      }
   }

   return EXIT_SUCCESS;
}
