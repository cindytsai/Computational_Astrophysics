#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 4;
   omp_set_num_threads( NThread );

// initialize array
   const int N = 12;
   int a[N];
   for (int i=0; i<N; i++)    a[i] = i;

   int x = 0;

#  pragma omp parallel firstprivate(x)
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

//    compute a[]=a[]+1 in parallel
#     pragma omp for
      for (int i=0; i<N; i++)
      {
         a[i] = a[i] + 1;
         printf( "a[%2d] is computed by thread %d/%d\n", i, tid, nt );
      }

      printf("OUTSIDE\n");
#     pragma omp for
      for(int i = 0; i < N; i++){
         printf("thread %d/%d, i = %d\n", tid, nt, i);
         printf("x = %d\n", x);
         x = x + 1;
      }

      printf("DONE FOR LOOP\n");
   }

   return EXIT_SUCCESS;
}
