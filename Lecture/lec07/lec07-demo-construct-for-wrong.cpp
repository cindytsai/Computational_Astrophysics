#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// initialize array
   const int N = 10;
   int a[N];
   for (int i=0; i<N; i++)    a[i] = i;

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

#     pragma omp for
      for (int i=1; i<N; i++)
      {
//       accumulate the data
         a[i] = a[i] + a[i-1];
         printf( "a[%2d] = %2d computed by thread %d/%d\n", i, a[i], tid, nt );
      }
   }

   printf( "Sum of a[] = %d\n", a[N-1] );

   return EXIT_SUCCESS;
}
