#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 4;
   omp_set_num_threads( NThread );

// initialize array
   const int Ni = 4;
   const int Nj = 2;
   int a[Nj][Ni];

   for (int j=0; j<Nj; j++)
   for (int i=0; i<Ni; i++)
      a[j][i] = j*Ni+i;

#  pragma omp parallel
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

//    without the collapse clause
#     pragma omp single
      printf( "\n** without collapse **\n" );

#     pragma omp for
      for (int j=0; j<Nj; j++)
      for (int i=0; i<Ni; i++)
      {
         a[j][i] = a[j][i] + 1;
         printf( "a[%2d][%2d] is computed by thread %d/%d\n", j, i, tid, nt );
      }

//    with the collapse clause
#     pragma omp single
      printf( "\n** with collapse **\n" );

#     pragma omp for collapse(2)
      for (int j=0; j<Nj; j++)
      for (int i=0; i<Ni; i++)
      {
         a[j][i] = a[j][i] + 1;
         printf( "a[%2d][%2d] is computed by thread %d/%d\n", j, i, tid, nt );
      }
   }

   return EXIT_SUCCESS;
}
