#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NIter   = 10000;
   const int NThread = 10;
   omp_set_num_threads( NThread );

   int counter1=0, counter2=0;
   int result1, result2;

   for (int t=0; t<NIter; t++)
   {
#     pragma omp parallel
      {
//       with barrier
#        pragma omp critical
         counter1++;
#        pragma omp barrier
#        pragma omp master
         result1 = counter1;

//       without barrier
#        pragma omp critical
         counter2++;
#        pragma omp master
         result2 = counter2;
      }
   }

   printf( "results with/without barrier = %d/%d\n", result1, result2 );

   return EXIT_SUCCESS;
}
