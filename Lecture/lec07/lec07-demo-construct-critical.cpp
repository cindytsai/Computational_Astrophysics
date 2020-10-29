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

   for (int t=0; t<NIter; t++)
   {
#     pragma omp parallel
      {
//       with critical
#        pragma omp critical
         counter1++;

//       without critical
#        pragma omp
         counter2++;
      }
   }

   printf( "results with/without critical = %d/%d\n", counter1, counter2 );

   return EXIT_SUCCESS;
}
