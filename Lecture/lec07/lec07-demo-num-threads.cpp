#include <cstdio>
#include <cstdlib>
#include <omp.h>

void funcTest(){
#  pragma omp parallel
   {
      printf( "inside void funcTest()\n" );
   }   
}

int main( int argc, char *argv[] )
{
#  pragma omp parallel
   {
      printf( "Environment variable OMP_NUM_THREADS, max = %d\n", omp_get_max_threads() );
   }

   omp_set_num_threads( 3 );
   printf("See if will the function omp set Nthread=3\n");
   funcTest();

   omp_set_num_threads( 5 );
#  pragma omp parallel
   {
      printf( "Runtime library routine omp_set_num_threads()\n" );
   }

#  pragma omp parallel num_threads( 4 )
   {
      printf( "Clause num_threads()\n" );
   }

#  pragma omp parallel
   {
      printf( "Runtime library routine omp_set_num_threads() again\n" );
   }

   return EXIT_SUCCESS;
}
