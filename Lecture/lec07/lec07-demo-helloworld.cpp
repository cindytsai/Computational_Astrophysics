#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
   printf( "Hello World (master-only)\n" );

#  pragma omp parallel
   {
      printf( "Hello World (multithreading)\n" );
   }

   return EXIT_SUCCESS;
}
