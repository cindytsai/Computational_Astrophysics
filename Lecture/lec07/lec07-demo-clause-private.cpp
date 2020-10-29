#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
// set the number of threads
   const int NThread = 2;
   omp_set_num_threads( NThread );

// initialization
   int  var;
   int *ptr = &var;

// example: private
   var = -1;
   printf( "\n** private **\n" );
#  pragma omp parallel private( var )
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      printf( "Before assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );

      var = tid+50;

      printf( " After assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );
   }

// example: private without default
//          so that you have to assign which variable is private and which is shared manually.
   var = -1;
   printf( "\n** private without default **\n" );
#  pragma omp parallel default( none ) private( var ) shared( ptr )
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      printf( "Before assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );

      var = tid+50;

      printf( " After assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );
   }

// example: firstprivate
//          so that var use the value declared outside the openmp threads
//          and set var as private as well, by using copy constructor.
//          
//          There is also "lastprivate" as well, which assign the value back to 
//          main process' variable.
   var = -1;
   printf( "\n** firstprivate **\n" );
#  pragma omp parallel firstprivate( var )
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      printf( "Before assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );

      var = tid+50;

      printf( " After assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );
   }

// example: shared
//          shared with other threads.
   var = -1;
   printf( "\n** shared (results are indeterministic) **\n" );
#  pragma omp parallel shared( var )
   {
      const int tid = omp_get_thread_num();
      const int nt  = omp_get_num_threads();

      printf( "Before assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );

      var = tid+50;

      printf( " After assignment: thread %d/%d: original %d, this thread %d\n", tid, nt, *ptr, var );
   }

   return EXIT_SUCCESS;
}
