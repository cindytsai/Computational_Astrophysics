#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main( int argc, char *argv[] )
{
// initialize MPI
   int NRank, MyRank;

   MPI_Init( &argc, &argv );

   MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );

   MPI_Comm_size( MPI_COMM_WORLD, &NRank );


// dt is set arbitrarily here
   float dt_MyRank = 0.1*(MyRank+1.0);


// find the minimum dt in the root
   const int Count    = 1;
   const int RootRank = 0;
   float dt_AllRank;

   MPI_Allreduce(&dt_MyRank, &dt_AllRank, Count, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

// print the sent and reduced data on each rank
   printf( "rank %d/%d: dt_MyRank %3.1f, dt_AllRank %3.1f\n", MyRank, NRank, dt_MyRank, dt_AllRank );


// terminate MPI
   MPI_Finalize();

   return EXIT_SUCCESS;
}
