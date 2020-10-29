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


// prepare the send buffer
   const int Count    = 5;
   const int RootRank = 0;

   int *Buf = new int [Count];

   if ( MyRank == RootRank ) {
      for (int t=0; t<Count; t++)   Buf[t] = t;
   }
   else {
      for (int t=0; t<Count; t++)   Buf[t] = 0;
   }


// print the data before transfer
   printf( "Before transfer: rank %d/%d:", MyRank, NRank );
   for (int t=0; t<Count; t++)   printf( " %2d", Buf[t] );
   printf( "\n" );


// broadcast from the root
   MPI_Bcast( Buf, Count, MPI_INT, RootRank, MPI_COMM_WORLD );


// print the data after transfer
   printf( " After transfer: rank %d/%d:", MyRank, NRank );
   for (int t=0; t<Count; t++)   printf( " %2d", Buf[t] );
   printf( "\n" );


// terminate MPI
   MPI_Finalize();

   delete [] Buf;

   return EXIT_SUCCESS;
}
