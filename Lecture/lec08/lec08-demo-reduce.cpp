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


// prepare data transfer
   const int Count    = 1;
   const int RootRank = 0;

   int *SendBuf = new int [Count];
   int *RecvBuf = new int [Count];  // only relevant for the root rank

   for (int t=0; t<Count; t++)   SendBuf[t] = MyRank*Count + t;


// store the reduced data in the root
   MPI_Reduce( SendBuf, RecvBuf, Count, MPI_INT, MPI_SUM, RootRank, MPI_COMM_WORLD );

// print the sent and reduced data
   printf( "Send buffer: rank %d/%d:", MyRank, NRank );
   for (int t=0; t<Count; t++)   printf( "  %3d", SendBuf[t] );
   printf( "\n" );

   if ( MyRank == RootRank ) {
      printf( "Recv buffer: rank %d/%d:", MyRank, NRank );
      for (int t=0; t<Count; t++)   printf( "  %3d", RecvBuf[t] );
      printf( "\n" );
   }


// terminate MPI
   MPI_Finalize();

   delete [] SendBuf;
   delete [] RecvBuf;

   return EXIT_SUCCESS;
}
