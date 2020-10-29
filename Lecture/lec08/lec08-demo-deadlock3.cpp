#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main( int argc, char *argv[] )
{
   int NRank, MyRank;

   MPI_Init( &argc, &argv );

   MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );

   MPI_Comm_size( MPI_COMM_WORLD, &NRank );

// this test assumes only two ranks
   if ( NRank != 2 )
   {
      fprintf( stderr, "ERROR: NRank (%d) != 2\n", NRank );
      MPI_Abort( MPI_COMM_WORLD, 1 );
   }

// prepare data transfer
   const int Count      = 1;
   const int TargetRank = (MyRank+1)%2;   // (0,1) --> (1,0)
   const int Tag        = 123;            // arbitrary

   int SendBuf = (MyRank+1)*10;  // arbitrary
   int RecvBuf;

// both ranks receive first and then send using blocking asynchronous transfer
// --> always cause a deadlock
   MPI_Recv( &RecvBuf, Count, MPI_INT, TargetRank, Tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
   MPI_Send( &SendBuf, Count, MPI_INT, TargetRank, Tag, MPI_COMM_WORLD );

   printf( "Rank %d/%d: Send %d, Recv %d\n", MyRank, NRank, SendBuf, RecvBuf );

   MPI_Finalize();

   return EXIT_SUCCESS;
}
