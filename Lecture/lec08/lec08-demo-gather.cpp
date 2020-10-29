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
   const int SendCount = 5;
   const int RecvCount = SendCount;
   const int RootRank  = NRank-1;

   int *SendBuf = new int [SendCount];
   int *RecvBuf = new int [RecvCount*NRank]; // only relevant for the root rank

   for (int t=0; t<SendCount; t++)  SendBuf[t] = MyRank*SendCount + t;

   if ( MyRank == RootRank ) {
      for (int t=0; t<RecvCount*NRank; t++)  RecvBuf[t] = 0;
   }


// gather data to the root
   MPI_Gather( SendBuf, SendCount, MPI_INT,
               RecvBuf, RecvCount, MPI_INT,
               RootRank, MPI_COMM_WORLD );


// print the send and recv buffers
   printf( "Send buffer on rank %d/%d:", MyRank, NRank );
   for (int t=0; t<SendCount; t++)   printf( " %2d", SendBuf[t] );
   printf( "\n" );

   if ( MyRank == RootRank ) {
      printf( "Recv buffer on rank %d/%d:", MyRank, NRank );
      for (int t=0; t<RecvCount*NRank; t++)  printf( " %2d", RecvBuf[t] );
      printf( "\n" );
   }


// terminate MPI
   MPI_Finalize();

   delete [] SendBuf;
   delete [] RecvBuf;

   return EXIT_SUCCESS;
}
