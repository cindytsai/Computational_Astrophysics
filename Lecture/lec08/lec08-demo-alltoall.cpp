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


// prepare tranfer buffer
   const int SendCount = 2;
   const int RecvCount = SendCount;

   int *SendBuf = new int [SendCount*NRank]; // relevant for all ranks
   int *RecvBuf = new int [RecvCount*NRank]; // relevant for all ranks

   for (int t=0; t<SendCount*NRank; t++)  SendBuf[t] = 100*MyRank + t;


// all ranks send data to all ranks
   MPI_Alltoall( SendBuf, SendCount, MPI_INT,
                 RecvBuf, RecvCount, MPI_INT,
                 MPI_COMM_WORLD );


// print the sent and received data
   printf( "Send buffer: rank %d/%d:", MyRank, NRank );
   for (int t=0; t<SendCount*NRank; t++)  printf( "  %3d", SendBuf[t] );
   printf( "\n" );

   printf( "Recv buffer: rank %d/%d:", MyRank, NRank );
   for (int t=0; t<RecvCount*NRank; t++)  printf( "  %3d", RecvBuf[t] );
   printf( "\n" );


// terminate MPI
   MPI_Finalize();

   delete [] SendBuf;
   delete [] RecvBuf;

   return EXIT_SUCCESS;
}
