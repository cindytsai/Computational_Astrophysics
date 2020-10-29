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


// initialize data
   const int Count = 10;
   int Data[Count];

   for (int t=0; t<Count; t++)   Data[t] = 100*MyRank + t;

   printf( " Input data on rank %d/%d:", MyRank, NRank );
   for (int t=0; t<Count; t++)   printf( "  %3d", Data[t] );
   printf( "\n" );


// prepare the send and recv buffers
   const int SendRank = 0;
   const int RecvRank = 1;
   const int Tag      = 123;

   if( MyRank == SendRank ){
      int SendBuf_Sum = 0;
      for(int t = 0; t < Count; t = t+1){
         SendBuf_Sum = SendBuf_Sum + Data[t];
      }

      MPI_Send(&SendBuf_Sum, 1, MPI_INT, RecvRank, Tag, MPI_COMM_WORLD);
   }

   if( MyRank == RecvRank ){
      int Sum = 0;
      for(int t = 0; t < Count; t = t+1){
         Sum = Sum + Data[t];
      }

      int RecvBuf;

      MPI_Recv(&RecvBuf, 1, MPI_INT, SendRank, Tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      Sum = Sum + RecvBuf;
      printf("Sum:%d\n", Sum);
   }

   MPI_Finalize();

   return EXIT_SUCCESS;
}
