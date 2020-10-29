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
   const int Count = 5;
   int Data[Count];

   for (int t=0; t<Count; t++)   Data[t] = 100*MyRank + t;

   printf( " Input data on rank %d/%d:", MyRank, NRank );
   for (int t=0; t<Count; t++)   printf( "  %3d", Data[t] );
   printf( "\n" );


// sum the local data
   int Sum = 0;
   for (int t=0; t<Count; t++)   Sum += Data[t];


// transfer data from SendRank to RecvRank 
   const int SendRank = 0;
   const int RecvRank = 1;
   const int Tag      = 123;
   int RecvBuf;
   MPI_Request Request;

   if ( MyRank == SendRank )  MPI_Isend( &Sum,     1, MPI_INT, RecvRank, Tag, MPI_COMM_WORLD, &Request );
   else                       MPI_Irecv( &RecvBuf, 1, MPI_INT, SendRank, Tag, MPI_COMM_WORLD, &Request );


// sum the received data
   if ( MyRank == RecvRank ) {
      MPI_Wait( &Request, MPI_STATUSES_IGNORE );
      Sum += RecvBuf;
      printf( "Summed data on rank %d/%d: %3d\n", MyRank, NRank, Sum );
   }


   MPI_Finalize();

   return EXIT_SUCCESS;
}
