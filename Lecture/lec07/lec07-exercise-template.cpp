#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main( int argc, char *argv[] )
{
   	const int NThread = 4;
   	omp_set_num_threads( NThread );

	// initialize matrices
	const int N = 4;
	float A[N][N], B[N][N], C[N][N];
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++) {
			A[i][j] = i;
			B[i][j] = j;
		}		
	}


	#pragma omp parallel
	{
   		const int tid = omp_get_thread_num();
   		const int nt  = omp_get_num_threads();

		//    compute A = B dot C
   		# pragma omp for
		for (int i=0; i<N; i++){
			for (int j=0; j<N; j++){
				C[i][j] = 0.0;

				for (int t=0; t<N; t++){
					C[i][j] += A[i][t]*B[t][j];
				}

      			printf( "C[%2d][%2d] is computed by thread %d/%d\n", i, j, tid, nt );
			}
		}
		
		//    print the results
		# pragma omp master
		{
			printf( "\nmatrix A:\n" );
			for (int i=0; i<N; i++) {
				for (int j=0; j<N; j++) {
					printf( "  %5.0f", A[i][j] );
				}
				printf( "\n" );
			}

			printf( "\nmatrix B:\n" );
			for (int i=0; i<N; i++) {
				for (int j=0; j<N; j++) {
					printf( "  %5.0f", B[i][j] );
				}
				printf( "\n" );
			}

			printf( "\nmatrix C = A dot B:\n" );
			for (int i=0; i<N; i++) {
				for (int j=0; j<N; j++) {
					printf( "  %5.0f", C[i][j] );
				}
				printf( "\n" );
			}
		}
	}

	return EXIT_SUCCESS;
}
