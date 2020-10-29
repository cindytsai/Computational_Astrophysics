/*
Compile 
mpic++ -o a.out -fopenmp SOR_parallel_mpi.cpp

Run
(Size N of Square Lattice)
mpirun -np 2 --output-filename log ./a.out THREADS_NUM N omega

 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <cstring>
#include <mpi.h>

// Parallel
double getError(int, int, int, int, double*, double*);

// Normal
void doPrint(int, double*);
void getLo(int, double*);
void reference(int, double*);

int main( int argc, char *argv[] ){
	// Settings
	// N 	   -> Size of the grid we are calculating, 
	// 			  does not include the boundary
	// NThread -> # of openmp threads
	// omega   -> Overrelaxation parameter
	// NRank   -> Total rank of OpenMPI
	// MyRank  -> the rank this process is
	int N;
	int NThread;
	double omega;
	int NRank, MyRank;

	// If does not input the right argument or NULL, then set as default
	// Otherwise set as the input.
	if(argc != 4){
		NThread = 4;
		N = 16;
		omega = 1.5;
	}
	else{
		NThread = atoi(argv[1]);
		N = atoi(argv[2]);
		omega = atof(argv[3]);
	}

	omp_set_num_threads( NThread );

	// Initialize OpenMPI
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );
	MPI_Comm_size( MPI_COMM_WORLD, &NRank );

	if ( NRank != 2 ){
		printf("ERROR: NRank (%d) != 2\n", NRank );
		MPI_Abort( MPI_COMM_WORLD, 1 );
   	}

	/*
	SOR Parallelize
	 */
	// Find the index of even and odd chestbox
	int *ieven, *iodd;
	ieven = (int*)malloc(((N + 2) * (N + 2) / 2) * sizeof(int));
	iodd  = (int*)malloc(((N + 2) * (N + 2) / 2) * sizeof(int));
	
	// For even chestbox index
	for(int i = 0; i < ((N + 2) * (N + 2) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % (N + 2);
		iy = ((2 * i) / (N + 2)) % (N + 2);
		parity = (ix + iy) % 2;
		ix = ix + parity;
		ieven[i] = ix + iy * (N + 2);
	}

	// For odd chestbox index
	for(int i = 0; i < ((N + 2) * (N + 2) / 2); i = i+1){
		int parity, ix, iy;
		ix = (2 * i) % (N + 2);
		iy = ((2 * i) / (N + 2)) % (N + 2);
		parity = (ix + iy + 1) % 2;
		ix = ix + parity;
		iodd[i] = ix + iy * (N + 2);
	}

	// Initialize phi, phi_old, lo
	double *phi, *phi_old, *lo;
	const double dx = 1.0 / (double)(N+1);
	phi 	= (double*)malloc((N + 2) * (N + 2) * sizeof(double));
	phi_old = (double*)malloc((N + 2) * (N + 2) * sizeof(double));
	lo 		= (double*)malloc((N + 2) * (N + 2) * sizeof(double));
	getLo(N, lo);
	memset(phi, 0, (N + 2) * (N + 2) * sizeof(double));
	memset(phi_old, 0, (N + 2) * (N + 2) * sizeof(double));

	/*
	Manipulate the workload and settings for OpenMPI
	 */
	const int Tag = 123;
	int RootRank = 0;
	int TargetRank = (MyRank + 1) % 2;				// Send to or Receive from TargetRank
	int Work_e, Work_o;								// Divide the workload into NRank = 2
	int Count = N + 2;								// Numbers of cell to be passed
	int Interface = (((N + 2)/2) - 1) * (N+2);		// The starting (lowest) index of the interface
	double *SendBuf = &phi[Interface + MyRank * (N+2)];
	double *RecvBuf = &phi[Interface + TargetRank * (N+2)];

	if((N + 2) % 2 == 0){
		Work_e = ((N + 2) * (N + 2) / 2) / NRank;
		Work_o = ((N + 2) * (N + 2) / 2) / NRank;
	}
	else{
		if(((N + 2) / 2) % 2 == 0){
			Work_e = (((N + 2) / 2) / 2) * (N + 2);
			Work_o = (((N + 2) / 2) / 2) * (N + 2);
		}
		else{
			Work_e = (((N + 2) / 2) / 2) * (N + 2) + ((N + 2) / 2) + 1;
			Work_o = (((N + 2) / 2) / 2) * (N + 2) + ((N + 2) / 2);
		}
	}

	// Start of the main SOR Method with parallel
	const double target_err = 1.0e-10;
	double err;
	double tot_err = target_err + 1.0;
	int iter = 0;
	double start_time, end_time;

	int index, ix, iy;
	int l, r, t, b;

	if(MyRank == RootRank){
		printf("~START~\n");
		fflush(stdout);
		start_time = MPI_Wtime();
	}

	
	while(true){

		// Update even chestbox
		#	pragma omp parallel
		{
			// Update even chestbox
			# 	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = MyRank * Work_e; i < MyRank * Work_e + Work_e; i = i+1){

				// Center index
				index = ieven[i];
				ix = index % (N + 2);
				iy = index / (N + 2);

				// printf("index = %d\n", index);
				// fflush(stdout);

				// Do not update the boundary
				if((ix == 0) || (ix == (N+1)) || (iy == 0) || (iy == (N+1))){
					continue;
				}

				// Neighboring index
				l = (ix - 1) + iy * (N + 2);
				r = (ix + 1) + iy * (N + 2);
				t = ix + (iy + 1) * (N + 2);
				b = ix + (iy - 1) * (N + 2);

				// Update result to phi
				phi[index] = phi_old[index] + 0.25 * omega * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi_old[index] - pow(dx, 2) * lo[index]);
			}

			#	pragma omp barrier
		}

		// Exchange interface data

		MPI_Send(SendBuf, Count, MPI_DOUBLE, TargetRank, Tag, MPI_COMM_WORLD);

		MPI_Recv(RecvBuf, Count, MPI_DOUBLE, TargetRank, Tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// doPrint(N, phi);

		#	pragma omp parallel
		{
			// Update odd chestbox
			#	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = MyRank * Work_o; i < MyRank * Work_o + Work_o; i = i+1){
				// Center index
				index = iodd[i];
				ix = index % (N + 2);
				iy = index / (N + 2);

				// printf("index = %d\n", index);
				// fflush(stdout);

				// Do not update the boundary
				if((ix == 0) || (ix == (N+1)) || (iy == 0) || (iy == (N+1))){
					continue;
				}

				// Neighboring index
				l = (ix - 1) + iy * (N + 2);
				r = (ix + 1) + iy * (N + 2);
				t = ix + (iy + 1) * (N + 2);
				b = ix + (iy - 1) * (N + 2);

				// Update result to phi
				phi[index] = phi_old[index] + 0.25 * omega * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi_old[index] - pow(dx, 2) * lo[index]);
			}

			#	pragma omp barrier

		}	

		// Exchange interface data

		MPI_Send(SendBuf, Count, MPI_DOUBLE, TargetRank, Tag, MPI_COMM_WORLD);

		MPI_Recv(RecvBuf, Count, MPI_DOUBLE, TargetRank, Tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// doPrint(N, phi);

		// Compute the error
		iter = iter + 1;
		err = getError(Interface, MyRank, N, NThread, phi, lo);
		// printf("MyRank = %d, err = %.3lf\n", MyRank, err);
		MPI_Reduce(&err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, RootRank, MPI_COMM_WORLD);
		if(MyRank == RootRank){
			printf("iter:%4d, tot_err=%.5e\n", iter, tot_err);
			fflush(stdout);
		}
		
		if(MyRank == RootRank){
			MPI_Send(&tot_err, 1, MPI_DOUBLE, TargetRank, 0, MPI_COMM_WORLD);
		}
		else{
			MPI_Recv(&tot_err, 1, MPI_DOUBLE, TargetRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// printf("MyRank = %d, tot_err = %.5e\n", MyRank, tot_err);

		if(tot_err < target_err){
			break;
		}

		memcpy(phi_old, phi, sizeof(double) * (N + 2) * (N + 2));
	}

	// Received from the target rank to get the final solution
	if(MyRank == RootRank){
		MPI_Recv(RecvBuf, Interface, MPI_DOUBLE, TargetRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		end_time = MPI_Wtime();
	}
	else{
		MPI_Send(SendBuf, Interface, MPI_DOUBLE, TargetRank, 1, MPI_COMM_WORLD);
	}
	

	/*
	Find the analytic solution, only on Root Rank
	 */	
	
	if(MyRank == RootRank){
		double *z;
		z = (double*)malloc((N + 2) * (N + 2) * sizeof(double));
		reference(N, z);

		// printf("=====Analytic Solution=====\n");
		// doPrint(N, z);
	
		// printf("=====SOR Solution==========\n");
		// doPrint(N, phi);
		
		/*
		Compare with analytic solution
		 */
		double ana_err = 0.0;
		for(int i = 0; i < (N+2)*(N+2); i = i+1){
			ana_err = ana_err + fabs(phi[i] - z[i]);
		}
		ana_err = ana_err / (double)((N+2)*(N+2));
		printf("===Error Compare to Analytic Solution===\n");
		printf("     OpenMPI Rank = %d\n", NRank);
		printf("            Error = %.5e\n", ana_err);
		printf("    Grid size NxN = %d\n", N);
		printf("   Iteration used = %d\n", iter);
		printf("        Time used = %.10e sec\n", end_time - start_time);
		printf("Number of threads = %d\n", NThread);

		free(z);		

	}

	// Free all the array
	free(phi);
	free(phi_old);
	free(lo);
	free(ieven);
	free(iodd);

	// Terminate OpenMPI
	MPI_Finalize();

	return 0;
}

void doPrint(int N, double *array){
	/*
	Print the array with the normal 2D-coordinates (x, y),
	x from left to right increase, y from bottom to top increase

	array[(N+2)*(N+2)]
	 */
	int index;
	for(int j = N + 1; j >= 0; j = j-1){
		for(int i = 0; i < (N + 2); i = i+1){
			index = i + (N + 2) * j;
			printf("%.3lf ", array[index]);
			fflush(stdout);
		}
		printf("\n");
		fflush(stdout);
	}
}

double getError(int Interface, int MyRank, int N, int NThread, double *phi, double *lo){
	/*
	Calculate their own error in each MPI rank
	phi[(N+2)*(N+2)] array
	lo[(N+2)*(N+2)] array
	 */
	double residual;
	int t, b, l, r;	
	double error = 0.0;
	double dx = 1.0 / (double) (N + 1);
	int num = 0;

	Interface = Interface + (N + 2);

	// Parallelize
	omp_set_num_threads( NThread );
	#	pragma omp parallel private(residual, t, b, l, r)
	{
		# 	pragma omp for reduction( +:error )
		for(int index = MyRank * Interface; index < MyRank * Interface + Interface; index = index + 1){
			// ignore the boundary
			if(index % (N+2) == 0 || index % (N+2) == (N+1) || index / (N+2) == 0 || index / (N+2) == (N+1)){
				continue;
			}
			l = index - 1;
			r = index + 1;
			t = index + (N + 2);
			b = index - (N + 2);
			residual = pow(dx, -2) * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi[index]) - lo[index];
			error = error + fabs(residual);
			num = num + 1;
		}
	}
	error = error / (double)num;

	return error;
}

void getLo(int N, double *lo){
	/*
	lo[(N+2)*(N+2)] array -> laplace equation of lo to be solved
	 */
	double dx = 1.0 / (double) (N + 1);
	double x, y;
	int index;

	for(int iy = 0; iy < (N + 2); iy = iy+1){
		for(int ix = 0; ix < (N + 2); ix = ix+1){
			x = (double)ix * dx;
			y = (double)iy * dx;
			index = ix + (N + 2) * iy;
			lo[index] = 2.0 * x * (y - 1) * (y - 2.0 * x + x * y + 2.0) * exp(x - y);
		}
	}
}

void reference(int N, double *z){
	/*
	z[(N+2)*(N+2)] array -> Analytic solution on the grid
	 */
	double dx = 1.0 / (double) (N + 1);
	double x, y;
	int index;

	for(int iy = 0; iy < (N + 2); iy = iy+1){
		for(int ix = 0; ix < (N + 2); ix = ix+1){
			x = (double)ix * dx;
			y = (double)iy * dx;
			index = ix + (N + 2) * iy;
			z[index] = exp(x - y) * x * (1 - x) * y * (1 - y);
		}
	}
}