/*
Compile 
g++ -o a.out -fopenmp SOR_parallel.cpp

Run
(Size N of Square Lattice)
./a.out THREADS_NUM N

 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <cstring>

int *fw, *bw;

// Parallel
double getError(int, int, double*, double*);

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
	int N;
	int NThread;
	double omega;

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

	// Find forward and backward index
	fw = (int*)malloc((N + 2) * sizeof(int));
	bw = (int*)malloc((N + 2) * sizeof(int));
	
	for(int i = 0; i < (N + 2); i = i+1){
		fw[i] = (i + 1) % (N + 2);
		bw[i] = (i - 1 + (N + 2)) % (N + 2);
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

	// Start of the main SOR Method with parallel
	const double target_err = 1.0e-10;
	double err = target_err + 1.0;
	int iter = 0;
	double start_time, end_time;

	int index, ix, iy;
	int l, r, t, b;

	start_time = omp_get_wtime();

	while( err > target_err){

		// Update even chestbox
		#	pragma omp parallel
		{
			// Update even chestbox
			# 	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = 0; i < ((N + 2) * (N + 2) / 2); i = i+1){
				// Center index
				index = ieven[i];
				ix = index % (N + 2);
				iy = index / (N + 2);

				// Do not update the boundary
				if((ix == 0) || (ix == (N+1)) || (iy == 0) || (iy == (N+1))){
					continue;
				}

				// Neighboring index
				l = bw[ix] + iy * (N + 2);
				r = fw[ix] + iy * (N + 2);
				t = ix + fw[iy] * (N + 2);
				b = ix + bw[iy] * (N + 2);

				// Update result to phi
				phi[index] = phi_old[index] + 0.25 * omega * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi_old[index] - pow(dx, 2) * lo[index]);
			}

			// Update odd chestbox
			#	pragma omp for private(index, ix, iy, l, r, t, b)
			for(int i = 0; i < ((N + 2) * (N + 2) / 2); i = i+1){
				// Center index
				index = iodd[i];
				ix = index % (N + 2);
				iy = index / (N + 2);

				// Do not update the boundary
				if((ix == 0) || (ix == (N+1)) || (iy == 0) || (iy == (N+1))){
					continue;
				}

				// Neighboring index
				l = bw[ix] + iy * (N + 2);
				r = fw[ix] + iy * (N + 2);
				t = ix + fw[iy] * (N + 2);
				b = ix + bw[iy] * (N + 2);

				// Update result to phi
				phi[index] = phi_old[index] + 0.25 * omega * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi_old[index] - pow(dx, 2) * lo[index]);
			}
		}

		// Compute the error
		iter = iter + 1;
		err = getError(N, NThread, phi, lo);
		memcpy(phi_old, phi, sizeof(double) * (N + 2) * (N + 2));

		// printf("iter:%4d, err=%.5e\n", iter, err);
	}

	end_time = omp_get_wtime();

	/*
	Find the analytic solution
	 */	
	double *z;
	z = (double*)malloc((N + 2) * (N + 2) * sizeof(double));
	reference(N, z);

	printf("=====Analytic Solution=====\n");
	doPrint(N, z);

	printf("=====SOR Solution==========\n");
	doPrint(N, phi);

	/*
	Compare with analytic solution
	 */
	double ana_err = 0.0;
	for(int i = 0; i < (N+2)*(N+2); i = i+1){
		ana_err = ana_err + fabs(phi[i] - z[i]);
	}
	ana_err = ana_err / (double)((N+2)*(N+2));
	printf("=====Error Compare to Analytic Solution=====\n");
	printf("            Error = %.5e\n", ana_err);
	printf("    Grid size NxN = %d\n", N);
	printf("   Iteration used = %d\n", iter);
	printf("        Time used = %.10e sec\n", end_time - start_time);
	printf("Number of threads = %d\n", NThread);

	// Free all the array
	free(phi);
	free(phi_old);
	free(z);
	free(lo);
	free(fw);
	free(bw);
	free(ieven);
	free(iodd);

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
		}
		printf("\n");
	}
}

double getError(int N, int NThread, double *phi, double *lo){
	/*
	phi[(N+2)*(N+2)] array
	lo[(N+2)*(N+2)] array
	 */
	double residual;
	int index, t, b, l, r;	
	double error = 0.0;
	double dx = 1.0 / (double) (N + 1);

	// Parallelize
	omp_set_num_threads( NThread );
	#	pragma omp parallel private(residual, index, t, b, l, r)
	{
		# 	pragma omp for reduction( +:error )
		for(int j = 1; j <= N; j = j+1){
			for(int i = 1; i <= N; i = i+1){
				index = i + j * (N + 2);
				l = bw[i] + j * (N + 2);
				r = fw[i] + j * (N + 2);
				t = i + fw[j] * (N + 2);
				b = i + bw[j] * (N + 2);
				residual = pow(dx, -2) * (phi[l] + phi[r] + phi[t] + phi[b] - 4.0 * phi[index]) - lo[index];
				error = error + fabs(residual);
			}
		}
	}
	error = error / (double)(N * N);

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