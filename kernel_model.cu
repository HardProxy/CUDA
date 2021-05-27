
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N XYXY //Tamanho da Malha
#define dom 50.0
#define IT 300 //Loops temporais

// Sine-Gordon without borders Kernels
__global__ void sineGordon_Kernel(double *m_s,double *m_act, double *m_previous,double dts, double dxts) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	if( (i > 0 && i < N-1) && (j > 0 && j < N-1 ))
	m_act[i*blockDim.x + j] = -m_previous[i*blockDim.x +j] + 2.0 * (1.0 - 2.0 * dxts) * m_s[i*blockDim.x + j] + dxts * (m_s[(i + 1)*blockDim.x+j] + m_s[(i - 1)*blockDim.x+j] + m_s[i*blockDim.x+j + 1] + m_s[i*blockDim.x+j - 1]) - dts * (sin((m_s[(i + 1)*blockDim.x+j] + m_s[(i - 1)*blockDim.x+j] + m_s[i*blockDim.x+j + 1] + m_s[i*blockDim.x+j - 1]) / 4.0));
}

__global__ void actualization_Kernel(double* m_s, double* m_act, double* m_previous) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	if ((i > 0 && i < N-1) && (j > 0 && j < N-1)) {
		m_previous[i * blockDim.x + j] = m_s[i * blockDim.x + j];
		m_s[i * blockDim.x + j] = m_act[i * blockDim.x + j];
	}
}

void showMatrix(double *m) {
	int i, j;

	for ( i = 0; i < N; i++)
	{
		for ( j = 0; j < N; j++)
		{
			printf("%lf\t",m[i*N + j]);
		}
		printf("\n");
	}
}

void inicMatrix(double *m) {
	int i, j;

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			m[i * N + j] = 0;
		}
	}
}

void initialCond(double* m, double dx, double dy) {
	int i, j;
	double x, y;
	y = -dom/2.0;
	for (j = 0; j < N; j++) {
		x = -dom/2.0;
		for (i = 0; i < N; i++) {
			m[i*N+j] = 4 * atan(exp(3 - sqrt(x * x + y * y)));
			x = x + dx;
		}
		y = y + dy;
	}

}

void deviceCapabilities() {

	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execition timeout : ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf(" --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n",
			prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n",
			prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2]);
		printf("\n");
	}
}

int  main() {
	double *hm_s, *hm_act,*hm_previous, dx, dy, dt, *dxts, *dts; //Host variables
	double *dm_s, *dm_act, *dm_previous, *d_dxts, *d_dts; // Device Variables
	int i, j, k;

	dts = (double*)malloc(sizeof(double));
	dxts = (double*)malloc(sizeof(double));

	//Definição de Parâmetros
	printf("Definindo parametros para a discretizao ... \n");
	dx = dom / N;
	dy = dom / N;
	dt = dx / sqrt(2.0);
	*dts = dt * dt;
	*dxts = (dt / dx) * (dt / dx);

	//deviceCapabilities();

	//printf("Alocando memória no HOST ... \n");
	hm_s = (double*)malloc((N * N) * sizeof(double)); // Matrix Solution on HOST
	hm_act = (double*)malloc((N * N) * sizeof(double)); // Matrix Actualizations on HOST 
	hm_previous = (double*)malloc((N * N) * sizeof(double)); //Previous Matrix results on HOST

	printf("Inicializando matrizes no HOST ... \n");
	inicMatrix(hm_s); // Zeros  Matrix
	inicMatrix(hm_act); // Zeros  Matrix
	inicMatrix(hm_previous); // Zeros  Matrix 

	printf("Aplicando as condicoes iniciais a matriz ... \n");
	initialCond(hm_s, dx, dy); // Appling initial conditions


	printf("Alocando memoria no DEVICE ... \n");
	cudaMalloc(&dm_s, (N * N) * sizeof(double)); // Matrix Solution on Device
	cudaMalloc(&dm_act, (N * N) * sizeof(double)); // Matrix Actualizations on Device
	cudaMalloc(&dm_previous, (N * N) * sizeof(double)); // Matrix Actualizations on Device
	cudaMalloc(&d_dts, sizeof(double)); // dts on Device
	cudaMalloc(&d_dxts, sizeof(double)); // dxts on Device

	//deviceCapabilities();

	cudaMemcpy(d_dts, dts, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dxts, dxts, sizeof(double), cudaMemcpyHostToDevice);
	
	clock_t begin = clock();
	printf("Temporal Evolution ... \n");
	for ( i = 0; i < IT; i++)
	{
		//printf("Transferindo informacoes do HOST para o DEVICE ... \n");
		cudaMemcpy(dm_s, hm_s, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dm_act, hm_act, (N * N) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dm_previous, hm_previous, (N * N) * sizeof(double), cudaMemcpyHostToDevice);

		// Parellelism in N blocks and N threads per block for the inner elements 
		sineGordon_Kernel <<< N, N >>> (dm_s, dm_act, dm_previous, *dts, *dxts); // Parallelism in N Blocks with N Threads
		//actualization_Kernel << < N, N >> > (dm_s, dm_act, dm_previous);

		//printf("Transferindo atualizacoes do DEVICE para o HOST ... \n");
		cudaMemcpy(hm_act, dm_act, (N * N) * sizeof(double), cudaMemcpyDeviceToHost);
		
		for (k = 1; k < N - 1; k++) {
			for ( j = 1; j < N - 1 ; j++)
			{
				hm_previous[k * N + j] = hm_s[k * N + j];
				hm_s[k * N + j] = hm_act[k * N + j];
			}
		}
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	FILE *pf;
	pf = fopen("tempo.dat","a");
	fprintf(pf,"%lf\n",time_spent);
	
	printf("Liberando memoria no HOST e no DEVICE ... \n");
	// Unallocing CPU variables
	free(hm_s);
	free(hm_act);
	free(hm_previous);
	free(dxts);
	free(dts);
	//Unallocing GPU variables
	cudaFree(dm_s); 
	cudaFree(dm_act);
	cudaFree(dm_previous);
	cudaFree(d_dxts);
	cudaFree(d_dts);
	printf("Tempo de Execucao : %lf\n", time_spent);
	return 0;	
}
