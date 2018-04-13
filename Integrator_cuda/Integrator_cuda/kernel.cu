#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;

#define X 10000
#define Y 1000
#define BLOCKS 1
#define aS 0
#define aE 5
#define a 0
#define b 2

__constant__ int d_iterationsX = X;
__constant__ int d_iterationsAlpha = Y*BLOCKS;
__constant__ int d_blockSize = Y;
__constant__ int d_alphaStart=aS;
__constant__ int d_alphaEnd=aE;
__constant__ int d_a=a;
__constant__ int d_b=b;
__device__ double d_maxValues[Y*BLOCKS];
__device__ double d_alphaValues[Y*BLOCKS];
__global__ void integrate()
{
	__shared__ double d_distAlpha;
	__shared__ double d_distX;
	//TODO shared csak blokk szinten mûködik, több blokk esetén nem jó
	if (blockIdx.x==0 && threadIdx.x == 0)
	{
		d_distAlpha = (d_alphaEnd - d_alphaStart) / (double)d_iterationsAlpha;
		d_distX= (d_b - d_a) / (double)d_iterationsX;
	}
	__syncthreads();

	double d_y = d_alphaStart + (blockIdx.x*d_blockSize + ((threadIdx.x)+1.0))*(double)d_distAlpha;
	double d_sum=0;
	for (int i = 1; i < d_iterationsX; i++)
	{
		double d_x1 = d_a + (i - 1) * d_distX;
		double d_x2 = d_a + i * d_distX;

		double d_fa = (double)((2 + sin(10 * d_y)) * pow(d_x1, d_y) * sin(d_y / (2 - d_x1)));
		double d_fb = (double)((2 + sin(10 * d_y)) * pow(d_x2, d_y) * sin(d_y / (2 - d_x2)));

		//trapéz módszer
		d_sum = d_sum + ((d_x2 - d_x1) * (d_fa + d_fb)) / 2.0;
	}

	d_alphaValues[threadIdx.x] = d_y;
	d_maxValues[threadIdx.x] = d_sum;
}


int main()
{
	/*int iterations;
	std::cout << "Iterations ?\n";
	std::cin >> iterations;
	*/

	double maxValues[Y*BLOCKS];
	double alphaValues[Y*BLOCKS];
	for (int i = 0; i < Y*BLOCKS; i++)
	{
		maxValues[i] = 0;
		alphaValues[i] = 0;
	}

	double alphavalue, maxvalue;

	//int *p_alphaStart=&alphaStart;
	//int *p_alphaEnd=&alphaEnd;
	//int *p_a=&a;
	//int *p_b=&b;

	//cudaMalloc((void**)&d_alphaStart, sizeof(int));
	//cudaMalloc((void**)&d_alphaEnd, sizeof(int));
	//cudaMalloc((void**)&d_a, sizeof(int));
	//cudaMalloc((void**)&d_b, sizeof(int));

	//cudaMemset(&d_alphaStart, 0, sizeof(int));
	//cudaMemset(&d_alphaEnd, 5, sizeof(int));
	//cudaMemset(&d_a, 0, sizeof(int));
	//cudaMemset(&d_b, 2, sizeof(int));

	//cudaMemcpy(d_alphaStart,p_alphaStart, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_alphaEnd, p_alphaEnd, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_a,p_a, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b,p_b, sizeof(int), cudaMemcpyHostToDevice);

	//int iterations = N;
	//int alphaEnd = aE;
	//int alphaStart = aS;
	//int aa = a;
	//int bb = b;
	//double distAlpha = (alphaEnd - alphaStart) / (double)iterations;
	//double distX = (bb - aa) / (double)iterations;
	//double *distA

	//cudaMalloc((void**)&d_distAlpha, sizeof(double));
	//cudaMalloc((void**)&d_distX, sizeof(double));
	//cudaMemset(&d_alphaStart, 0, sizeof(int));
	//cudaMemset(&d_alphaEnd, 5, sizeof(int));
	//cudaMemcpy(&d_distAlpha, &distAlpha, sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(&d_distX, &distX, sizeof(double), cudaMemcpyHostToDevice);

	integrate << <BLOCKS, Y >> > ();

	cudaMemcpyFromSymbol(&alphaValues, d_alphaValues, Y*BLOCKS * sizeof(double));
	cudaMemcpyFromSymbol(&maxValues, d_maxValues, Y*BLOCKS * sizeof(double));

	//TODO parallel reduction

	for (size_t i = 0; i < Y*BLOCKS; i++)
	{
		std::cout << maxValues[i]<<"\n";
	}
	int index = std::distance(maxValues,std::max_element(maxValues, maxValues + Y));
	double max = maxValues[index];
	double alpha = alphaValues[index];
	std::cout << max << " - max value\n";
	std::cout << alpha << " - at alpha\n";

	printf("End\n");
	std::cin >> "";

	return 0;
}