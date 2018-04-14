#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;

#define iterX 10000
#define Y 1000
#define BLOCKS 10
#define aS 0.0
#define aE 5.0
#define a 0.0
#define b 2.0

__constant__ double d_distAlpha=(aE-aS)/(Y*BLOCKS);
__constant__ double d_distX=(b-a)/iterX;

__device__ double d_maxValues[Y*BLOCKS];
__device__ double d_alphaValues[Y*BLOCKS];

__global__ void integrate()
{
	//if (blockIdx.x==0 && threadIdx.x == 0)
	//{
	//	d_distAlpha = (d_alphaEnd - d_alphaStart) / (double)d_iterationsAlpha;
	//	d_distX= (d_b - d_a) / (double)d_iterationsX;
	//}
	//__syncthreads();

	int currentThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	double d_y = aS + (currentThreadId+1.0)*(double)d_distAlpha;
	double d_sum=0;
	for (int i = 1; i < iterX; i++)
	{
		double d_x1 = a + (i - 1) * d_distX;
		double d_x2 = a + i * d_distX;

		double d_fa = (double)((2 + sin(10 * d_y)) * pow(d_x1, d_y) * sin(d_y / (2 - d_x1)));
		double d_fb = (double)((2 + sin(10 * d_y)) * pow(d_x2, d_y) * sin(d_y / (2 - d_x2)));

		d_sum = d_sum + ((d_x2 - d_x1) * (d_fa + d_fb)) / 2.0;
	}

	d_alphaValues[currentThreadId] = d_y;
	d_maxValues[currentThreadId] = d_sum;
}


int main()
{
	double maxValues[Y*BLOCKS];
	double alphaValues[Y*BLOCKS];
	for (int i = 0; i < Y*BLOCKS; i++)
	{
		maxValues[i] = 0;
		alphaValues[i] = 0;
	}

	double alphavalue, maxvalue;

	integrate << <BLOCKS, Y>> >();

	cudaMemcpyFromSymbol(&alphaValues, d_alphaValues, Y*BLOCKS * sizeof(double));
	cudaMemcpyFromSymbol(&maxValues, d_maxValues, Y*BLOCKS * sizeof(double));

	//TODO parallel reduction

	//for (size_t i = 0; i < Y*BLOCKS; i++)
	//{
	//	std::cout << maxValues[i]<<" - "<<i<<"\n";
	//}
	int index = std::distance(maxValues,std::max_element(maxValues, maxValues + Y*BLOCKS));
	double max = maxValues[index];
	double alpha = alphaValues[index];
	std::cout << max << " - max value\n";
	std::cout << alpha << " - at alpha\n";

	printf("End\n");
	std::cin >> "";

	return 0;
}