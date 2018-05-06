#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <thrust\device_vector.h>
#include <thrust\extrema.h>
#include <chrono>
using namespace std;

#define iterX 10000
#define Y 512
#define BLOCKS 24
#define aS 0.0
#define aE 5.0
#define a 0.0
#define b 2.0

__constant__ double d_distAlpha=(aE-aS)/(Y*BLOCKS);
__constant__ double d_distX=(b-a)/iterX;

__global__ void integrate(double *d_maxValues)
{
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
	d_maxValues[currentThreadId] = d_sum;
}


int main()
{
	double *d_maxValues;
	double max;
	double alpha;

	cudaMalloc((void**)&d_maxValues,BLOCKS*Y*sizeof(double));

	auto start = std::chrono::high_resolution_clock::now();

	integrate << <BLOCKS, Y >> > (d_maxValues);

	thrust::device_ptr<double> td_maxValues(d_maxValues);
	int end = BLOCKS*Y;
	thrust::device_ptr<double> td_max = thrust::max_element(td_maxValues, td_maxValues + end);

	auto finish = std::chrono::high_resolution_clock::now();

	max = *td_max;
	int diff = td_max - &td_maxValues[0];
	alpha = ((aE - aS) / (Y*BLOCKS))*diff;

	std::chrono::duration<double> elapsed = finish - start;

	std::cout.precision(15);
	std::cout << max << " - max value\n";
	std::cout.precision(15);
	std::cout << alpha << " - at alpha\n";
	std::cout << "Elapsed time: " << elapsed.count();

	cudaFree(d_maxValues);

	printf("End\n");
	std::cin >> "";

	return 0;
}