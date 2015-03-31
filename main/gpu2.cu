typedef float Qfloat;
typedef signed char schar;
#include <stdio.h>

#include <helper_cuda.h>
#include "svm_function.h"
#include <stdlib.h>

__device__
#ifdef FLOAT1g
float
#else
double
#endif
#ifdef FLOAT1g
 dot(float*px, float*py, int dim) {
	float sum = 0;
#else
	dot(double*px, double*py, int dim) {
		double sum = 0;
#endif
	for (int i = 0; i < dim; i++)
		sum += px[i] * py[i];
	return sum;
}

__shared__
#ifdef FLOAT1g
float
#else
double
#endif
 x_square[512];
#ifdef FLOAT1g
__global__ void kernel(float* data, int l, int dim, float gamma,
		signed char* y, float* x, float* QD) {
#else
__global__ void kernel(float* data, int l, int dim, double gamma,
			signed char* y, double* x, double* QD) {
#endif

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
// you should include check for elements




		unsigned int idx_dim= idx * dim;
		if (threadIdx.y==0)
			for (int i=0;i<gridDim.x;i++) {
					int id=blockDim.x*i+threadIdx.x;
					int id_dim=id*dim;
					x_square[id] = dot(x + id_dim, x + id_dim, dim);
				}
				__syncthreads();
			//exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]))); -kernel_function
				if (idx < l) {
				if (threadIdx.y==0)	QD[idx] = exp(
				-gamma
						* (x_square[idx] + x_square[idx]
								- 2 * dot(x + idx_dim, x + idx_dim, dim)));

		//for (int j = 0; j < l; j++)
		const int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (j<l)
			data[idx * l + j] =
					(Qfloat) (y[idx] * y[j]
							* exp(
									-gamma
											* (x_square[idx] + x_square[j]
													- 2
															* dot(x + idx_dim,
																	x
																			+ j
																					* dim,
																	dim))));

	}

}

#ifdef FLOAT1g
__global__ void kernel2(float* data, int l, int dim, float gamma,
		signed char* y, float* x, float* QD) {
#else
__global__ void kernel2(float* data, int l, int dim, double gamma,
			signed char* y, double* x, double* QD) {
#endif
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
// you should include check for elements
	//gridDim.x

	for (int i=0;i<gridDim.x;i++) {
		int id=blockDim.x*i+threadIdx.x;

		x_square[id] = dot(x + id * dim, x + id * dim, dim);
	}
	__syncthreads();


	if (idx < l) {
	//	const int idy = blockIdx.y * blockDim.y + threadIdx.y;

		unsigned int idx_dim= idx * dim;
		//exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]))); -kernel_function



		QD[idx] = exp(
				-gamma
						* (x_square[idx] + x_square[idx]
								- 2 * dot(x + idx_dim, x + idx_dim, dim)));

		for (int j = 0; j < l; j++)
	//	const int j = idy;
	//	if (j<l)
			data[idx * l + j] =
					(Qfloat) (y[idx] * y[j]
							* exp(
									-gamma
											* (x_square[idx] + x_square[j]
													- 2
															* dot(x + idx_dim,
																	x
																			+ j
																					* dim,
																	dim))));

	}

}

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/*if you want to separate kernel*/ //#include <kernel.cu>
//uncomment if you want to make function external


signed char *y_gpu;
	float* data_gpu;

#ifdef FLOAT1g
float
#else
double
#endif
*QD_gpu;

#ifdef FLOAT1g
float
#else
double
#endif
* x_gpu;

int max_threads;
int d_blocks;
	void initGPU(int l,int dim) {
		cudaDeviceProp deviceProp;
		        cudaGetDeviceProperties(&deviceProp, 0);
		        max_threads=deviceProp.maxThreadsPerBlock;
		        d_blocks=deviceProp.multiProcessorCount;

		cudaMalloc(&y_gpu, l * sizeof(signed char));
			cudaMalloc(&data_gpu, l * l * sizeof(float));
			cudaMalloc(&x_gpu, l * dim * sizeof(double));
			cudaMalloc(&QD_gpu, l * sizeof(double));
			 	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	}

	void freeGPU() {
		checkCudaErrors(cudaFree(y_gpu));
			checkCudaErrors(cudaFree(data_gpu));
			checkCudaErrors(cudaFree(x_gpu));
			checkCudaErrors(cudaFree(QD_gpu));

			checkCudaErrors(cudaDeviceReset());
	}
#ifdef FLOAT1g
	void RunGPUKERNEL(const svm_problem& prob, schar *y, float gamma, float* data,
			float*QD) {
#else
	void RunGPUKERNEL(const svm_problem& prob, schar *y, double gamma, float* data,
			double*QD) {
#endif
	int l = prob.l;

	int dim = prob.x[0].dim;
	int block;
	int thread;

if (l<max_threads) {
	block=d_blocks;
	thread=l/d_blocks+1;
} else {
	 block = l/max_threads +1;
		 thread =max_threads;
}
int max_2d_t = sqrt(max_threads);
		dim3 threads(max_2d_t, max_2d_t);
		dim3 blocks(l/max_2d_t+1,l/max_2d_t+1);
	//	printf("%s Starting...\n\n", sSDKsample);





	checkCudaErrors(
			cudaMemcpyAsync(y_gpu, y, l * sizeof(schar), cudaMemcpyHostToDevice,
					0));

#ifdef FLOAT1g
float
#else
double
#endif
* x_cpu = new
#ifdef FLOAT1g
float
#else
double
#endif
[l * dim];
	for (int i = 0; i < l; i++)
		for (int j = 0; j < dim; j++)
			x_cpu[i * dim + j] = prob.x[i].values[j];

	checkCudaErrors(
			cudaMemcpy(x_gpu, x_cpu, l * dim * sizeof(
#ifdef FLOAT1g
float
#else
double
#endif
),
					cudaMemcpyHostToDevice));

	//	checkCudaErrors(cudaMemcpy(data_gpu, data, l * sizeof(float),
	//										cudaMemcpyHostToDevice));

//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	kernel<<<blocks,threads>>>(data_gpu,l,dim,gamma,y_gpu,x_gpu,QD_gpu);

//	kernel2<<<block,thread>>>(data_gpu,l,dim,gamma,y_gpu,x_gpu,QD_gpu);

	delete[] x_cpu;
//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
//			CUDA_CHECK_RETURN(cudaGetLastError());
	checkCudaErrors(
			cudaMemcpy(data, data_gpu, sizeof(float) * l*l,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(QD, QD_gpu, sizeof(
#ifdef FLOAT1g
float
#else
double
#endif
) * l, cudaMemcpyDeviceToHost));



	//	for (int i=0;i<len;i++)
	//	printf("predicted %d class: %f \n",i,result[i]);


}
