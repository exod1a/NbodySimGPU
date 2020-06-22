// A2.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

/*template <unsigned int blockSize>
__global__ void reduce(double *g_idata, double *g_odata, unsigned int n)
{
	extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}
	__syncthreads();
	
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}*/

template <unsigned int blockSize>
__global__ void reduce(double *g_idata, double *g_odata)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s > 32; s >>= 1)
    {
     	if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *v0arr, int numParticles)
{
    int bid = blockIdx.x;

    double dirvec[3];
    dirvec[0] = r[0] - r[3*(bid+1)];
    dirvec[1] = r[1] - r[3*(bid+1) + 1];
    dirvec[2] = r[2] - r[3*(bid+1) + 2];

    double dist = sqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                       (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                       (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

    v[3*(bid+1)]   += (m[0] / dist) * dirvec[0] * dt;
    v[3*(bid+1)+1] += (m[0] / dist) * dirvec[1] * dt;
    v[3*(bid+1)+2] += (m[0] / dist) * dirvec[2] * dt;

    v0arr[0]              = v[0];
    v0arr[numParticles]   = v[1];
    v0arr[2*numParticles] = v[2];

    v0arr[bid+1]                = -(m[bid+1] / dist) * dirvec[0] * dt;
    v0arr[numParticles+1+bid]   = -(m[bid+1] / dist) * dirvec[1] * dt;
    v0arr[2*numParticles+1+bid] = -(m[bid+1] / dist) * dirvec[2] * dt;
}

extern "C" {
void A2(double* r_h, double* v_h, double* m_h, double dt, int numParticles)
{
    size_t i; unsigned int warpSize = 32;
    size_t N_bytes = 3 * numParticles * sizeof(double);
	size_t numBlocks = numParticles/(2*warpSize);

	if (numParticles % warpSize != 0)
	{
		printf("Error: The number of particles must be a multiple of the warp size (32).");
		return;
	}

	double *v0arr_h = (double*)malloc(N_bytes);
	double *vout_h  = (double*)malloc(numBlocks*sizeof(double));

    double *r_d, *v_d, *m_d, *v0arr_d, *vout_d;
    cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &v0arr_d, N_bytes);
	cudaMalloc((void**) &vout_d, numBlocks*sizeof(double));

    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);

    A2_kernel<<<numParticles-1,1>>>(r_d, v_d, m_d, dt, v0arr_d, numParticles);

    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);

	for (i = 0; i < 3; i++)
    {
        reduce<32><<<numBlocks, warpSize, N_bytes/3>>>(v0arr_d+i*numParticles, vout_d);
        reduce<32><<<1, warpSize, N_bytes/3>>>(vout_d, vout_d);
        cudaMemcpy(vout_h, vout_d, sizeof(double), cudaMemcpyDeviceToHost);
        v_h[i] = vout_h[0];
    }

    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(v0arr_d);
	cudaFree(vout_d);
    free(v0arr_h);
	free(vout_h);
}
}
