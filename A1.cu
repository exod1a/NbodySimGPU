//A1.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void A1_kernel(double* r, double* v, double dt)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	r[id] += v[id] * dt;
}

extern "C" {
void A1(double* r_h, double* v_h, double dt, int numParticles)
{
    size_t N = 3 * numParticles;
    size_t N_bytes = N * sizeof(double);

    double *r_d, *v_d;
    cudaMalloc(&r_d, N_bytes);
    cudaMalloc(&v_d, N_bytes);

    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);

    A1_kernel<<<N, 3>>>(r_d, v_d, dt);

    cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);

    cudaFree(r_d);
    cudaFree(v_d);
}
}
