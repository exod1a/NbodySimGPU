#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// compute the A1 operator
__global__ void A1_kernel(double* r, double* v, double dt, size_t N)
{
    size_t id = blockIdx.x*blockDim.x + threadIdx.x;
    r[id] += v[id] * dt;
}

// compute the A2 operator
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *v0arr, size_t numParticles)
{
    int bid = blockIdx.x;

	// x, y and z components of vector that points from particle i to particle 0
    double dirvec[3];
    dirvec[0] = r[0] - r[3*(bid+1)];
    dirvec[1] = r[1] - r[3*(bid+1) + 1];
    dirvec[2] = r[2] - r[3*(bid+1) + 2];

	// distance between particle i to 0
    double dist = sqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                       (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                       (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

	// update velocities of particles 1 -> N-1 due to acceleration from 0 
    v[3*(bid+1)]   += (m[0] / dist) * dirvec[0] * dt;
    v[3*(bid+1)+1] += (m[0] / dist) * dirvec[1] * dt;
    v[3*(bid+1)+2] += (m[0] / dist) * dirvec[2] * dt;

	// deal with particle 0 since it isn't completely independent for each thread
    v0arr[0]              = v[0];
    v0arr[numParticles]   = v[1];
    v0arr[2*numParticles] = v[2];

	// store acceleration due to particles in the format:
	// v0arr = [x1, x2, ..., xN-1, y1, y2, ..., yN-1, ...]
	// i.e. x direction updates due to all particles, then y direction, then z  
    v0arr[bid+1]                = -(m[bid+1] / dist) * dirvec[0] * dt;
    v0arr[numParticles+1+bid]   = -(m[bid+1] / dist) * dirvec[1] * dt;
    v0arr[2*numParticles+1+bid] = -(m[bid+1] / dist) * dirvec[2] * dt;
}

// perform reduction to add up the x, y and z segments separately 
__global__ void reduce0arr(double *in_data) {
    size_t tid = threadIdx.x;
    size_t n   = blockDim.x;

    while (n != 0)
    {
     	if (tid < n)
            in_data[tid] += in_data[tid + n];
        __syncthreads();
        n /= 2;
    }
}

// compute the B operator
__global__ void B_kernel(double *r, double *v, double *m, double dt, size_t numParticles)
{
    size_t bid = blockIdx.x;
    double dirvec[3];
    double dist;

	// forward loop: goes from current particle to particle N-1
    for (int i = 1; i+bid+1 < numParticles; i++)
    {
		// x, y and z components of vector that points from particle j to particle k
     	dirvec[0] = r[3*(bid+1)]   - r[3*(i+bid+1)];
        dirvec[1] = r[3*(bid+1)+1] - r[3*(i+bid+1)+1];
        dirvec[2] = r[3*(bid+1)+2] - r[3*(i+bid+1)+2];

		// distance between particle j and k
        dist = sqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                    (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                    (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		// update one particle per thread
        v[3*(bid+1)]   -= (m[bid+1+i] / dist) * dirvec[0] * dt;
        v[3*(bid+1)+1] -= (m[bid+1+i] / dist) * dirvec[1] * dt;
        v[3*(bid+1)+2] -= (m[bid+1+i] / dist) * dirvec[2] * dt;
    }
    if (bid >= 1)
    {
		// backwards loop: goes from current particle to particle 1
     	for (int i = bid; i > 0; i--)
        {
            dirvec[0] = r[3*(bid+1)]   - r[3*i];
            dirvec[1] = r[3*(bid+1)+1] - r[3*i+1];
            dirvec[2] = r[3*(bid+1)+2] - r[3*i+2];

            dist = sqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                        (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                        (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

            v[3*(bid+1)]   -= (m[i] / dist) * dirvec[0] * dt;
            v[3*(bid+1)+1] -= (m[i] / dist) * dirvec[1] * dt;
            v[3*(bid+1)+2] -= (m[i] / dist) * dirvec[2] * dt;
        }
    }
}

int main()
{
    size_t numParticles = 4; size_t i;

    size_t N = 3 * numParticles;
    size_t N_bytes = N * sizeof(double);

	// allocate memory on host
    double *r_h = (double*)malloc(N_bytes);
    double *v_h = (double*)malloc(N_bytes);
    double *m_h     = (double*)malloc(N_bytes/3);
    double *v0arr_h = (double*)malloc(N_bytes);
	double *in_data_h = (double*)malloc(N_bytes/3);

	// initalize variables
    r_h[0]=0; r_h[1]=0; r_h[2]=0; r_h[3]=1; r_h[4]=0; r_h[5]=0; r_h[6]=0; r_h[7]=1.932; r_h[8]=0; r_h[9]=2.45; r_h[10]=0; r_h[11]=0;
    v_h[0]=0; v_h[1]=0; v_h[2]=0; v_h[3]=0; v_h[4]=1; v_h[5]=0; v_h[6]=-0.72; v_h[7]=0; v_h[8]=0; v_h[9]=0; v_h[10]=-0.65; v_h[11]=0;
    m_h[0]=1; m_h[1]=0.00095; m_h[2]=0.000285; m_h[3]=0.000111;

	// time step
    double dt = 0.05;

	// allocate memory on device
    double *r_d, *v_d, *m_d, *v0arr_d, *in_data_d;
    cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &v0arr_d, N_bytes);
    cudaMalloc((void**) &in_data_d, N_bytes);

	// write to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);

	// compute one time step
	A1_kernel<<<numParticles, 3>>>(r_d, v_d, dt/4, N);
	A2_kernel<<<numParticles-1,1>>>(r_d, v_d, m_d, dt/2, v0arr_d, numParticles);

	cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(v0arr_h, v0arr_d, N_bytes, cudaMemcpyDeviceToHost);
	// perform reduction for each segment of v0arr and copy to v_h 
    for (i = 0; i < 3; i++)
    {
     	memcpy(in_data_h, v0arr_h + i*numParticles, N_bytes/3);
        cudaMemcpy(in_data_d, in_data_h, N_bytes/3, cudaMemcpyHostToDevice);
        reduce0arr<<<1,numParticles/2>>>(in_data_d);
        cudaMemcpy(in_data_h, in_data_d, N_bytes/3, cudaMemcpyDeviceToHost);
        v_h[i] = in_data_h[0];
    }

	cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(v0arr_d, v0arr_h, N_bytes, cudaMemcpyHostToDevice);
	A1_kernel<<<numParticles, 3>>>(r_d, v_d, dt/4, N);
	B_kernel<<<numParticles-1, 1>>>(r_d, v_d, m_d, dt, numParticles);
	A1_kernel<<<numParticles, 3>>>(r_d, v_d, dt/4, N);
    A2_kernel<<<numParticles-1,1>>>(r_d, v_d, m_d, dt/2, v0arr_d, numParticles);

	cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(v0arr_h, v0arr_d, N_bytes, cudaMemcpyDeviceToHost);
	// perform reduction for each segment of v0arr and copy to v_h
    for (i = 0; i < 3; i++)
    {
     	memcpy(in_data_h, v0arr_h + i*numParticles, N_bytes/3);
        cudaMemcpy(in_data_d, in_data_h, N_bytes/3, cudaMemcpyHostToDevice);
        reduce0arr<<<1,numParticles/2>>>(in_data_d);
        cudaMemcpy(in_data_h, in_data_d, N_bytes/3, cudaMemcpyDeviceToHost);
        v_h[i] = in_data_h[0];
    }
	cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(v0arr_d, v0arr_h, N_bytes, cudaMemcpyHostToDevice);
	A1_kernel<<<numParticles, 3>>>(r_d, v_d, dt/4, N);
	// end time step

	cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < N; i++)
		printf("%.15lf ", r_h[i]);
	printf("\n");
	for (i=0; i < N; i++)
		printf("%.15lf ", v_h[i]);
	printf("\n");

	// release memory allocated on host and device
    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(v0arr_d);
    cudaFree(in_data_d);
    free(r_h);
    free(v_h);
    free(m_h);
    free(v0arr_h);
    free(in_data_h);
}
