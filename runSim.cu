// runSim.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Executes the A1 operator optimized
__global__ void A1_kernel(double* r, double* v, double dt)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    r[id] += v[id] * dt;
}

// Reduce last warp (unrolled) in reduction for A2 operator
template <unsigned int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid)
{
	// All statements evaluated at compile time
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

// Reduction kernel for A2 operator for particle 0
template <unsigned int blockSize>
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
}

// Executes the A2 operator
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *v0arr, int numParticles)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numParticles - 1)
	{
		// Direction vector between particle 0 and i
		double dirvec[3];
		dirvec[0] = r[0] - r[3*(id+1)];
		dirvec[1] = r[1] - r[3*(id+1)+1];
		dirvec[2] = r[2] - r[3*(id+1)+2];

		// Distance between particle 0 and i
		double invdist = dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
        	              		 	(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
            	        		 	(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		// Update velocities of particles 1 through N-1
		v[3*(id+1)]   += m[0] * invdist * dirvec[0];
		v[3*(id+1)+1] += m[0] * invdist * dirvec[1];
		v[3*(id+1)+2] += m[0] * invdist * dirvec[2];

		// Store forces on particle 0 for reduction
		v0arr[0]              = v[0];
		v0arr[numParticles]   = v[1];
		v0arr[2*numParticles] = v[2];

		v0arr[id+1]                = -m[id+1] * invdist * dirvec[0];
		v0arr[numParticles+1+id]   = -m[id+1] * invdist * dirvec[1];
		v0arr[2*numParticles+1+id] = -m[id+1] * invdist * dirvec[2];
	}
}

// Executes the B operator
__global__ void B_kernel(double *r, double *v, double *m, double dt, int numParticles)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	double dirvec[3];
    double invdist;

	if (id < numParticles - 1)
	{
    	// forward loop: goes from current particle to particle N-1
    	for (int i = 1; i+id+1 < numParticles; i++)
    	{
     		// x, y and z components of vector that points from particle j to particle k
        	dirvec[0] = r[3*(id+1)]   - r[3*(i+id+1)];
        	dirvec[1] = r[3*(id+1)+1] - r[3*(i+id+1)+1];
        	dirvec[2] = r[3*(id+1)+2] - r[3*(i+id+1)+2];

        	// distance between particle j and k
        	invdist = m[i+id+1] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
            	        		 			 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                	    		 			 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

        	// update one particle per thread
        	v[3*(id+1)]   -= invdist * dirvec[0];
        	v[3*(id+1)+1] -= invdist * dirvec[1];
        	v[3*(id+1)+2] -= invdist * dirvec[2];
    	}
    	// backwards loop: goes from current particle to particle 1
    	for (int i = id; i > 0; i--)
    	{
     		dirvec[0] = r[3*(id+1)]   - r[3*i];
        	dirvec[1] = r[3*(id+1)+1] - r[3*i+1];
        	dirvec[2] = r[3*(id+1)+2] - r[3*i+2];

        	invdist = m[i] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
            	        	 		    (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                	   			   	    (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

        	v[3*(id+1)]   -= invdist * dirvec[0];
        	v[3*(id+1)+1] -= invdist * dirvec[1];
        	v[3*(id+1)+2] -= invdist * dirvec[2];
    	}
	}
}

// Perform the simulation
extern "C" {
void runSim(double *r_h, double *v_h, double *m_h, double dt, int numParticles, int n, int numSteps)
{
	// Declare useful variables
    size_t i, j, k; 
	const unsigned int warpSize = 32;
	size_t N = 3 * numParticles;
    size_t N_bytes = N * sizeof(double);
	const unsigned int blockDim = 158*64/(2*2*warpSize); //need to know number of particles ahead of time

	// Make sure the number of particles is multiple of twice the warp size (2*32)
	// for efficiency and reduction
    if (numParticles % (2*warpSize) != 0)
    {
    	printf("Error: The number of particles must be a multiple of two time the warp size (2 * 32 = 64).\n");
        return;
    }

	// Allocate arrays on host
    double *v0arr_h = (double*)malloc(N_bytes);
    double *vout_h  = (double*)malloc(2*blockDim*sizeof(double)); // change to 2*blockDim*sizeof(double)
	
	// Allocate arrays on device
    double *r_d, *v_d, *m_d, *v0arr_d, *vout_d;
    cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &v0arr_d, N_bytes);
    cudaMalloc((void**) &vout_d, 2*blockDim*sizeof(double));  // change to 2*blockDim*sizeof(double)

	// Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);

	// Use for fewer than 64 particles
	/*for (j = 0; j < n; j++) {
		A1_kernel<<<N, 1>>>(r_d, v_d, dt/(4*n));	
    	A2_kernel<<<numParticles-1, 1>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
		for (k = 0; k < 3; k++) {
     		reduce<2><<<1, numParticles/(2), N_bytes/3>>>(v0arr_d+k*numParticles, &v_d[k], numParticles);
			//reduce<blockDim><<<1, blockDim, 2*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 2*blockDim);
		}
		A1_kernel<<<N, 1>>>(r_d, v_d, dt/(4*n));
	}
	B_kernel<<<numParticles-1, 1>>>(r_d, v_d, m_d, dt, numParticles);	
	for (j = 0; j < n; j++) {
		A1_kernel<<<N, 1>>>(r_d, v_d, dt/(4*n));
    	A2_kernel<<<numParticles-1, 1>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
    	for (k = 0; k < 3; k++) {
     		reduce<2><<<1, numParticles/(2), N_bytes/3>>>(v0arr_d+k*numParticles, &v_d[k], numParticles);
        	//reduce<blockDim><<<1, blockDim, 2*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 2*blockDim);
    	}
		A1_kernel<<<N, 1>>>(r_d, v_d, dt/(4*n));
	}*/

    // Run complete simulation for desired number of time steps
    for (i = 0; i < numSteps; i++) {
    	// One time step
    	for (j = 0; j < n; j++) {
        	A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
        	A2_kernel<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
        	for (k = 0; k < 3; k++) {
        		reduce<warpSize><<<numParticles/(2*warpSize), warpSize, 2*warpSize*sizeof(double)>>>(v0arr_d+k*numParticles, vout_d, numParticles);
				reduce<blockDim><<<1, blockDim, 2*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 2*blockDim);
			}
			A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
    	}
    	B_kernel<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, dt, numParticles);
    	for (j = 0; j < n; j++) {
        	A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
        	A2_kernel<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
        	for (k = 0; k < 3; k++) {
            	reduce<warpSize><<<numParticles/(2*warpSize), warpSize, 2*warpSize*sizeof(double)>>>(v0arr_d+k*numParticles, vout_d, numParticles);
            	reduce<blockDim><<<1, blockDim, 2*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 2*blockDim);
        	}
			A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
    	}
	}

    // Copy arrays from device to host
    /*cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);

	printf("After %d time step(s):\n", numSteps);
	printf("r\n");
	for (i = 0; i < 9; i += 3)
	{
		printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
	}
    printf("...\n");
    for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
    {
     	printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
    }
	printf("\n");
	printf("v\n");
	for (i = 0; i < 9; i += 3)
	{
		printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);
	}
	printf("...\n");

    for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
    {
     	printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);
    }

	printf("%d\n", numParticles);*/

	// Free allocated memory on host and device
    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(v0arr_d);
    cudaFree(vout_d);
    free(v0arr_h);
    free(vout_h);
}
}
