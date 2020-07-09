// runSim.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Executes the A1 operator optimized
/*__global__ void A1_kernel(double* r, double* v, double dt)
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
	double invdist0;
	double invdisti;

	if (id < numParticles - 1)
	{
		// Direction vector between particle 0 and i
		double dirvec[3];
		dirvec[0] = r[0] - r[3*(id+1)];
		dirvec[1] = r[1] - r[3*(id+1)+1];
		dirvec[2] = r[2] - r[3*(id+1)+2];

		// Distance between particle 0 and i
		invdist0 = m[0] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
        	              		 	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
            	        		 	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		invdisti = m[id+1] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                                        (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                                        (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		// deal with out of bounds particles
		if (invdisti == 0 || isnan(invdisti))
		{
			v[3*(id+1)]   += 0;
			v[3*(id+1)+1] += 0;
			v[3*(id+1)+2] += 0;
		
			v0arr[id+1]   			   -= 0;
			v0arr[numParticles+1+id]   -= 0;
			v0arr[2*numParticles+1+id] -= 0;
		}	

		else
		{
			// Update velocities of particles 1 through N-1
			v[3*(id+1)]   += invdist0 * dirvec[0];
			v[3*(id+1)+1] += invdist0 * dirvec[1];
			v[3*(id+1)+2] += invdist0 * dirvec[2];

			// Store forces on particle 0 for reduction
			v0arr[0]              = v[0];
			v0arr[numParticles]   = v[1];
			v0arr[2*numParticles] = v[2];

			v0arr[id+1]                = -invdisti * dirvec[0];
			v0arr[numParticles+1+id]   = -invdisti * dirvec[1];
			v0arr[2*numParticles+1+id] = -invdisti * dirvec[2];
		}
	}
}

// Executes the B operator when ALL particles interact
__global__ void B_kernel(double *r, double *v, double *m, double dt, int numParticles)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	double dirvec[3];
    double invdist;

	// I don't think one loop in this function can be achieved without 
	// using double iterators which I'm not sure is faster or without
	// using logic which will undoubtedly slow it down so I will leave
	// it as is. 
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

// Execute the B operator when only embryo and other particles interact
__global__ void B_kernelnew(double *r, double *v, double *m, double *v0arr, double dt, int numParticles, double *status, double eps)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    double dirvec[3];
    double invdist;

    if (id < numParticles - 2)
	{
        dirvec[0] = r[3]   - r[3*(id+2)];
        dirvec[1] = r[3+1] - r[3*(id+2)+1];
        dirvec[2] = r[3+2] - r[3*(id+2)+2];

        invdist = dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps)*\
					      	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps)*\
  					     	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps));

		// update id'th satelitesimal 
        v[3*(id+2)]   += m[1] * invdist * dirvec[0] * status[id+2];
        v[3*(id+2)+1] += m[1] * invdist * dirvec[1] * status[id+2];
        v[3*(id+2)+2] += m[1] * invdist * dirvec[2] * status[id+2];

        // update embryo
        // Store forces on embryo for reduction
        v0arr[0]                =   v[3];
		v0arr[numParticles-1]   =      0;
        v0arr[numParticles]     = v[3+1];
		v0arr[2*numParticles-1] = 	   0;
        v0arr[2*numParticles]   = v[3+2];
		v0arr[3*numParticles-1] =      0;

        v0arr[id+1]                = -m[id+2] * invdist * dirvec[0];
        v0arr[numParticles+1+id]   = -m[id+2] * invdist * dirvec[1];
        v0arr[2*numParticles+1+id] = -m[id+2] * invdist * dirvec[2];
	}
}

// check if particles are below 0.03rH or above rH. Above get ejected, below are accreted by central planet
__global__ void mergeEject(double *r, double *v, double *m, double *status, double rH, int numParticles)
{
	double dist;

	for (int id = 1; id < numParticles; id++)
	{
		// check if distance is within the bounds
		dist = sqrt((r[3*id]-r[0])*(r[3*id]-r[0]) + (r[3*id+1]-r[1])*(r[3*id+1]-r[1]) + (r[3*id+2]-r[2])*(r[3*id+2]-r[2]));

		// if not, set its status element to 0  NEED TO UPDATE RADIUS SOMEHOW
		if (dist < 0.03*rH)
		{
			// use conservation of momentum to update central planet's velocity
			v[0]       = 1/(m[0] + m[id]) * (m[0]*v[0] + m[id]*v[3*id])   * status[id];
			v[1]       = 1/(m[0] + m[id]) * (m[0]*v[1] + m[id]*v[3*id+1]) * status[id];
			v[2]       = 1/(m[0] + m[id]) * (m[0]*v[2] + m[id]*v[3*id+2]) * status[id];
			// conservation of mass
			m[0]      += m[id];
			status[id] = 0;
		}

		// eject if too far away
		else if (dist > rH) {
			status[id] = 0;
		}

		else {
			status[id] = 1;
		}

		// multiple all components by status element
        m[id]	  *= status[id];
        r[3*id]   *= status[id];
        r[3*id+1] *= status[id];
        r[3*id+2] *= status[id];
        v[3*id]   *= status[id];
        v[3*id+1] *= status[id];
        v[3*id+2] *= status[id];
	}
}*/

__global__ void calcEccentricity(double *r, double *v, double *m, double *ecc, int numParticles)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	double L[3];                                                            // angular momentum
	double eccTemp[3];                                                      // hold components of eccentricity vector
	double mu;          					                                // standard gravitational parameter
	double invdist;															// inverse distance between particle and central planet
	
	if (id < numParticles - 1)
	{
		mu         = m[0] + m[id+1];	
		invdist    = rsqrt((r[3*(id+1)]-r[0])*(r[3*(id+1)]-r[0])+\
						   (r[3*(id+1)+1]-r[1])*(r[3*(id+1)+1]-r[1])+\
						   (r[3*(id+1)+2]-r[2])*(r[3*(id+1)+2]-r[2]));		
	
		L[0]  	   = (r[3*(id+1)+1]-r[1])*v[3*(id+1)+2] - (r[3*(id+1)+2]-r[2])*v[3*(id+1)+1];
		L[1]  	   = (r[3*(id+1)+2]-r[2])*v[3*(id+1)] - (r[3*(id+1)]-r[0])*v[3*(id+1)+2];
		L[2]  	   = (r[3*(id+1)]-r[0])*v[3*(id+1)+1] - (r[3*(id+1)+1]-r[1])*v[3*(id+1)];

		eccTemp[0] = (1./mu) * (v[3*(id+1)+1]*L[2] - v[3*(id+1)+2]*L[1]) - (r[3*(id+1)]-r[0]) * invdist;
		eccTemp[1] = (1./mu) * (v[3*(id+1)+2]*L[0] - v[3*(id+1)]*L[2]) - (r[3*(id+1)+1]-r[1]) * invdist;
		eccTemp[2] = (1./mu) * (v[3*(id+1)]*L[1] - v[3*(id+1)+1]*L[0]) - (r[3*(id+1)+2]-r[2]) * invdist;

		ecc[id]    = sqrt(eccTemp[0]*eccTemp[0] + eccTemp[1]*eccTemp[1] + eccTemp[2]*eccTemp[2]); // real eccentricity
	}
}

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// Perform the simulation
extern "C" {
void runSim(double *r_h, double *v_h, double *m_h, double dt, int numParticles, int n, double eps, int numSteps, double *ecc_h)
{
	// Declare useful variables
    size_t i, j, k; 
	const unsigned int warpSize   = 32;
	size_t N                      = 3 * numParticles;
    size_t N_bytes                = N * sizeof(double);
	const unsigned int blockDim   = 16*64/(2*2*warpSize);
	double *status_h 			  = (double*)malloc(N_bytes/3);
	double rH 					  = 5.37e10/8.6805e9; // scaled 

	for (i = 0; i < numParticles; i++) {
		status_h[i] = 1;
	}

	// Make sure the number of particles is multiple of twice the warp size (2*32)
	// for efficiency and reduction
    if (numParticles % (warpSize) != 0)
    {
    	printf("Error: The number of particles must be a multiple of the warp size (32).\n");
        return;
    }

	// Allocate arrays on host
    double *v0arr_h = (double*)malloc(N_bytes);
    double *vout_h  = (double*)malloc(2*blockDim*sizeof(double)); // change to 2*blockDim*sizeof(double)
	//double *ecc_h	= (double*)malloc(N_bytes/3);	

	//for (i = 0; i < numParticles; i++) {
	//	ecc_h[i] = 0;
	//}

	// Allocate arrays on device
    double *r_d, *v_d, *m_d, *v0arr_d, *vout_d;
	double *status_d, *ecc_d;
    cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &v0arr_d, N_bytes);
    cudaMalloc((void**) &vout_d, 2*blockDim*sizeof(double));  // change to 2*blockDim*sizeof(double)
	cudaMalloc((void**) &status_d, N_bytes/3);
	cudaMalloc((void**) &ecc_d, N_bytes/3);

	// Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(status_d, status_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(ecc_d, ecc_h, N_bytes/3, cudaMemcpyHostToDevice);

	calcEccentricity<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, ecc_d, numParticles);
	CudaCheckError();
	
	cudaMemcpy(ecc_h, ecc_d, N_bytes, cudaMemcpyDeviceToHost);
	printf("%.15lf\n", ecc_h[1]);

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

	/*for (i = 0; i < numSteps; i++)
	{
   		// One time step
    	for (j = 0; j < n; j++) {
        	A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
			mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
			cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(status_h, status_d, N_bytes/3, cudaMemcpyDeviceToHost);
			for (int kk = 0; kk < numParticles; kk++)
    		{
     			if (status_h[kk] == 0)
            		printf("X: %f Y: %f Z: %f Index: %d\n", r_h[kk], r_h[kk+1], r_h[kk+2], kk);
    		}
			A2_kernel<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
        	for (k = 0; k < 3; k++) {
				reduce<warpSize/2><<<numParticles/(warpSize), warpSize/2, warpSize*sizeof(double)>>>(v0arr_d+k*numParticles, vout_d, numParticles);
				reduce<2*blockDim><<<1, 2*blockDim, 4*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 4*blockDim);
			}
			A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
    		//mergeEject<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, status_d, rH, numParticles);
		}
    	B_kernelnew<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, v0arr_d, dt, numParticles, status_d, eps);
		for (k = 0; k < 3; k++) {
			reduce<warpSize/2><<<numParticles/(warpSize), warpSize/2, warpSize*sizeof(double)>>>(v0arr_d+k*numParticles, vout_d, numParticles);
			reduce<2*blockDim><<<1, 2*blockDim, 4*blockDim*sizeof(double)>>>(vout_d, &v_d[3+k], 4*blockDim);
    	}
		for (j = 0; j < n; j++) {
        	A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
			//mergeEject<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, status_d, rH, numParticles);
        	A2_kernel<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, dt/(2*n), v0arr_d, numParticles);
        	for (k = 0; k < 3; k++) {
        		reduce<warpSize/2><<<numParticles/(warpSize), warpSize/2, warpSize*sizeof(double)>>>(v0arr_d+k*numParticles, vout_d, numParticles);
            	reduce<2*blockDim><<<1, 2*blockDim, 4*blockDim*sizeof(double)>>>(vout_d, &v_d[k], 4*blockDim);
			}
			A1_kernel<<<N/warpSize, warpSize>>>(r_d, v_d, dt/(4*n));
			//mergeEject<<<numParticles/warpSize, warpSize>>>(r_d, v_d, m_d, status_d, rH, numParticles);
    	}
	}*/	
	
    // Copy arrays from device to host
    cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_h, m_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(status_h, status_d, numParticles*sizeof(double), cudaMemcpyDeviceToHost);

	/*printf("After %d time step(s):\n", numSteps);
	printf("r\n");
	for (i = 0; i < 9; i += 3)
	{
	printf("%.16lf %.16lf %.16lf\n", r_h[3], r_h[3+1], r_h[3+2]);
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
	printf("%.16lf %.16lf %.16lf\n", v_h[3], v_h[3+1], v_h[3+2]);
	}
	printf("\n");
	printf("...\n");

    for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
    {
     	printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);
    }*/

	//printf("%d\n", numParticles);

    /*for (i = 0; i < numParticles; i++)
	{
		if (status_h[i] == 0) {
     		printf("Status: %f Index: %d\n", status_h[i], i);
		}
	}

	for (i = 0; i < numParticles; i++)
	{
		if (status_h[i] == 0) {
        	printf("X: %f Y: %f Z: %f Index: %d\n", r_h[i], r_h[i+1], r_h[i+2], i);
		}
	}*/

	//printf("%.11lf\n", m_h[0]);

	// Free allocated memory on host and device
    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(v0arr_d);
    cudaFree(vout_d);
	cudaFree(status_d);
    cudaFree(ecc_d);
	free(v0arr_h);
    free(vout_h);
	free(status_h);
	//free(ecc_h);
}
}
