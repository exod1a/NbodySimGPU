// runSim.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Executes the A1 operator optimized
__global__ void A1_kernel(double* r, double* v, double dt) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    r[id] += v[id] * dt;
}

// Executes the A2 operator
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *varr, double *status, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	double invdist;
	double dirvec[3];

	if (id < numParticles) {
		dirvec[0] = r[0] - r[3*id];
		dirvec[1] = r[1] - r[3*id+1];
		dirvec[2] = r[2] - r[3*id+2];

		// Distance between particle 0 and i
		invdist = status[id] * dt * rnorm3d(dirvec[0], dirvec[1], dirvec[2])*\
						   		    rnorm3d(dirvec[0], dirvec[1], dirvec[2])*\
					   		   		rnorm3d(dirvec[0], dirvec[1], dirvec[2]);
		
		if (invdist == 0 || isnan(invdist)) {
        	v[3*id]   += 0;
        	v[3*id+1] += 0;
        	v[3*id+2] += 0;

        	varr[id]                = 0;
        	varr[numParticles+id]   = 0;
        	varr[2*numParticles+id] = 0;
		}		
		else {	
			// Update velocities of particles 1 through N-1
			v[3*id]   += m[0] * invdist * dirvec[0];
			v[3*id+1] += m[0] * invdist * dirvec[1];
			v[3*id+2] += m[0] * invdist * dirvec[2];

			varr[id]                = -m[id] * invdist * dirvec[0];
			varr[numParticles+id]   = -m[id] * invdist * dirvec[1];
			varr[2*numParticles+id] = -m[id] * invdist * dirvec[2];
		}

        varr[0]              = v[0];
        varr[numParticles]   = v[1];
    	varr[2*numParticles] = v[2];
	}
}

// Execute the B operator when only embryo and other particles interact
__global__ void B_kernel(double *r, double *v, double *m, double *varr, double dt, int numParticles, double *status, double eps) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2;
    double dirvec[3];
    double invdist;

    if (id < numParticles) {
        dirvec[0] = r[3]   - r[3*id];
        dirvec[1] = r[3+1] - r[3*id+1];
        dirvec[2] = r[3+2] - r[3*id+2];

        invdist = dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps)*\
					      	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps)*\
  					     	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps));

		// update id'th satelitesimal 
        v[3*id]   += m[1] * invdist * dirvec[0] * status[id];
        v[3*id+1] += m[1] * invdist * dirvec[1] * status[id];
        v[3*id+2] += m[1] * invdist * dirvec[2] * status[id];

        // update embryo
        // Store forces on embryo for reduction
        varr[0]                =   v[3];
		varr[numParticles-1]   =      0;
        varr[numParticles]     = v[3+1];
		varr[2*numParticles-1] = 	   0;
        varr[2*numParticles]   = v[3+2];
		varr[3*numParticles-1] =      0;

        varr[id-1]                = -m[id] * invdist * dirvec[0];
        varr[numParticles+id-1]   = -m[id] * invdist * dirvec[1];
        varr[2*numParticles+id-1] = -m[id] * invdist * dirvec[2];
	}
}

__global__ void mergeEject(double *r, double *status, int numParticles, double rH) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1; //change 1 back to 2
    double dist;

    if (id < numParticles) {
        dist = norm3d(r[0]-r[3*id], r[1]-r[3*id+1], r[2]-r[3*id+2]);

        if (dist < 0.03*rH) 
            status[id] = 0;
    	else if (dist > rH)
			status[id] = 2;  // so that momentum conservation doesn't include ejected particles
							 // will be set to 0 in the consMomentum function
	}
}

__global__ void collision(double *r, double *status, double *rSatellites, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2;
    double dist;

	if (id < numParticles) {
        dist = norm3d(r[3]-r[3*id], r[4]-r[3*id+1], r[5]-r[3*id+2]);

        if (dist < rSatellites[0] + rSatellites[1])
            status[id] = 3; // for consMomentum function so only collision particles update embryo radius
							// will be set to 0 in the consMomentum function
	}
}

__global__ void consMomentum(double *v, double *m, double *status, int numParticles, int particleNum, double *rSatellites) {
// change id back to 2
	for (int id = 2; id < numParticles; id++) {
		if (status[id] == 0) {
			// use conservation of momentum to update central velocity
    		v[3*particleNum]    = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum]   + m[id]*v[3*id]);
    		v[3*particleNum+1]  = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum+1] + m[id]*v[3*id+1]);
    		v[3*particleNum+2]  = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum+2] + m[id]*v[3*id+2]);
    		// conservation of mass
    		m[particleNum]     += m[id];
		}
		else if (status[id] == 3) {
			status[id] 	       = 0;
			rSatellites[0]     = cbrt((m[1]+m[2])/m[2])*rSatellites[1];
			// use conservation of momentum to update velocity
            v[3*particleNum]   = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum]   + m[id]*v[3*id]);
            v[3*particleNum+1] = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum+1] + m[id]*v[3*id+1]);
            v[3*particleNum+2] = 1./(m[particleNum] + m[id]) * (m[particleNum]*v[3*particleNum+2] + m[id]*v[3*id+2]);
            // conservation of mass
            m[particleNum]    += m[id];
		}
		else if (status[id] == 2)
			status[id] = 0;
	}
}

__global__ void statusUpdate(double *r, double *v, double *m, double *status, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    m[id/3] *= status[id/3];
   	r[id] 	*= status[id/3];
    v[id] 	*= status[id/3];
}

// Function to find 
// cross product of two vector array. 
__device__ void crossProduct(double *vect_A, double *vect_B, double *cross_P) { 
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2]; 
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
} 

__global__ void missedCollision(double* r, double* v, double* status, double* rSatellites, int numParticles, double dt) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2;

	if (id < numParticles) {
    	double rTemp[3], vTemp[3], crossP[3], vecA[3], vecB[3];
    	double t, dist, d1, d2;

    	// go to rest frame of embryo
    	vTemp[0] = v[3*id]   - v[3];
    	vTemp[1] = v[3*id+1] - v[4];
    	vTemp[2] = v[3*id+2] - v[5];

    	// evolve satelitesimal
    	rTemp[0] = r[3*id]   + vTemp[0] * dt;
    	rTemp[1] = r[3*id+1] + vTemp[1] * dt;
    	rTemp[2] = r[3*id+2] + vTemp[2] * dt;

    	// the equation ((r-r[1]) * (rTemp-r)) / |rTemp-r|^2 where r[1] is the embryo's
    	// position in its rest frame, r is the satelitesimal's original position and rTemp is the
    	// satelitesimal's updated position in the rest frame. * indicates a dot product in this case
    	// this is the time that minimizes the distance function from a line segment to a point
    	t = ((r[3*id]-r[3])      *(rTemp[0]-r[3*id])    +\
         	 (r[3*id+1]-r[4])    *(rTemp[1]-r[3*id+1])  +\
         	 (r[3*id+2]-r[5])    *(rTemp[2]-r[3*id+2])) /\
        	((rTemp[0]-r[3*id])  *(rTemp[0]-r[id])      +\
         	 (rTemp[1]-r[3*id+1])*(rTemp[1]-r[3*id+1])  +\
         	 (rTemp[2]-r[3*id+2])*(rTemp[2]-r[3*id+2]));

    	if (0 <= t <= 1) {
    		// the equation |(r[1]-r) x (r[1]-rTemp)|/|rTemp-r| where r[1] is the embryo's position
    		// in its rest frame, r is the satelitesimal's original position and rTemp is the
    		// satelitesimal's updated position in the rest frame
    		// if t is in this range, then the point in within line segment
 			vecA[0] = r[3]-r[3*id],  vecA[1] = r[4]-r[3*id+1], vecA[2] = r[5]-r[3*id+2];
			vecB[0]	= r[3]-rTemp[0], vecB[1] = r[4]-rTemp[1],  vecB[2] = r[5]-rTemp[2];    	
			crossProduct(vecA, vecB, crossP);
			dist 	= norm3d(crossP[0],crossP[1],crossP[2])*rnorm3d(rTemp[0]-r[3*id], rTemp[1]-r[3*id+1], rTemp[2]-r[3*id+2]);
    	}

    	else if (t > 1 || t < 0) {
    		// if t is not in the range, it does not lie within the line segment
    		// the equation |r-r[1]|
    		d1   = norm3d(r[3*id]-r[3], r[3*id+1]-r[4], r[3*id+2]-r[5]);

    		// the equation |rTemp-r[1]|
        	d2   = norm3d(rTemp[0]-r[3], rTemp[1]-r[4], rTemp[2]-r[5]);

			dist = fmin(d1, d2); 
    	}

		if (dist < rSatellites[0] + rSatellites[1])
			status[id] = 3;
	}
}

// Find eccentricity of all particles
__global__ void calcEccentricity(double *r, double *v, double *m, double *ecc, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	double L[3];                                                            // angular momentum
	double eccTemp[3];                                                      // hold components of eccentricity vector
	double mu;          					                                // standard gravitational parameter
	double invdist;															// inverse distance between particle and central planet
	
	if (id < numParticles) {
		mu         = m[0] + m[id];	
		invdist    = rnorm3d(r[3*id]-r[0], r[3*id+1]-r[1], r[3*id+2]-r[2]);		
	
		L[0]  	   = (r[3*id+1]-r[1])*v[3*id+2] - (r[3*id+2]-r[2])*v[3*id+1];
		L[1]  	   = (r[3*id+2]-r[2])*v[3*id]   - (r[3*id]-r[0])*v[3*id+2];
		L[2]  	   = (r[3*id]-r[0])*v[3*id+1]   - (r[3*id+1]-r[1])*v[3*id];

		eccTemp[0] = (1./mu) * (v[3*id+1]*L[2] - v[3*id+2]*L[1]) - (r[3*id]-r[0])   * invdist;
		eccTemp[1] = (1./mu) * (v[3*id+2]*L[0] - v[3*id]*L[2])   - (r[3*id+1]-r[1]) * invdist;
		eccTemp[2] = (1./mu) * (v[3*id]*L[1]   - v[3*id+1]*L[0]) - (r[3*id+2]-r[2]) * invdist;

		ecc[id]    = norm3d(eccTemp[0], eccTemp[1], eccTemp[2]); // real eccentricity
	}
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

// Perform the simulation
extern "C" {
void runSim(double *r_h, double *v_h, double *m_h, double dt, int numParticles, int n, double eps, int numSteps, double *ecc_h, double *status_h, double *rSatellites_h) {
	// Declare useful variables
    size_t i, j; 
	//const unsigned int warpSize   = 32;
	size_t N                      = 3 * numParticles;
    size_t N_bytes                = N * sizeof(double);
	double rH 					  = 5.37e10/8.8605e9; // scaled 

	// Make sure the number of particles is multiple of twice the warp size (2*32)
	// for efficiency and reduction
    /*if (numParticles % (warpSize) != 0) {
    	printf("Error: The number of particles must be a multiple of the warp size (32).\n");
        return;
    }*/

	// Allocate arrays on device
    double *r_d, *v_d, *m_d, *ecc_d, *varr_d, *rSatellites_d, *status_d;
	cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
	cudaMalloc((void**) &varr_d, N_bytes);
	cudaMalloc((void**) &status_d, N_bytes/3);
	cudaMalloc((void**) &ecc_d, N_bytes/3);
	cudaMalloc((void**) &rSatellites_d, 2*sizeof(double));

	// Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(status_d, status_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(rSatellites_d, rSatellites_h, 2*sizeof(double), cudaMemcpyHostToDevice);

    collision<<<numParticles/64, 64>>>(r_d, status_d, rSatellites_d, numParticles);
    consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
    statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
    for (i = 0; i < numSteps; i++) {
        // One time step
        for (j = 0; j < n; j++) {
            missedCollision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
            collision<<<numParticles/64, 64>>>(r_d, status_d, rSatellites_d, numParticles);
			consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
			statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
			mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
			consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
			statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);

            A2_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles);
	        reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);	
			reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[1], numParticles);
			reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2], numParticles);

            missedCollision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
            collision<<<numParticles/64, 64>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
        }
        B_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, varr_d, dt, numParticles, status_d, eps);
        reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[3], numParticles);
        reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[4], numParticles);
        reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[5], numParticles);

        for (j = 0; j < n; j++) {
            missedCollision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
            collision<<<numParticles/64, 64>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);

            A2_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles);
            reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);
            reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[1], numParticles);
            reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2], numParticles);

            missedCollision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
            collision<<<numParticles/64, 64>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
        }
    }

    /*collision<<<1, numParticles>>>(r_d, status_d, rSatellites_d, numParticles);
    consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
    statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
    for (i = 0; i < numSteps; i++) {
        // One time step
        for (j = 0; j < n; j++) {
			missedCollision<<<1, numParticles>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
			consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
            collision<<<1, numParticles>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<1, numParticles>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);

            A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[1], numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2], numParticles);

            missedCollision<<<1, numParticles>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
            collision<<<1, numParticles>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<1, numParticles>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
        }
		B_kernel<<<1, numParticles>>>(r_d, v_d, m_d, varr_d, dt, numParticles, status_d, eps);
        reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[3], numParticles);
        reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[4], numParticles);
        reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[5], numParticles);

        for (j = 0; j < n; j++) {
            missedCollision<<<1, numParticles>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
            collision<<<1, numParticles>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<1, numParticles>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);

            A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[1], numParticles);
            reduce<2><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2], numParticles);

            missedCollision<<<1, numParticles>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
            collision<<<1, numParticles>>>(r_d, status_d, rSatellites_d, numParticles);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 1, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
            mergeEject<<<1, numParticles>>>(r_d, status_d, numParticles, rH);
            consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, 0, rSatellites_d);
            statusUpdate<<<1, N>>>(r_d, v_d, m_d, status_d, numParticles);
        }
    }*/

    // Copy arrays from device to host
    cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_h, m_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(status_h, status_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(rSatellites_h, rSatellites_d, 2*sizeof(double), cudaMemcpyDeviceToHost);

	printf("%.16lf\n", rSatellites_h[0]);
	for (int kk = 0; kk < numParticles; kk++) {
    	if (status_h[kk] == 0) {
        	printf("Index: %d\n", kk);
            printf("New Position\n");
            printf("%.16lf %.16lf %.16lf\n", r_h[3*kk], r_h[3*kk+1], r_h[3*kk+2]);
            printf("New Velocity\n");
            printf("%.16lf %.16lf %.16lf\n", v_h[3*kk], v_h[3*kk+1], v_h[3*kk+2]);
        }
    }
	printf("New Mass Planet\n");
	printf("%.16lf\n", m_h[0]);
    printf("New Velocity Planet\n");
    printf("%.16lf %.16lf %.16lf\n", v_h[0], v_h[1], v_h[2]);
	printf("New Mass Embryo\n");
	printf("%.16lf\n", m_h[1]);
   	printf("New Velocity Embryo\n");
    printf("%.16lf %.16lf %.16lf\n", v_h[3], v_h[4], v_h[5]);
	printf("After %d time step(s):\n", numSteps);
    printf("r\n");
    for (i = 0; i < 9; i += 3)
	    printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
    printf("...\n");
    for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
     	printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
    printf("\n");
    printf("v\n");
    for (i = 0; i < 9; i += 3)
	    printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);
    printf("\n");
    printf("...\n");

    for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
     	printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);

	// Free allocated memory on host and device
    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
	cudaFree(varr_d);
	cudaFree(status_d);
    cudaFree(ecc_d);
	cudaFree(rSatellites_d);
}
}

