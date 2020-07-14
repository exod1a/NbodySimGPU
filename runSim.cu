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

// Update either the central planet or the embryo velocity 
// depending on the stride (s) s=0 is the central planet, s=3 is the embryo
__global__ void reduce(double *v, double *varr, int numParticles, int s) {
    v[s]   = thrust::reduce(thrust::device, &varr[0], &varr[numParticles]);
	v[1+s] = thrust::reduce(thrust::device, &varr[numParticles], &varr[2*numParticles]);
	v[2+s] = thrust::reduce(thrust::device, &varr[2*numParticles], &varr[3*numParticles]);
}

// Executes the A2 operator
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *varr, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	double invdist0;
	double invdisti;

	if (id < numParticles) {
		// Direction vector between particle 0 and i
		double dirvec[3];
		dirvec[0] = r[0] - r[3*id];
		dirvec[1] = r[1] - r[3*id+1];
		dirvec[2] = r[2] - r[3*id+2];

		// Distance between particle 0 and i
		invdist0 = m[0] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
        	              		 	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
            	        		 	 (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		invdisti = m[id] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                                      (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*\
                                      (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));

		// deal with out of bounds particles
		if (invdisti == 0 || isnan(invdisti)) {
			v[3*id]   += 0;
			v[3*id+1] += 0;
			v[3*id+2] += 0;
		
			varr[id]   			    = 0;
			varr[numParticles+id]   = 0;
			varr[2*numParticles+id] = 0;
		}	

		else {
			// Update velocities of particles 1 through N-1
			v[3*id]   += invdist0 * dirvec[0];
			v[3*id+1] += invdist0 * dirvec[1];
			v[3*id+2] += invdist0 * dirvec[2];

			varr[id]                = -invdisti * dirvec[0];
			varr[numParticles+id]   = -invdisti * dirvec[1];
			varr[2*numParticles+id] = -invdisti * dirvec[2];
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

// check if particles are below 0.03rH or above rH. Above get ejected, below are accreted by central planet
__global__ void mergeEject(double *r, double *v, double *m, double *status, double rH, int numParticles) {
	double dist;

	for (int id = 1; id < numParticles; id++) {
		// check if distance is within the bounds
		dist = sqrt((r[3*id]-r[0])*(r[3*id]-r[0]) + (r[3*id+1]-r[1])*(r[3*id+1]-r[1]) + (r[3*id+2]-r[2])*(r[3*id+2]-r[2]));

		// if not, set its status element to 0  NEED TO UPDATE RADIUS SOMEHOW
		if (dist < 0.03*rH && status[id] != 0) {
			// use conservation of momentum to update central planet's velocity
			v[0]       = 1./(m[0] + m[id]) * (m[0]*v[0] + m[id]*v[3*id]);
			v[1]       = 1./(m[0] + m[id]) * (m[0]*v[1] + m[id]*v[3*id+1]);
			v[2]       = 1./(m[0] + m[id]) * (m[0]*v[2] + m[id]*v[3*id+2]);
			// conservation of mass
			m[0]      += m[id];
			status[id] = 0;
		}
		else if (dist < 0.03*rH && status[id] == 0)
			v[0] += 0, v[1] += 0, v[2] += 0;

		// eject if too far away
		else if (dist > rH)
			status[id] = 0;

		else
			status[id] = 1;

		// multiple all components by status element
        m[id]	  *= status[id];
        r[3*id]   *= status[id];
        r[3*id+1] *= status[id];
        r[3*id+2] *= status[id];
        v[3*id]   *= status[id];
        v[3*id+1] *= status[id];
        v[3*id+2] *= status[id];
	}
}

__global__ void collision(double *r, double *v, double *m, double *status, double *rSatellites, int numParticles) {
	double dist;

    for (int id = 2; id < numParticles; id++) {
        dist = sqrt((r[3] - r[3*id])  *(r[3] - r[3*id]) +   \
                    (r[4] - r[3*id+1])*(r[4] - r[3*id+1]) + \
                    (r[5] - r[3*id+2])*(r[5] - r[3*id+2]));

        if (dist < rSatellites[0] + rSatellites[1]) {
            rSatellites[0] *= cbrt(2.);
            status[id]       = 0;
            // use conservation of momentum to update central planet's velocity
            v[3]         = 1./(m[1] + m[id]) * (m[1]*v[3] + m[id]*v[3*id]);
            v[4]         = 1./(m[1] + m[id]) * (m[1]*v[4] + m[id]*v[3*id+1]);
            v[5]         = 1./(m[1] + m[id]) * (m[1]*v[5] + m[id]*v[3*id+2]);
            // conservation of mass
            m[1]           += m[id];
        }

        else
            continue;

        m[id]     *= status[id];
        r[3*id]   *= status[id];
        r[3*id+1] *= status[id];
        v[3*id+2] *= status[id];
        v[3*id]   *= status[id];
        v[3*id+1] *= status[id];
        v[3*id+2] *= status[id];
    }
}

__global__ void calcEccentricity(double *r, double *v, double *m, double *ecc, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	double L[3];                                                            // angular momentum
	double eccTemp[3];                                                      // hold components of eccentricity vector
	double mu;          					                                // standard gravitational parameter
	double invdist;															// inverse distance between particle and central planet
	
	if (id < numParticles) {
		mu         = m[0] + m[id];	
		invdist    = rsqrt((r[3*id]-r[0])*(r[3*id]-r[0])+\
						   (r[3*id+1]-r[1])*(r[3*id+1]-r[1])+\
						   (r[3*id+2]-r[2])*(r[3*id+2]-r[2]));		
	
		L[0]  	   = (r[3*id+1]-r[1])*v[3*id+2] - (r[3*id+2]-r[2])*v[3*id+1];
		L[1]  	   = (r[3*id+2]-r[2])*v[3*id] - (r[3*id]-r[0])*v[3*id+2];
		L[2]  	   = (r[3*id]-r[0])*v[3*id+1] - (r[3*id+1]-r[1])*v[3*id];

		eccTemp[0] = (1./mu) * (v[3*id+1]*L[2] - v[3*id+2]*L[1]) - (r[3*id]-r[0]) * invdist;
		eccTemp[1] = (1./mu) * (v[3*id+2]*L[0] - v[3*id]*L[2]) - (r[3*id+1]-r[1]) * invdist;
		eccTemp[2] = (1./mu) * (v[3*id]*L[1] - v[3*id+1]*L[0]) - (r[3*id+2]-r[2]) * invdist;

		ecc[id]    = sqrt(eccTemp[0]*eccTemp[0] + eccTemp[1]*eccTemp[1] + eccTemp[2]*eccTemp[2]); // real eccentricity
	}
}

// Perform the simulation
extern "C" {
void runSim(double *r_h, double *v_h, double *m_h, double dt, int numParticles, int n, double eps, int numSteps, double *ecc_h, double *status_h, double *rSatellites_h) {
	// Declare useful variables
    size_t i, j; 
	const unsigned int warpSize   = 32;
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

	collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);

	// Run the simulation
	/*for (i = 0; i < numSteps; i++) {
   		// One time step
    	for (j = 0; j < n; j++) {
        	A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
			collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
			mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
			
			A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, numParticles);
			reduce<<<1,1>>>(v_d, varr_d, numParticles, 0);
			
			A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
			collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
			mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
		}
    	B_kernel<<<1, numParticles>>>(r_d, v_d, m_d, varr_d, dt, numParticles, status_d, eps);
		reduce<<<1,1>>>(v_d, varr_d, numParticles, 3);
		
		for (j = 0; j < n; j++) {
        	A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
			collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
        	mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
			
			A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, numParticles);
			reduce<<<1,1>>>(v_d, varr_d, numParticles, 0);
			
			A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
			collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
			mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
    	}
	}*/

	// One time step
    for (j = 0; j < n; j++) {
    	A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
        collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
        mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);

        A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, numParticles);
        reduce<<<1,1>>>(v_d, varr_d, numParticles, 0);

        A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
        collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
        mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
    }
    B_kernel<<<1, numParticles>>>(r_d, v_d, m_d, varr_d, dt, numParticles, status_d, eps);
    reduce<<<1,1>>>(v_d, varr_d, numParticles, 3);

    for (j = 0; j < n; j++) {
        A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
        collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
        mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);

        A2_kernel<<<1, numParticles>>>(r_d, v_d, m_d, dt/(2*n), varr_d, numParticles);
        reduce<<<1,1>>>(v_d, varr_d, numParticles, 0);

        A1_kernel<<<1, N>>>(r_d, v_d, dt/(4*n));
        collision<<<1, 1>>>(r_d, v_d, m_d, status_d, rSatellites_d, numParticles);
        mergeEject<<<1, 1>>>(r_d, v_d, m_d, status_d, rH, numParticles);
    }

    // Copy arrays from device to host
    cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_h, m_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(status_h, status_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(rSatellites_h, rSatellites_d, 2*sizeof(double), cudaMemcpyDeviceToHost);

	/*printf("%.16lf\n", rSatellites_h[0]);
	for (int kk = 0; kk < numParticles; kk++) {
    	if (status_h[kk] == 0) {
        	printf("Index: %d\n", kk);
            printf("New Position\n");
            printf("%.16lf %.16lf %.16lf\n", r_h[3*kk], r_h[3*kk+1], r_h[3*kk+2]);
            printf("New Velocity\n");
            printf("%.16lf %.16lf %.16lf\n", v_h[3*kk], v_h[3*kk+1], v_h[3*kk+2]);
        }
    }
	printf("New Mass\n");
	printf("%.16lf\n", m_h[1]);
   	printf("New Velocity Embryo\n");
    printf("%.16lf %.16lf %.16lf\n", v_h[3], v_h[4], v_h[5]);
	printf("After %d time step(s):\n", numSteps);
    printf("r\n");
    for (i = 0; i < N; i += 3)
	    printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
    printf("...\n");*/
    /*for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
     	printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[i+1], r_h[i+2]);
    printf("\n");*/
    /*printf("v\n");
    for (i = 0; i < N; i += 3)
	    printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);
    printf("\n");
    printf("...\n");*/

    /*for (i = 3*numParticles - 9; i < 3*numParticles; i += 3)
     	printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[i+1], v_h[i+2]);*/

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

