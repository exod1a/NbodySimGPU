// B.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void B_kernel(double *r, double *v, double *m, double dt, int numParticles)
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

extern "C" {
void B(double *r_h, double *v_h, double *m_h, double dt, int numParticles)
{
    size_t N_bytes = 3 * numParticles * sizeof(double);

    double *r_d, *v_d, *m_d;
	cudaMalloc((void**) &r_d, N_bytes);
	cudaMalloc((void**) &v_d, N_bytes);
	cudaMalloc((void**) &m_d, N_bytes/3);

	cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);

	B_kernel<<<numParticles, 1>>>(r_d, v_d, m_d, dt, numParticles);

	cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);

	cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
}
}
