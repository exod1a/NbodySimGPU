// test.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

__global__ void calcEccentricity(double *r, double *v, double *m, double *ecc, int numParticles)
{
    //size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    double L[3];                                                            // angular momentum
    double eccTemp[3];                                                      // hold components of eccentricity vector
    double mu;                                                              // standard gravitational parameter
    double invdist;                                                         // inverse distance between particle and central planet

    //if (id < numParticles - 1)
    for (int id = 0; id < numParticles - 1; id++)
	{
     	mu         = m[0] + m[id+1];
        invdist    = rsqrt((r[3*(id+1)]-r[0])*(r[3*(id+1)]-r[0])+\
                           (r[3*(id+1)+1]-r[1])*(r[3*(id+1)+1]-r[1])+\
                           (r[3*(id+1)+2]-r[2])*(r[3*(id+1)+2]-r[2]));

        L[0]	   = (r[3*(id+1)+1]-r[1])*v[3*(id+1)+2] - (r[3*(id+1)+2]-r[2])*v[3*(id+1)+1];
        L[1]	   = (r[3*(id+1)+2]-r[2])*v[3*(id+1)]   - (r[3*(id+1)]-r[0])*v[3*(id+1)+2];
        L[2]	   = (r[3*(id+1)]-r[0])*v[3*(id+1)+1]   - (r[3*(id+1)+1]-r[1])*v[3*(id+1)];

        eccTemp[0] = (1./mu) * (v[3*(id+1)+1]*L[2] - v[3*(id+1)+2]*L[1]) - (r[3*(id+1)]-r[0])   * invdist;
        eccTemp[1] = (1./mu) * (v[3*(id+1)+2]*L[0] - v[3*(id+1)]*L[2])   - (r[3*(id+1)+1]-r[1]) * invdist;
        eccTemp[2] = (1./mu) * (v[3*(id+1)]*L[1]   - v[3*(id+1)+1]*L[0]) - (r[3*(id+1)+2]-r[2]) * invdist;

        ecc[id]    = sqrt(eccTemp[0]*eccTemp[0] + eccTemp[1]*eccTemp[1] + eccTemp[2]*eccTemp[2]); // real eccentricity
    }
}

int main()
{
	int numParticles = 2;
	size_t N_bytes   = 3 * numParticles * sizeof(double);
	double *r_h 	 = (double*)malloc(N_bytes);
	double *v_h	 	 = (double*)malloc(N_bytes); 
    double *m_h		 = (double*)malloc(N_bytes/3); 
	double *ecc_h 	 = (double*)malloc(N_bytes/3);

	r_h[0] = 0, r_h[1] = 0, r_h[2] = 0, r_h[3] = 0.1882315144676964, r_h[4] = 0, r_h[5] = 0;
	v_h[0] = 0, v_h[1] = 0, v_h[2] = 0,	v_h[3] = 0, v_h[4] = 2.2517605710860709, v_h[5] = 0;
	m_h[0] = 1, m_h[1] = 0.0000002100632244;
	ecc_h[0] = 0, ecc_h[1] = 0;

    printf("R\n");
    for (int i = 0; i < numParticles; i++)
    {
        printf("%.16lf %.16lf %.16lf\n", r_h[3*i], r_h[3*i+1], r_h[3*i+2]);
    }

    printf("V\n");
    for (int i = 0; i < numParticles; i++)
    {
     	printf("%.16lf %.16lf %.16lf\n", v_h[3*i], v_h[3*i+1], v_h[3*i+2]);
    }

    printf("M\n");
    printf("%.16lf %.16lf\n", m_h[0], m_h[1]);

    printf("Initial Eccentricity Array\n");
    printf("%.16lf %.16lf\n", ecc_h[0], ecc_h[1]);

    printf("numParticles = %d\n", numParticles);

    // Allocate arrays on device
    double *r_d, *v_d, *m_d, *ecc_d;
    cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &ecc_d, N_bytes/3);

    // Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);
    cudaMemcpy(ecc_d, ecc_h, N_bytes/3, cudaMemcpyHostToDevice);

    calcEccentricity<<<1, 1>>>(r_d, v_d, m_d, ecc_d, numParticles);
    CudaCheckError();
    cudaMemcpy(ecc_h, ecc_d, N_bytes, cudaMemcpyDeviceToHost);
    printf("Updated Eccentricity\n");
    printf("%.16lf %.16lf\n", ecc_h[0], ecc_h[1]);

    printf("What the eccentricity should be\n");
    printf("0.0455862977217524\n");

    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(ecc_d);

	free(r_h);
	free(v_h);
	free(m_h);
	free(ecc_h);

	return 0;
}

/*extern "C" {
void testrun(double *r_h, double *v_h, double *m_h, int numParticles, double *ecc_h)
{
	size_t N_bytes = 3 * numParticles * sizeof(double);

	printf("R\n");
	for (int i = 0; i < numParticles; i++)
	{
		printf("%.16lf %.16lf %.16lf\n", r_h[3*i], r_h[3*i+1], r_h[3*i+2]);
	}

	printf("V\n");
    for	(int i = 0; i <	numParticles; i++) 
    {
    	printf("%.16lf %.16lf %.16lf\n", v_h[3*i], v_h[3*i+1], v_h[3*i+2]);
    }

	printf("M\n");
    printf("%.16lf %.16lf\n", m_h[0], m_h[1]);

	printf("Initial Eccentricity Array\n");
    printf("%.16lf %.16lf\n", ecc_h[0], ecc_h[1]);

	printf("numParticles = %d\n", numParticles);

	// Allocate arrays on device
    double *r_d, *v_d, *m_d, *ecc_d;
	cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
    cudaMalloc((void**) &ecc_d, N_bytes/3);

    // Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);
    cudaMemcpy(ecc_d, ecc_h, N_bytes/3, cudaMemcpyHostToDevice);

    calcEccentricity<<<1, 1>>>(r_d, v_d, m_d, ecc_d, numParticles);	
	cudaDeviceSynchronize();
	CudaCheckError();
    cudaMemcpy(ecc_h, ecc_d, N_bytes, cudaMemcpyDeviceToHost);
	printf("Updated Eccentricity\n");
	printf("%.16lf %.16lf\n", ecc_h[0], ecc_h[1]);

	printf("What the eccentricity should be\n");
	printf("0.0455862977217524\n");

    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
    cudaFree(ecc_d);
}
}*/
