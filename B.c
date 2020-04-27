// B.c
/*void B(double* r, double* v, double* m, double dt, int numParticles, double* dirvec)
{
	for (int i = 1; i < numParticles; i++)
    {
		// j th particle position, direction vector and update velocities
		for (int j = i+1; j < numParticles; j++)
        {
			for (int k = 0; k < 3; k++)
				dirvec[k] = r[3*i + k] - r[3*j + k];

			for (int k = 0; k < 3; k++)
			{        	
				v[3*i + k] -= m[j] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2), 3./2.)) * dirvec[k] * dt; 
				v[3*j + k] += m[i] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2), 3./2.)) * dirvec[k] * dt;
			}
		}
	}
}*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define DIM 3

// Load the kernel source code into the array source_str
const char *kernelSource =
"#define DIM 3\n"\
"__kernel void B_kernel(__global float *r, __global float *v, __global float* m, float dt)\n"\
"{\n"\
"   size_t idx = get_global_id(0);\n"\
"   float dirvec[DIM];\n"\
"   for (int k = 0; k < DIM; k++)\n"\
"       dirvec[k] = r[3*(idx+1) + k] - r[3*(idx+2) + k];\n"\
"   for (int k = 0; k < DIM; k++)\n"\
"   {\n"\
"       v[3*(idx+1) + k] += m[idx+2] / (pow(pow(dirvec[0], (float)2.) + pow(dirvec[1], (float)2.) + pow(dirvec[2], (float)2), (float)3./2.)) * dirvec[k] * dt;\n"\
"       v[3*(idx+2) + k] -= m[idx+1] / (pow(pow(dirvec[0], (float)2.) + pow(dirvec[1], (float)2.) + pow(dirvec[2], (float)2.), (float)3./2.)) * dirvec[k] * dt;\n"\
"   }\n"\
"}\n";


// for some reason, ctypes doesn't let me send arguments as floats
void B(double* r_h, double* v_h, double* m_h, double dt_h, int numParticles)
{
    size_t N = DIM * numParticles;
    size_t N_bytes = N * sizeof(float);

    float* r_hnew = (float*)malloc(N_bytes);
    float* v_hnew = (float*)malloc(N_bytes);
    float* m_hnew = (float*)malloc(N_bytes/3);
    float dt_hnew = (float) dt_h;

    // convert to floats so it can be used on GPU
    int i;
    for (i = 0; i < N; i++)
    {
        if (i < N/DIM)
            m_hnew[i] = m_h[i];

        r_hnew[i] = (float) r_h[i];
        v_hnew[i] = (float) v_h[i];
    }

    // problem-related declarations
    // openCL declarations
    cl_platform_id platform;
    cl_context context;
    cl_context_properties properties[3];
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_mult;
    cl_uint num_of_platforms=0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices=0;

    // retreive a list of platforms avaible
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms)!= CL_SUCCESS)
    {
        printf("Unable to get platform_id\n");
        return;
    }

    // try to get a supported GPU device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
    {
        printf("Unable to get device_id\n");
        return;
    }

    // context properties list - must be terminated with 0
    properties[0]= CL_CONTEXT_PLATFORM;
    properties[1]= (cl_context_properties) platform_id;
    properties[2]= 0;

    // global & local number of threads
    size_t globalSize, localSize;
    globalSize = N / DIM - 1;
    localSize = 1;

    // setup OpenCL stuff
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("building program failed\n");
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("%s\n", log);
        }
    }

    k_mult = clCreateKernel(program, "B_kernel", &err);

    // create arrays on host and write them
    cl_mem r_d, v_d, m_d;
    r_d = clCreateBuffer(context, CL_MEM_READ_ONLY, N_bytes, NULL, NULL);
    v_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    m_d = clCreateBuffer(context, CL_MEM_READ_ONLY, N_bytes/DIM, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, r_d, CL_TRUE, 0, N_bytes, r_hnew, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, v_d, CL_TRUE, 0, N_bytes, v_hnew, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, m_d, CL_TRUE, 0, N_bytes/3, m_hnew, 0, NULL, NULL);

    // set kernel arguments
    err = clSetKernelArg(k_mult, 0, sizeof(cl_mem), &r_d);
    err = clSetKernelArg(k_mult, 1, sizeof(cl_mem), &v_d);
    err = clSetKernelArg(k_mult, 2, sizeof(cl_mem), &m_d);
    err = clSetKernelArg(k_mult, 3, sizeof(float), &dt_hnew);

    err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, v_d, CL_TRUE, 0, N_bytes, v_hnew, 0, NULL, NULL );

    // give info to v_h so it updates the actual v in the python code
    // for some reason, it doesn't work if I add the v_d data to v_h from the start
    for (i=0; i<N; i++)
        v_h[i] = v_hnew[i];

    // release OpenCL resources
    clReleaseMemObject(r_d);
    clReleaseMemObject(v_d);
    clReleaseMemObject(m_d);
    clReleaseProgram(program);
    clReleaseKernel(k_mult);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(r_hnew);
    free(v_hnew);
    free(m_hnew);
}

