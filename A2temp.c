// A2.c

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

// Load the kernel source code into the array source_str
const char *kernelSource =
"__kernel void A2_kernel(__global float* r, __global float* v, __global float* m, float dt)\n"\
"{\n"\
"   size_t idx = get_global_id(0);\n"\
"	float dirvec[3];\n"\
"	dirvec[0] = r[0] - r[3*(idx+1)];\n"\
"	dirvec[1] = r[1] - r[3*(idx+1) + 1];\n"\
"   dirvec[2] = r[2] - r[3*(idx+1) + 2];\n"\
"	float dist = sqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])*(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]));\n"\
"\n"\
"   v[3*(idx+1)]   += (m[0] / dist) * dirvec[0] * dt;\n"\
"   v[3*(idx+1)+1] += (m[0] / dist) * dirvec[1] * dt;\n"\
"	v[3*(idx+1)+2] += (m[0] / dist) * dirvec[2] * dt;\n"\
"   v[0]           -= (m[idx+1] / dist) * dirvec[0] * dt;\n"\
"   v[1]           -= (m[idx+1] / dist) * dirvec[1] * dt;\n"\
"	v[2] 		   -= (m[idx+1] / dist) * dirvec[2] * dt;\n"\
"}\n";

// for some reason, ctypes doesn't let me send arguments as floats
void A2(double* r_h, double* v_h, double* m_h, double dt_h, int numParticles)
{
    size_t N = 3 * numParticles;
    size_t N_bytes = N * sizeof(float);

    float* r_hnew = (float*)malloc(N_bytes);
    float* v_hnew = (float*)malloc(N_bytes);
	float* m_hnew = (float*)malloc(numParticles * sizeof(float));
    float dt_hnew = (float) dt_h;

    // convert to floats so it can be used on GPU
    int i;
    for (i = 0; i < N; i++)
    {
		if (i < numParticles)
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
    globalSize = numParticles - 1;
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

    k_mult = clCreateKernel(program, "A2_kernel", &err);

    // create arrays on host and write them
    cl_mem r_d, v_d, m_d;
    r_d = clCreateBuffer(context, CL_MEM_READ_ONLY, N_bytes, NULL, NULL);
    v_d = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    m_d = clCreateBuffer(context, CL_MEM_READ_ONLY, numParticles * sizeof(float), NULL, NULL);
	err = clEnqueueWriteBuffer(queue, r_d, CL_TRUE, 0, N_bytes, r_hnew, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, v_d, CL_TRUE, 0, N_bytes, v_hnew, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, m_d, CL_TRUE, 0, numParticles * sizeof(float), m_hnew, 0, NULL, NULL);

    // set kernel arguments
    err = clSetKernelArg(k_mult, 0, sizeof(r_d), &r_d);
    err = clSetKernelArg(k_mult, 1, sizeof(v_d), &v_d);
	err = clSetKernelArg(k_mult, 2, sizeof(m_d), &m_d);
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

