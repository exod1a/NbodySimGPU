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

#define N 100
#define __CL_ENABLE_EXCEPTIONS

// Load the kernel source code into the array source_str
const char *kernelSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"\
"#define N 100\n"\
"__kernel void sinof(__global float *A, __global float *B, __global float *C)\n"\
"{\n"\
"   size_t id = get_global_id(0);\n"\
"   C[id] = sin(A[id]) + sin(B[id]);\n"\
"}\n";
 
int main() 
{
    // problem-related declarations
    size_t N_bytes = N * sizeof(float);
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

    // host version of v
    float *A, *B; 
    A = (float*) malloc(N_bytes);
    B = (float*) malloc(N_bytes);

	int i;
    for(i = 0; i < N; i++) {
        A[i] = i;
        B[i] = N - i;
    }
 
	// retreive a list of platforms avaible
	if (clGetPlatformIDs(1, &platform_id, &num_of_platforms)!= CL_SUCCESS)
	{
		printf("Unable to get platform_id\n");
		return 1;
	}
 
	// try to get a supported GPU device
	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
	{
		printf("Unable to get device_id\n");
		return 1;
		}	

	// context properties list - must be terminated with 0
	properties[0]= CL_CONTEXT_PLATFORM;
	properties[1]= (cl_context_properties) platform_id;
	properties[2]= 0;

    // global & local number of threads
    size_t globalSize, localSize;
    globalSize = N;
    localSize = 10;

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
    k_mult = clCreateKernel(program, "sinof", &err);
 
    // create arrays on host and write them
    cl_mem Amem, Bmem, Cmem;
    Amem = clCreateBuffer(context, CL_MEM_READ_ONLY, N_bytes, NULL, NULL);
    Bmem = clCreateBuffer(context, CL_MEM_READ_ONLY, N_bytes, NULL, NULL);
	Cmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*sizeof(float), NULL, NULL);
    err = clEnqueueWriteBuffer(queue, Amem, CL_TRUE, 0, N_bytes, A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, Bmem, CL_TRUE, 0, N_bytes, B, 0, NULL, NULL);

    // set all of ze kernel args...
    err = clSetKernelArg(k_mult, 0, sizeof(cl_mem), &Amem);
    err = clSetKernelArg(k_mult, 1, sizeof(cl_mem), &Bmem);
	err = clSetKernelArg(k_mult, 2, sizeof(cl_mem), &Cmem);
 
    err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	clFinish(queue);

	float *C = (float*)malloc(N*sizeof(float));
	clEnqueueReadBuffer(queue, Cmem, CL_TRUE, 0, N*sizeof(float), C, 0, NULL, NULL );

    for (i=0; i<N; i++)
        //printf("sin(%f) = %f\n", A[i], C[i]);
		printf("sin(%f) + sin(%f) = %f\n", A[i], B[i], C[i]);

    // release OpenCL resources
    clReleaseMemObject(Amem);
    clReleaseMemObject(Bmem);
    clReleaseMemObject(Cmem);
	clReleaseProgram(program);
    clReleaseKernel(k_mult);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(A);
    free(B);
	free(C); 

    return 0;
}
