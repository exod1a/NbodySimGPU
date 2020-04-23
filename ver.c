#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int main(int argc, char* const argv[]) {
    /*cl_uint num_devices, i;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    char buf[128];
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, buf, NULL);
        fprintf(stdout, "Device %s supports ", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 128, buf, NULL);
        fprintf(stdout, "%s\n", buf);
    }*/

	cl_platform_id platform_id;
	cl_device_id device_id;
    cl_uint num_of_devices=0;
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_of_devices);

	//cl_device_fp_config cfg;
	char cfg[2000];
	//clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
	clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 2000*sizeof(char), &cfg, NULL);
	//printf("Double FP config = %llu\n", cfg);
	printf("%s\n", cfg);
}
