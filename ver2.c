cl_device_fp_config cfg;
clGetDeviceInfo(devicesIds[0], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
printf("Double FP config = %llu\n", cfg);
