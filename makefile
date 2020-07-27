# makefile

# compile OpenCL files with gcc FILENAME.c -framework OpenCL -o FILENAME
# run with ./FILENAME  Don't have to link .cl files

# compiler and flags
CXX = gcc
CXXCUDA = nvcc
FLAGS = -O3 -Xcompiler -fPIC -shared

all: runSim.so energy.so

runSim.so: runSim.cu
	${CXXCUDA} ${FLAGS} -lineinfo -o runSim.so runSim.cu

energy.so: energy.c energy.h
	${CXX} -O3 -fPIC -shared -o energy.so energy.c

# delete .so and .pyc files
clean:
	rm -f *.so *.pyc
