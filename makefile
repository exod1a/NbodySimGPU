# makefile

# compile OpenCL files with gcc FILENAME.c -framework OpenCL -o FILENAME
# run with ./FILENAME  Don't have to link .cl files

# compiler and flags
CXX = gcc
CXXCUDA = nvcc
FLAGS = -O3 -Xcompiler -fPIC -shared

all: runSim.so energy.so test.so

runSim.so: runSim.cu
	${CXXCUDA} ${FLAGS} -o runSim.so runSim.cu

test.so: test.cu
	${CXXCUDA} ${FLAGS} -o test.so test.cu

energy.so: energy.c energy.h
	${CXX} -O3 -fPIC -shared -o energy.so energy.c

# delete .so and .pyc files
clean:
	rm -f *.so *.pyc
