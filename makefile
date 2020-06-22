# makefile

# compile OpenCL files with gcc FILENAME.c -framework OpenCL -o FILENAME
# run with ./FILENAME  Don't have to link .cl files

# compiler and flags
CXX = gcc
CXXCUDA = nvcc
FLAGS = -O3 -Xcompiler -fPIC -shared

all: runSim.so A1.so A2.so B.so energy.so LF_U.so

runSim.so: runSim.cu
	${CXXCUDA} ${FLAGS} -o runSim.so runSim.cu

A1.so: A1.cu A1.h 
	${CXXCUDA} ${FLAGS} -o A1.so A1.cu

A2.so: A2.cu A2.h
	${CXXCUDA} ${FLAGS} -o A2.so A2.cu

B.so: B.cu B.h
	${CXXCUDA} ${FLAGS} -o B.so B.cu

energy.so: energy.c energy.h
	${CXX} -O3 -fPIC -shared -o energy.so energy.c

LF_U.so: LF_U.c LF_U.h
	${CXX} -O3 -fPIC -shared -o LF_U.so -fPIC LF_U.c

# delete .so and .pyc files
clean:
	rm -f *.so *.pyc
