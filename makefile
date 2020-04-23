# makefile

# compile OpenCL files with gcc FILENAME.c -framework OpenCL -o FILENAME
# run with ./FILENAME  Don't have to link .cl files

# compiler and flags
CXX = gcc
FLAGS = -Wall -std=c99 -O3 -shared

all: A1.so A2.so B.so energy.so LF_U.so

A1.so: A1.c A1.h 
	${CXX} ${FLAGS} -Wl,-install_name,A1.so -o A1.so -fPIC A1.c

A2.so: A2.c A2.h
	${CXX} ${FLAGS} -Wl,-install_name,A2.so -o A2.so -fPIC A2.c

B.so: B.c B.h
	${CXX} ${FLAGS} -Wl,-install_name,B.so -o B.so -fPIC B.c

energy.so: energy.c energy.h
	${CXX} ${FLAGS} -Wl,-install_name,energy.so -o energy.so -fPIC energy.c

LF_U.so: LF_U.c LF_U.h
	${CXX} ${FLAGS} -Wl,-install_name,LF_U.so -o LF_U.so -fPIC LF_U.c

# delete .so and .pyc files
clean:
	rm -f *.so *.pyc
