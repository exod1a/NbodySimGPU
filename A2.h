/// @file   A2.h
/// @author Chandler Ross
/// @date   March 17, 2020
/// @brief  Module copmutes A2, norm squared of a vector, and the direction vector between 2 points. 
#include <stdio.h>
#ifndef A2
#define A2

/// @brief Computes the A2 operator part of the Hamiltonian
/// @param      r         A 1D array: Lists the x,y,z position of particle 0, then 1, ...
/// @param      v         A 1D array: Lists the vx,vy,vz position of particle 0, then 1, ...
/// @param      m         A 1D array: contains the masses for particle 0, 1, ..., N-1.
/// @param      dt        The time step over which you wish to update the positions.
/// @param  numParticles  The number of particles ie. the size of r divided by 3..
template <unsigned int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid);
template <unsigned int blockSize>
__global__ void reduce(double *g_idata, double *g_odata);
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *v0arr, int numParticles);

#endif
