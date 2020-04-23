/// @file   B.h
/// @author Chandler Ross
/// @date   March 17, 2020
/// @brief  Module for computing the B part of the Hamiltonian
#include <stdio.h>
#ifndef B
#define B

/// @brief Computes the A2 operator part of the Hamiltonian
/// @param      r         A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
/// @param      v         A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
/// @param      m         A 1D array: contains the masses for particle 0, 1, ..., N-1.
/// @param      dt        The time step over which you wish to update the positions.
/// @param  numParticles  The number of particles ie. the size of the first index of r and v.
/// @param    dirvec      1D array: to store the output of dirVec function.
void B(double r[][3], double v[][3], double m[], double dt, int numParticles, double dirvec[]);

#endif
