/// @file   A1.h
/// @author Chandler Ross
/// @date   March 17, 2020
/// @brief  Module for computing the A1 part of the Hamiltonian operator. Updates particle positions. 
#include <stdio.h>
#ifndef A1
#define A1

/// @brief  Updates positions of particles via the A1 operator
/// @param      r         A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
/// @param      v         A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
/// @param      dt        The time step over which you wish to update the positions.
/// @param  numParticles  The number of particles ie. the size of the first index of r and v.

void A1(double& r[][3], double v[][3], double dt, int numParticles);

#endif
