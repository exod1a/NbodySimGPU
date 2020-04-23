/// @file   LF_U.h
/// @author Chandler Ross
/// @date   March 19, 2020
/// @brief  Compute the leap frog potential operator.
#include <stdio.h>
#ifndef LF_U
#define LF_U

/// @brief Update velocities with the leap frog potential operator.
/// @param      r         A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
/// @param      v         A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
/// @param      m         A 1D array: contains the masses for particle 0, 1, ..., N-1.
/// @param      dt        The time step over which you wish to update the positions.
/// @param  numParticles  The number of particles ie. the size of the first index of r and v.
/// @param    dirvec      1D array: to store the output of dirVec function.
void LF_U_Op(double r[][3], double v[][3], double m[], dt, numParticles, double dirvec[]);

#endif
