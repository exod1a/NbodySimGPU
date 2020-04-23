/// @file   energy.h
/// @author Chandler Ross
/// @date   March 19, 2020
/// @brief  Calculate the energy of an N body system in gravitational potential.
#include <stdio.h>
#ifndef ENERGY
#define ENERGY

/// @brief Returns the energy of an N body system in gravitational potential.
/// @param       r        A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
/// @param       v        A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
/// @param       m        A 1D array: contains the masses for particle 0, 1, ..., N-1.
/// @param  numParticles  The number of particles ie. the size of the first index of r and v.
double energy(double r[][3], double v[][3], double m[], int numParticles);

#endif
