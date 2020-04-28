/// @file   A1.h
/// @author Chandler Ross
/// @date   March 17, 2020
/// @brief  Module for computing the A1 part of the Hamiltonian operator. Updates particle positions. 
#include <stdio.h>
#ifndef A1
#define A1

/// @brief  Updates positions of particles via the A1 operator
/// @param      r         A 1D array: Lists the x,y,z position of particle 0, then 1, ...
/// @param      v         A 1D array: Lists the vx,vy,vz data of particle 0, then 1, ... 
/// @param      dt        The time step over which you wish to update the positions.
/// @param  numParticles  The number of particles ie. the size of the r vector divided by 3.

void A1(double* r_h, double* v_h, double dt_h, int numParticles);

#endif
