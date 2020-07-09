// energy.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy.h"

// used to calculate energy with interactions between all particles
double energy(double* r, double* v, double* m, int numParticles)
{
	double T = 0;  // kinetic energy
	double U = 0;  // potential energy

	// to hold the vector that points between particle i and particle j
	double* dirvec = (double*)malloc(3 * sizeof(double));

	for (int i = 0; i < numParticles; i++)
	{
		T += 0.5 * m[i] * (pow(v[3*i], 2) + pow(v[3*i+1], 2) + pow(v[3*i+2], 2)); 

		for (int j = i+1; j < numParticles; j++)
		{
			for (int k = 0; k < 3; k++)
				dirvec[k] = r[3*i+k] - r[3*j+k];

			U -= m[i] * m[j] / sqrt(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2));
		}
    }
	free(dirvec);

	return T + U;
}

// used to calculate energy for interactions between central planet and
// satelitesimals and embryo and satelitesimals only
double energynew(double* r, double* v, double* m, int numParticles, double eps)
{
    double T = 0;  // kinetic energy
    double U = 0;  // potential energy
	double invdist;

    // to hold the vector that points between particle i and particle j
    double* dirvec = (double*)malloc(3 * sizeof(double));

    for (int i = 0; i < numParticles; i++)
    {
     	T += 0.5 * m[i] * (pow(v[3*i], 2) + pow(v[3*i+1], 2) + pow(v[3*i+2], 2));
    
		if (i > 0)
		{
			for (int k = 0; k < 3; k++)
        		dirvec[k] = r[k] - r[3*i+k];
			
			invdist = m[i] / sqrt(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2));
			//if (isnan(invdist))
			//	U -= 0;	
			//else
			U -= m[0] * invdist;
		}		
		if (i > 1)
		{
        	for (int k = 0; k < 3; k++)
            	dirvec[k] = r[3+k] - r[3*i+k];

			invdist = m[i] / sqrt(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2) + eps);
    	    //if (isnan(invdist))
    	    //	U -= 0;
    	    //else
            U -= m[1] * invdist;
		}	
	}
    free(dirvec);

    return T + U;
}
