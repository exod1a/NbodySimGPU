// B.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute the B step
void B(double r[][3], double v[][3], double m[], double dt, int numParticles, double dirvec[])
{
	for (int i = 1; i < numParticles; i++)
    {
		// j th particle position, direction vector and update velocities
		for (int j = i+1; j < numParticles; j++)
        {
			for (int k = 0; k < 3; k++)
				dirvec[k] = r[i][k] - r[j][k];

			for (int k = 0; k < 3; k++)
			{        	
				v[i][k] -= m[j] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2],2), 3./2.)) * dirvec[k] * dt; 
				v[j][k] += m[i] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2],2), 3./2.)) * dirvec[k] * dt;
			}
		}
	}
}
