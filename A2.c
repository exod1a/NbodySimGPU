// A2.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute the A2 step
void A2(double r[][3], double v[][3], double m[], double dt, int numParticles, double dirvec[])
{
	for (int i = 1; i < numParticles; i++)
	{
		for (int j = 0; j < 3; j++)
			dirvec[j] = r[0][j] - r[i][j];

		// update particles 1 -> N then particle 0		
		for (int j = 0; j < 3; j++)
		{
			v[i][j] += m[0] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2), 3./2.)) * dirvec[j] * dt;

			// -1 to account for direction along 0 to 1 instead of 1 to 0
			v[0][j] -= m[i] / (pow(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2), 3./2.)) * dirvec[j] * dt;				
		}
	}
}
