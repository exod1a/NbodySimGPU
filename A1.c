// A1.c
#include <stdio.h>

void A1(double r[][3], double v[][3], double dt, int numParticles)
{
	// update position of all particles
	for (int i = 0; i < numParticles; i++)
	{
		for (int j = 0; j < 3; j++)
			r[i][j] += v[i][j] * dt;
	}
}
