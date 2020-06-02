#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double r2()
{
    return (double)rand() / (double)RAND_MAX - 0.5;
}

int main()
{
	int N = 98*64; //10112

	for (int i = 1; i < N+1; i++)
	{
		for (int j = 1; j < 8; j++)
		{
			double u = r2();
			double v = r2();
			double w = r2();
			double x = r2();
			if (j == 1 && u < 0)				//m
				printf("%.15lf ", -u/(i*j));	 
			else if (j == 1 && u > 0)			//m
				printf("%.15lf ", u/(i*j));
			else if (j == 2)					//x
				printf("%.15lf ", v*i*j);
			else if (j == 3)					//y
				printf("%.15lf ", w*i*j);
			else if (j == 4)					//z
				printf("0.000000000000000 ");
			else if (j == 5)					//vx
				printf("%.15lf ", x/(100*i*j));
			else if (j == 6)					//vy
			{
				double v2 = 1/(pow(v*i*j, 2)+pow(w*i*j, 2));
				printf("%.15lf ", v2/sqrt(v2-pow(x/(100*i*j), 2)));
			}
			else								//vz
				printf("0.000000000000000 ");
		}
		printf("\n");
	}
}
