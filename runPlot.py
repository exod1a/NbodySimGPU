### @file   runPlot.py
### @author Chandler Ross
### @date   March 19, 2020
### @brief  Plots the output from the N body simulation.

import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

drift = ctypes.CDLL('./A1.so')
kickA = ctypes.CDLL('./A2.so')
kickB = ctypes.CDLL('./B.so')

### @brief Module computes the method and plots the output.
### @param      r         A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
### @param      v         A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
### @param      m         
### @param    numSteps    Integer > 0... The number of times the loop iterates. Sets how long the simulation runs.
### @param  numParticles  The number of particles ie. the size of the first index of r and v.
### @param      dt        The time step over which you wish to update the positions.
### @param      n         Integer > 0... Lower the timestep and how many times you call A1 and A2.
def runPlot(r, v, m, numSteps, numParticles, dt, n):
	# Store the updated values
	# Format: Rx = [x01,x11,...,xN1,x02,x12,...,xN2,...]
	# First digit is the particle, second is the time step
	Rx = np.zeros(numSteps*numParticles)
	Ry = np.zeros(numSteps*numParticles)
	Rz = np.zeros(numSteps*numParticles)

	# array to find direction vector along particle j to particle i
	dirvec = np.zeros(3)

	for i in np.arange(numSteps):
		for j in np.arange(numParticles):
			# x,y and z components of each planet
			# for each time step
			Rx[numParticles*i+j] = r[j][0]
			Ry[numParticles*i+j] = r[j][1]
			Rz[numParticles*i+j] = r[j][2]

		for k in np.arange(n):
			drift.A1(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                 	 ctypes.c_double(dt/(n*4.)), ctypes.c_uint(numParticles))

			kickA.A2(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	 m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt/(n*2.)), ctypes.c_uint(numParticles),  \
                 	 dirvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

			drift.A1(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	 ctypes.c_double(dt/(n*4.)), ctypes.c_uint(numParticles))

		# dirvec will now hold the direction vector along particle j to particle i

		kickB.B(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_uint(numParticles),  \
                dirvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	
		for k in np.arange(n):	
			drift.A1(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	 ctypes.c_double(dt/(n*4.)), ctypes.c_uint(numParticles))

			kickA.A2(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	 m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt/(n*2.)), ctypes.c_uint(numParticles),  \
                	 dirvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

			drift.A1(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	 ctypes.c_double(dt/(n*4.)), ctypes.c_uint(numParticles))

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	for i in np.arange(numParticles):
		ax.plot(Rx[i::numParticles],Ry[i::numParticles],Rz[i::numParticles])
	plt.title("Real Space N Body Problem: HR")
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.show()
