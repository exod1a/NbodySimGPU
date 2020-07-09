### @file   nbodysim.py
### @author Chandler Ross
### @date   March 17, 2020
### @brief  The main driver file to execute code from all the modules in this directory for the N body simulation
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import math as M
import timeit

from init_cond import initial_Conditions
from runPlot import runPlot
#from runLFPlot import runLFPlot
from runError import runError
#from runLFError import runLFError

# Parameters for simulation
# Redefine units such that mass of Jupiter (M) = 1 and G = 1
G     		  = 6.673e-11                   	# gravitational constant
M0    		  = 1.898e27                    	# set mass scale
R0    		  = 8.8605e9                    	# set length scale
T0    	 	  = np.sqrt(R0**3/(G * M0))     	# set time scale
flag 		  = "-"								# decide what part of program to execute... -p = plot, -e = error			
dt 			  = 86400/T0						# default time step (arbitrary)
n 			  = 1								# Lowers the time step for each call to A1 and A2. Also more calls
numSteps 	  = 4     							# default number of time steps to take (arbitrary)
fileName 	  = "particles.txt"  			 	# file to read initial conditions from
File 		  = open(fileName, "r")
lines 		  = File.readlines()
numParticles  = len(lines) - 1 			       	# number of particles in simulation
File.close()
r 			  = np.zeros(3 * numParticles)		# array to hold positions of particles
v 			  = np.zeros(3 * numParticles)  	# array to hold velocities of particles
m 			  = np.zeros(numParticles)	        # array to hold masses of particles
dirvec 		  = np.zeros(3)						# array to find direction vector along particle j to particle i
timeStep_iter = np.logspace(-4,-1,20)           # loop over time steps
runTime 	  = np.zeros(len(timeStep_iter))    # the total run time
rel_err 	  = np.zeros(len(timeStep_iter))    # largest relative error
runTimeLF 	  = np.zeros(len(timeStep_iter))   	# the total run time for LF
rel_errLF 	  = np.zeros(len(timeStep_iter))    # largest relative error for LF
eps 		  = 0.001
ecc			  = np.zeros(numParticles, dtype="double")

# set ICs
r, v, m = initial_Conditions(r, v, m, fileName)

if flag == "-p":
	r, v, m = initial_Conditions(r, v, m, fileName)
	# make plot output
	runPlot(r, v ,m, numSteps, numParticles, dt, n)

elif flag == "-e":
	# make error and run time plot
	runTime, rel_err = runError(r, v, m, numParticles, n)
	
	"""plt.figure(2)
	plt.loglog(timeStep_iter,rel_err, label='HR')
	plt.legend(loc='best')
	plt.xlabel('Time Step')
	plt.ylabel('Relative Error')

	#plt.figure(3)
	#plt.loglog(runTime,rel_err, label='HR')
	#plt.legend(loc='best')
	#plt.xlabel('Run Time')
	#plt.ylabel('Relative Error')

	plt.show()"""

sim   = ctypes.CDLL('./runSim.so')
test  = ctypes.CDLL('./test.so')

"""test.testrun(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
             m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles),\
		     ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))"""

sim.runSim(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
           m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_uint(numParticles),  \
		   ctypes.c_uint(n), ctypes.c_double(eps), ctypes.c_uint(numSteps), ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

