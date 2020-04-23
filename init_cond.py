### @file   init_cond.py
### @author Chandler Ross
### @date   March 17, 2020
### @brief  Module for reading particle position and velocity initial conditions from file.

### @brief  Fills arrays with the initial conditions given in a file.
### @param     r      A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
### @param     v      A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
### @param     m      A 1D array: dimension is the number of particles, each element contains that particles mass.
### @param  fileName  The name of the file to read data from.  
import numpy as np

def initial_Conditions(r, v, m, fileName):
    
    File = open(fileName,"r")
    lines = File.readlines()
    
    for i in np.arange(len(lines))[1:]:
        info = lines[i].split()
        m[i-1] = float(info[0])
        r[i-1] = np.array([float(info[1]),float(info[2]),float(info[3])])
        v[i-1] = np.array([float(info[4]),float(info[5]),float(info[6])])

    File.close()
    
    return r, v, m
