# NbodySimGPU
Compute the symplectic integrator method described by Hanno Rein in Embedded operator splitting methods for perturbed systems optimized for the GPU.
In this directory, the arrays have been changed from 2D arrays to 1D arrays in order to pass them to the GPU easily.
On my system, I need to convert the arrays of doubles to arrays of floats in order to pass them to the GPU but on the MIST cluster, I should be able to remedy that.

