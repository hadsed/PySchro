'''

Author: Hadayat Seddiqi
Date: Feb. 28th, 2013
Description: n/a

'''

import scipy as sp
import numpy as np
from scipy import integrate, sparse, linalg
import scipy.sparse.linalg
import pylab as pl
import subprocess

nx = 8000
dx = 0.0025
dt = 0.00002
niter = 20
nonlin = 0.0
gridx = sp.zeros(nx)
igridx = sp.array(range(nx))
psi = sp.zeros(nx)
pot = sp.zeros(nx)
depth = 0.01

# Set up grid, potential, and initial state
gridx = dx*(igridx - nx/2)
pot = depth*gridx*gridx
psi = sp.pi**(-1/4)*sp.exp(-0.5*gridx*gridx)

# Normalize Psi
#psi /= sp.integrate.simps(psi*psi, dx=dx)

# Plot parameters
xlimit = [gridx[0], gridx[-1]]
ylimit = [0, 2*psi[nx/2]]

# Set up diagonal coefficients
Adiag = sp.empty(nx)
Asup = sp.empty(nx)
Asub = sp.empty(nx)
bdiag = sp.empty(nx)
bsup = sp.empty(nx)
bsub = sp.empty(nx)
Adiag.fill(1 - dt/dx**2)
Asup.fill(dt/(2*dx**2))
Asub.fill(dt/(2*dx**2))
bdiag.fill(1 + dt/dx**2)
bsup.fill(-dt/(2*dx**2))
bsub.fill(-dt/(2*dx**2))

# Construct tridiagonal matrix
A = sp.sparse.spdiags([Adiag, Asup, Asub], [0, 1, -1], nx, nx)
b = sp.sparse.spdiags([bdiag, bsup, bsub], [0, 1, -1], nx, nx)

# Loop through time
for t in range(0, niter) :
    # Calculate effect of potential and nonlinearity
    psi *= sp.exp(-dt*(pot + nonlin*psi*psi))

    # Calculate spacial derivatives
    psi = sp.sparse.linalg.bicg(A, b*psi)[0]

    # Normalize Psi
    psi /= sp.integrate.simps(psi*psi, dx=dx)

    # Output figures
    pl.plot(gridx, psi)
    pl.plot(gridx, psi*psi)
    pl.plot(gridx, pot)
    pl.xlim(xlimit)
    pl.ylim(ylimit)
    pl.savefig('outputla/fig' + str(t))
    pl.clf()
