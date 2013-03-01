'''

Author: Hadayat Seddiqi
Date: Feb. 28th, 2013
Description: n/a

'''

import scipy as sp
import numpy as np
from scipy import integrate, sparse
import scipy.sparse.linalg
import pylab as pl
import subprocess

nx = 100000
dx = 0.0001
dt = 0.000001
niter = 30
nonlin = 0.0
gridx = sp.zeros(nx)
psi = sp.zeros(nx)
psi2 = sp.zeros(nx)

pot = sp.zeros(nx)
depth = 0.00003

Adiag = np.empty(nx)
Asup = np.empty(nx)
Asub = np.empty(nx)

# Set up grid, potential, and initial state
for i in range(0, nx):
    gridx[i] = dx*(i - nx/2)
    pot[i] = depth*(gridx[i]**2)
    psi[i] = sp.exp(-0.5*gridx[i]**2)
    psi2[i] = psi[i]**2

# Normalize Psi
psi /= sp.integrate.simps(psi2)

# Plot parameters
xlimit = [gridx[0], gridx[-1]]
ylimit = [0, 2*psi[nx/2]]

# Set up diagonal coefficients
Adiag.fill(1 + dt/dx**2)
Asup.fill(-dt/(2*dx**2))
Asub.fill(-dt/(2*dx**2))

# Construct tridiagonal matrix
A = sp.sparse.spdiags([Adiag, Asup, Asub], [0, 1, -1], nx, nx)

# Loop through time
for t in range(0, niter) :
    # Calculate effect of potential and nonlinearity
    psi *= sp.exp(dt*(pot + nonlin))

    # Calculate spacial derivatives
    psi = sp.sparse.linalg.spsolve(A, psi)

    # Normalize Psi
    psi /= sp.integrate.simps(psi**2)

    # Plot parameters
    xlimit = [gridx[0], gridx[-1]]
    ylimit = [0, 2*psi[nx/2]]
    # Output figures
    pl.plot(gridx, psi)
    pl.plot(gridx, pot)
    pl.xlim(xlimit)
    pl.ylim(ylimit)
    pl.savefig('outputla/fig' + str(t))
    pl.clf()
