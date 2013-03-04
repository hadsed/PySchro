'''

Author: Hadayat Seddiqi
Date: Feb. 24th, 2013
Description: n/a

'''

import scipy as sp
from scipy import integrate
import pylab as pl
import subprocess

nx = 8000
dx = 0.0025
dt = 0.00002
niter = 20
nonlin = 0.0
gridx = sp.zeros(nx)
igridnx = sp.array(range(nx))
psi = sp.zeros(nx)
psi2 = sp.zeros(nx)
pot = sp.zeros(nx)
depth = 0.1

# Set up grid, potential, and initial state
gridx = dx*(igridnx - nx/2)
pot = depth*(gridx*gridx)
psi = sp.pi**(-1/4)*sp.exp(-0.5*gridx*gridx)

# Normalize Psi
psi /= sp.integrate.simps(psi*psi, dx=dx)

# Plot parameters
xlimit = [gridx[0], gridx[-1]]
ylimit = [0, 2*psi[nx/2]]

# Compute imaginary time
alpha = sp.zeros(nx)
beta = sp.zeros(nx)
gamma = sp.zeros(nx)
dx2 = dx*dx
tdD = 1 - dt/dx2
tdS = 0.5*dt/dx2
bconst = 0
alpha[nx - 1] = 0.0
gamma[nx - 1] = -1.0/tdD

# Calculate alphas and gammas
for i in range(nx - 1, 0, -1) :
    alpha[i - 1] = gamma[i]*tdS
    gamma[i - 1] = -1.0/(tdD + tdS*alpha[i - 1]);

# Loop through time
for t in range(0, niter) :
    # Calculate effect of potential and nonlinearity
    psi *= sp.exp(-dt*(pot + nonlin*psi2))

    # Calculate spatial derivatives
    beta[-1] = psi[-1]
    for i in range(nx - 2, 0, -1) :
        bconst = psi[i] - (psi[i+1] - 2.0*psi[i] + psi[i-1])*tdS
        beta[i - 1] = gamma[i]*(tdS*beta[i] - bconst)

    # Normalize Psi
    psi /= sp.integrate.simps(psi*psi, dx=dx)

    # Output figures
    pl.plot(gridx, psi)
    pl.plot(gridx, pot)
    pl.xlim(xlimit)
    pl.ylim(ylimit)
    pl.savefig('output/fig' + str(t))
    pl.clf()

