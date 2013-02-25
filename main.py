'''

Author: Hadayat Seddiqi
Date: Feb. 24th, 2013
Description: n/a

'''

import scipy as sp
from scipy import integrate
import pylab as pl
import subprocess

nx = 100
dx = 0.0001
dt = 0.000001
niter = 20
nonlin = 0.0
gridx = sp.zeros(nx + 1)
psi = sp.zeros(nx + 1)
psi2 = sp.zeros(nx + 1)
pot = sp.zeros(nx + 1)

# Set up grid, potential, and initial state
for i in range(0, nx + 1):
    gridx[i] = dx*(i - nx/2)
    pot[i] = 1000*(gridx[i]**2)
    psi[i] = sp.exp(-0.5*gridx[i]**2)
    psi2[i] = psi[i]**2

# Compute imaginary time
alpha = sp.zeros(nx)
beta = sp.zeros(nx)
gamma = sp.zeros(nx)
dx2 = dx**2
tdD = 1 + dt/dx2
tdS = -0.5*dt/dx2
bconst = 0
alpha[nx - 1] = 0.0
gamma[nx - 1] = -1.0/tdD

# Calculate alphas and gammas
for i in range(nx - 1, 0, -1) :
    alpha[i - 1] = gamma[i]*tdS
    gamma[i - 1] = -1.0/(tdD + tdS*alpha[i - 1]);

# Normalize Psi
norm = sp.integrate.simps(psi2)
for i in range(0, nx + 1) : psi[i] /= norm

# Loop through time
for t in range(0, niter) :
    # Calculate effect of potential and nonlinearity
    for i in range(0, nx + 1) : psi[i] *= sp.exp(-dt*(pot[i] + nonlin*psi[i]**2))

    # Calculate spatial derivatives
    beta[nx - 1] = psi[nx]
    for i in range(nx - 1, 0, -1) :
        bconst = psi[i] - (psi[i+1] - 2.0*psi[i] + psi[i-1])*tdS
        beta[i - 1] = gamma[i]*(tdS*beta[i] - bconst)

    # Calculate Psi2
    for i in range(0, nx + 1) : psi2[i] = psi[i]**2

    # Normalize Psi
    norm = sp.integrate.simps(psi2)
    for i in range(0, nx + 1) : psi[i] /= norm

    # Output figures
    pl.plot(gridx, psi)
    pl.plot(gridx, pot)
    pl.savefig('output/fig' + str(t))
    pl.clf()
    
#subprocess.Popen("convert 
