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

def Solve(gridParams, solver, niter, psi, pot, nonlin, propType):
    nx = gridParams['nx']
    dx = gridParams['dx']
    dt = gridParams['dt']

    # Set up diagonal coefficients
    Adiag = sp.empty(nx)
    Asup = sp.empty(nx)
    Asub = sp.empty(nx)
    bdiag = sp.empty(nx)
    bsup = sp.empty(nx)
    bsub = sp.empty(nx)
    Adiag.fill(1 + dt/dx**2)
    Asup.fill(-dt/(2*dx**2))
    Asub.fill(-dt/(2*dx**2))
    bdiag.fill(1 - dt/dx**2)
    bsup.fill(dt/(2*dx**2))
    bsub.fill(dt/(2*dx**2))

    
    # Construct tridiagonal matrix
    A = sp.sparse.spdiags([Adiag, Asup, Asub], [0, 1, -1], nx, nx)
    b = sp.sparse.spdiags([bdiag, bsup, bsub], [0, 1, -1], nx, nx)

    # Loop through time
    for t in range(0, niter) :
        # Calculate effect of potential and nonlinearity
        psi = sp.exp(-dt*(pot + nonlin*psi*psi))*psi
        
        # Calculate spacial derivatives
        psi = sp.sparse.linalg.bicg(A, b*psi)[0]

        # Normalize Psi
        psi /= sp.sqrt(sp.integrate.simps(psi*psi, dx=dx))

        # Output figures
        gridx = dx*(sp.arange(nx) - nx/2)
        xlimit = [gridx[0], gridx[-1]]
        ylimit = [0, 2*psi[nx/2]]
        pl.plot(gridx, psi)
        pl.plot(gridx, psi*psi)
        pl.plot(gridx, pot)
        pl.xlim(xlimit)
        pl.ylim(ylimit)
        pl.savefig('outputla/fig' + str(t))
        pl.clf()

    return psi
