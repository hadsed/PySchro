'''

Author: Hadayat Seddiqi
Date: Feb. 28th, 2013
Description: n/a

'''

import scipy as sp
from scipy import integrate

# Simulation params
iterations              = 100
solver                  = 'SplitStepCrankNicolson'
outputDirectory         = 'data/'
imaginaryTime           = False

# Grid parameters
nDimensions             = 1
nXPoints                = 800
xStep                   = 0.025
tStep                   = 0.02

# Build the grid
igx                     = sp.arange(nXPoints)
xGrid                   = xStep*(igx - nXPoints/2)

# Numerical value of nonlinear term
nonlinearTerm           = 10.0

# Specify intial wavefunction
psi                     = sp.exp(-0.5*xGrid**2)
initialWaveFunction     = psi / sp.sqrt(sp.integrate.simps(psi*psi, dx=xStep))

# Specify potential
depth                   = 0.0
potential               = depth*xGrid**2
