'''

File: run.py
Author: Hadayat Seddiqi
Date: 8.23.13
Description: Runs everything.

'''

import os
import optparse
import collections
import scipy as sp
from scipy import linalg

import solve

# Command line options
if __name__=="__main__":
    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")
    parser.add_option("-p", "--problem", dest="problem", default="",
                      type="string", help="The problem that you want to run.")
    (options, args) = parser.parse_args()
    problem = options.problem

# Parse the dirs
problem = problem.replace('/', '.')
problemPath = ''
while problem.rfind('.') > 0:
    idx = problem.rfind('.') + 1
    problemPath = problem[0:idx]
    problem = problem[idx:]

# Import proper problem file
if problem.endswith('.py'): 
    problem = problem[:-3]

try:
    params = __import__("problems." + problemPath + problem, fromlist=[problem])
except ImportError:
    print ("Unable to import config file for '%s'." % problem)
    raise SystemExit

# Create data directory
pathPrefix =  os.path.dirname(os.path.realpath(__file__)) + "/"

try:
    os.makedirs(pathPrefix + params.outputDirectory)
except OSError:
    if not os.path.isdir(pathPrefix + params.outputDirectory):
        raise

# Get parameters from problem file
dim = params.nDimensions
if dim == 1:
    gridParams = { 'dim': dim,
                   'nx': params.nXPoints,
                   'dx': params.xStep }
elif dim == 2:
    gridParams = { 'dim': dim,
                   'nx': params.nXPoints,
                   'ny': params.nYPoints,
                   'dx': params.xStep,
                   'dy': params.yStep }
elif dim == 3:
    gridParams = { 'dim': dim,
                   'nx': params.nXPoints,
                   'ny': params.nYPoints,
                   'nz': params.nZPoints,
                   'dx': params.xStep,
                   'dy': params.yStep,
                   'dz': params.zStep }
else:
    print ("Must have dimensions of 1, 2, or 3.")
    sys.exit()
gridParams.update({ 'dt': params.tStep })
nonlin = params.nonlinearTerm
psi = params.initialWaveFunction
pot = params.potential
solver = params.solver
outdir = params.outputDirectory
imagTime = params.imaginaryTime
niter = params.iterations

#
# Need to think harder about incorporating different solvers
#
#solverPath = 'solvers/'
#solverList = [ f for f in listdir(solverPath) if isfile(join(solverPath,f)) ]

# Solve it
psi = solve.Solve(gridParams, solver, niter, psi, pot, nonlin, imagTime)

# Output final snapshot
import pylab as pl
gridx = gridParams['dx']*(sp.arange(gridParams['nx']) - gridParams['nx']/2)
xlimit = [gridx[0], gridx[-1]]
ylimit = [0, 2*psi[gridParams['nx']/2]]
pl.plot(gridx, psi)
pl.plot(gridx, psi*psi)
pl.plot(gridx, pot)
pl.xlim(xlimit)
pl.ylim(ylimit)
pl.savefig(outdir + '/final')
pl.clf()
