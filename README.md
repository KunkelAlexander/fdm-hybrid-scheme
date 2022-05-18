# FDM hybrid scheme
This Python code is a proof of concept for a hybrid scheme for simulating the Schrödinger-Poisson system that combines the wave and fluid formulation. 
It implements a wave solver (higher order FTCS) as well as a solver for the Continuity-Hamilton-Jacobi equations and adaptively switched between them based on the level of interference.

## Features
- Wave schemes (Spectral, FTCS, Crank-Nicolson)
- Fluid scheme (MUSCL-Hancock)
- Phase schemes (First-order and second order)
- Poisson solver (Explicit and implicit with finite differences, Spectral)
- CPU parallelisation for hybrid solver

## Dependencies
- numpy, scipy, matplotlib, findiff

## Usage
 - The Jupyter notebook accompanying chapter 4 is called "chapter4.ipynb". 

## Known issues
 - Convective schemes do not work properly
 - Not all schemes are stable in the standard settings
 
## Credit

https://github.com/sperseguers/gradiompy/ for wonderful integration routines

Goes to the wonderful jupyter notebooks and explanations by Philip Mocz
https://github.com/pmocz/finitevolume-python
https://github.com/pmocz/quantumspectral-python
