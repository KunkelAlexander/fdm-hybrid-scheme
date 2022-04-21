# FDM hybrid scheme
This Python code is a proof of concept for a hybrid scheme for simulating the Schrödinger-Poisson system that combines the wave and fluid formulation. 
It implements a wave solver (higher order FTCS) as well as a solver for the Continuity-Hamilton-Jacobi equations and adaptively switched between them based on the level of interference.

## Features
- Wave schemes (Spectral, FTCS, Crank-Nicolson)
- Fluid scheme (MUSCL-Hancok)
- Phase schemes (First-order and second order)
- Poisson solver (Explicit and implicit with finite differences, Spectral)
- CPU parallelisation for hybrid solver

## Dependencies
- numpy, scipy, matplotlib, findiff

## Credit
