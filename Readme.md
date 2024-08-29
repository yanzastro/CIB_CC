# Introduction
This repo contains core functions for CIB cross-correlation. For more information see the papers https://iopscience.iop.org/article/10.1088/1475-7516/2024/05/058/pdf (unWISE x CIB), https://www.aanda.org/articles/aa/pdf/2022/09/aa43710-22.pdf (KiDS x CIB).\
The `sources` folder contains the codes for doing pseudo- $C_{\ell}$ measurement and calculating Gaussian covariance based on the `NaMaster` package. It also contains codes to run MCMC based on the `emcee` package.\
The `scripts` folder contains the codes defining the CIB profiles according to M21, S12, and Y23 models defined in the unWISE x CIB paper, as well as the associated tracers. They are based on the `PYCCL` format of halo model profiles and tracers.

# Prerequisites
In addition to general numerical calculation packages like `numpy`, `scipy`, the following packages are requested: \
`PYCCL`: https://ccl.readthedocs.io/en/latest/ (a public standardized library of routines to calculate basic observables used in cosmology)\
`NaMaster`: https://namaster.readthedocs.io/en/latest/ (is a python package that computes the angular power spectrum of masked fields with arbitrary spin using the so-called pseudo- $C_{\ell}$ formalism)\
`emcee`: https://emcee.readthedocs.io/en/stable/ (a package to run MCMC, it's optional)
