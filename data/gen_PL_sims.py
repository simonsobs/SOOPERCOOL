import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os


os.system('mkdir -p PL')

nside = 64
ls = np.arange(3*nside)
cl = 1/(ls+10)**2
np.savez('PL/cl_PL.npz', ls=ls,
         clEE=cl, clEB=cl, clBE=cl, clBB=cl)

nsims = 100
for i in range(nsims):
    print(i)
    seed = 1000+i
    alm = hp.synalm(cl)
    alm0 = alm*0

    mpE_Q, mpE_U = hp.alm2map_spin([alm, alm0], nside, spin=2, lmax=3*nside-1)
    mpB_Q, mpB_U = hp.alm2map_spin([alm0, alm], nside, spin=2, lmax=3*nside-1)

    hp.write_map(f"PL/plsim_{seed}_E.fits", [mpE_Q, mpE_U], overwrite=True)
    hp.write_map(f"PL/plsim_{seed}_B.fits", [mpB_Q, mpB_U], overwrite=True)
