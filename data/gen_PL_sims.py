import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os


os.system('mkdir -p PL')
os.system('mkdir -p val')

nside = 64
ls = np.arange(3*nside)
cl = 1/(ls+10)**2
np.savez('PL/cl_PL.npz', ls=ls,
         clEE=cl, clEB=cl, clBE=cl, clBB=cl)

clEE = 1/(ls+10)**0.5
clBB = 0.05/(ls+10)**0.5
cl0 = np.zeros(3*nside)
np.savez('val/cl_val.npz', ls=ls,
         clEE=clEE, clEB=cl0, clBE=cl0, clBB=clBB)

nsims = 100
for i in range(nsims):
    print(i)

    # Fiducial PL sims
    seed = 1000+i
    alm = hp.synalm(cl)
    alm0 = alm*0
    mpE_Q, mpE_U = hp.alm2map_spin([alm, alm0], nside, spin=2, lmax=3*nside-1)
    mpB_Q, mpB_U = hp.alm2map_spin([alm0, alm], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f"PL/plsim_{seed}_E.fits", [mpE_Q, mpE_U], overwrite=True)
    hp.write_map(f"PL/plsim_{seed}_B.fits", [mpB_Q, mpB_U], overwrite=True)

    # Alternative PL sims
    almE = hp.synalm(clEE)
    almB = hp.synalm(clBB)
    mp_Q, mp_U = hp.alm2map_spin([almE, almB], nside, spin=2, lmax=3*nside-1)
    hp.write_map(f"val/valsim_{seed}.fits", [mp_Q, mp_U], overwrite=True)
