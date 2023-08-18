import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


nhits = hp.read_map("norm_nHits_SA_35FOV.fits")
mask_binary = (nhits > 0).astype(float)
hp.write_map("mask_binary.fits.gz", mask_binary)
hp.mollview(nhits)
hp.mollview(mask_binary)
plt.show()
