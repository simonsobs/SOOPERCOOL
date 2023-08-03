import argparse
import numpy as np
import pymaster as nmt
import healpy as hp

parser = argparse.ArgumentParser(description='MCM calculator')
parser.add_argument("--mask", type=str, help='Path to mask file')
parser.add_argument("--nside", type=int, help='Nside')
parser.add_argument("--output-dir", type=str, help='Output directory')
parser.add_argument("--plot", action='store_true',
                    help='Pass to generate a plot of the MCM.')
o = parser.parse_args()

# Number of ells
nl = 3*o.nside
# Number of polarisation combinations
nspec = 4

# Read and degrade mask
print("Reading mask")
mask = hp.ud_grade(hp.read_map(o.mask), nside_out=o.nside)
# Create dummy NaMaster field
f = nmt.NmtField(mask, None, spin=2)
# Binning scheme is irrelevant for us, but NaMaster needs one.
b = nmt.NmtBin(o.nside, nlb=10)

# Alright, compute and reshape coupling matrix
print("Computing MCM")
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(f, f, b)
mcm = np.transpose(w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
                   axes=[1, 0, 3, 2])

# Save to file
print("Saving")
np.savez(f"{o.output_dir}/mcm.npz", mcm=mcm)

if o.plot:
    import matplotlib.pyplot as plt

    print("Plotting")
    plt.figure()
    plt.imshow(mcm.reshape([nspec*nl, nspec*nl]))
    plt.colorbar()
    plt.savefig(f"{o.output_dir}/mcm.pdf", bbox_inches='tight')
