import argparse
import numpy as np
import pymaster as nmt
import healpy as hp
import os
from bbmaster.utils import PipelineManager


def mcmer(o):
    man = PipelineManager(o.globals)

    # Number of ells
    nl = 3*man.nside
    # Number of polarisation combinations
    nspec = 4

    # Read and degrade mask
    print("Reading mask")
    mask = hp.ud_grade(hp.read_map(man.fname_mask), nside_out=man.nside)
    # Create dummy NaMaster field
    f = nmt.NmtField(mask, None, spin=2)
    # Binning scheme is irrelevant for us, but NaMaster needs one.
    b = man.get_nmt_bins()

    # Alright, compute and reshape coupling matrix.
    # Compute also the binned MCM (this is actually what we need).
    print("Computing MCM")
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)
    mcm = np.transpose(w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
                       axes=[1, 0, 3, 2])

    # Save to file
    if os.path.isabs(o.mcm_dir):
        # absolute path
        mcm_dir = o.mcm_dir
    else:
        # Relative path, relative to mbatch job ouput directory
        # which is the parent directory of current script output
        # directory.
        mcm_dir = os.path.join(o.output_dir, '..', o.mcm_dir)
    # Create mcm_dir if not exists
    os.makedirs(mcm_dir, exist_ok=True)
    fname_out = os.path.join(mcm_dir, 'mcm.npz')
    print(f"Saving to {fname_out}")
    np.savez(fname_out, mcm=mcm)

    if o.plot:
        import matplotlib.pyplot as plt

        print("Plotting")
        plt.figure()
        plt.imshow(mcm.reshape([nspec*nl, nspec*nl]))
        plt.colorbar()
        fname = os.path.join(man.get_filenam("mcm_plots", o.output_dir),
                             'mcm.pdf')
        plt.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mode-Coupling Matrix (MCM) calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--plot", action='store_true',
                        help='Pass to generate a plot of the MCM.')
    parser.add_argument("--mcm-dir", type=str,
                        help='Directory to store computed MCM.')
    o = parser.parse_args()
    mcmer(o)
