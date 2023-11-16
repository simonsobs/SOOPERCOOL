import argparse
import numpy as np
import pymaster as nmt
import healpy as hp
import os
from bbmaster.utils import PipelineManager
from bbmaster import BBmeta


def mcmer(args):
    """
    Compute the mode coupling matrix
    from the masks defined in the global
    parameter file.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)

    coupling_dir = meta.coupling_directory

    # Read and degrade mask
    print("Reading mask")
    mask = meta.read_mask("analysis")

    # Create dummy NaMaster field
    field_spin0 = nmt.NmtField(mask, None, spin=0)
    field_spin2 = nmt.NmtField(mask, None, spin=2)
    
    # Binning scheme is irrelevant for us, but NaMaster needs one.
    nmt_bins = meta.read_nmt_binning()

    # Alright, compute and reshape coupling matrix.
    # Compute also the binned MCM (this is actually what we need).
    print("Computing MCM")
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(field_spin0, field_spin2, nmt_bins, is_teb=True)

    nl = meta.lmax
    nspec = 7
    mcm = np.transpose(w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
                       axes=[1, 0, 3, 2])

    # Save to file
    fname = f"{coupling_dir}/mcm.npz"
    np.savez(fname, mcm=mcm, spin0xspin0=mcm[0,:,0,:], spin0xspin2=mcm[1:3,:,1:3,:], spin2xspin2=mcm[3:,:,3:,:])

    if args.plot:
        import matplotlib.pyplot as plt

        print("Plotting")
        plt.figure()
        plt.imshow(mcm.reshape([nspec*nl, nspec*nl]))
        plt.colorbar()
        fname = f"{coupling_dir}/mcm.png"
        plt.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCM calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--plot", action='store_true',
                        help='Pass to generate a plot of the MCM.')
    args = parser.parse_args()

    mcmer(args)
