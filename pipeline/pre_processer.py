import argparse
from bbmaster import BBmeta, utils
import numpy as np
import os
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm

def pre_processer(args):
    """
    """
    # TO DO
    # 1. Pre-process the binning (bpw_edges)
    # 2. Pre-process the noise level for simulations
    # 3. Pre-process the beams
    # 4. Pre-process the power spectra used for TF estimation and validation ?

    meta = BBmeta(args.globals)
    os.makedirs(meta.pre_process_directory, exist_ok=True)


    # First step : create bandpower edges / binning_file
    bin_low, bin_high, bin_center = utils.create_binning(meta.lmin, meta.lmax, meta.deltal)
    np.savez(meta.path_to_binning, bin_low=bin_low, bin_high=bin_high, bin_center=bin_center)

    # Second step is to create the survey mask from a hitmap
    os.makedirs(meta.mask_directory, exist_ok=True)
    hitmap = meta.read_hitmap()
    binary_mask = (hitmap > 0.).astype(float)
    meta.save_mask("binary", binary_mask, overwrite=True)

    if args.plots:
        cmap = cm.YlOrRd
        cmap.set_under("w")
        plt.figure(figsize=(16,9))
        hp.mollview(binary_mask, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(meta.binary_mask_name.replace('.fits', '.png'))

    # Then create the fiducial cl
    lth, psth = utils.theory_cls(
        meta.cosmology,
        lmax=meta.lmax
    )
    meta.save_fiducial_cl(lth, psth, cl_type="cosmo_cls")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-processing steps for the pipeline")
    parser.add_argument("--globals", type=str, help="Path to the yaml with global parameters")
    #parser.add_argument("--output_dir", type=str, help="Path to save the data")
    parser.add_argument("--plots", action="store_true", help="Pass to generate plots")

    args = parser.parse_args()
    pre_processer(args)
