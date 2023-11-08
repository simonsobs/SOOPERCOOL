import argparse
from bbmaster import BBmeta, utils
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm

def pre_processer(args):
    """
    """
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

    # Create the CMB fiducial cl
    lth, psth = utils.theory_cls(
        meta.cosmology,
        lmax= meta.lmax + 1000 # to ensure that the cl will be accurate up to lmax
    )
    meta.save_fiducial_cl(lth, psth, cl_type="cosmo")

    # Create the fiducial power law spectra
    ## Let's start with the power law used to compute the transfer function
    cl_power_law_tf_estimation = utils.power_law_cl(lth, **meta.power_law_pars_tf_est)
    meta.save_fiducial_cl(lth, cl_power_law_tf_estimation, cl_type="tf_est")

    ## Then the power law used to validate the transfer function
    cl_power_law_tf_validation = utils.power_law_cl(lth, **meta.power_law_pars_tf_val)
    meta.save_fiducial_cl(lth, cl_power_law_tf_validation, cl_type="tf_val")

    if args.sims:
        # Now we iterate over the number of simulations to generate maps for each of them
        hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]
        for id_sim in range(meta.tf_est_num_sims):

            for cl_type in ["cosmo", "tf_est", "tf_val"]:

                out_dir = getattr(meta, f"{cl_type}_sims_dir")
                os.makedirs(out_dir, exist_ok=True)

                cls = meta.load_fiducial_cl(cl_type=cl_type)
                alms_T, alms_E, alms_B = hp.synalm(
                    [cls[k] for k in hp_ordering],
                    lmax=3*meta.nside-1,
                )

                map_T = hp.alm2map(alms_T, meta.nside, lmax=3*meta.nside-1)

                if cl_type == "tf_est":
                    for case in ["pureE", "pureB"]:
                        
                        pol_alms_list = [
                            alms_E if case=="pureE" else np.zeros_like(alms_T),
                            alms_B if case=="pureB" else np.zeros_like(alms_T)
                        ]
                        map_Q, map_U = hp.alm2map_spin(pol_alms_list, meta.nside, spin=2, lmax=3*meta.nside-1)
                        
                        sim = np.array([map_T, map_Q, map_U])
                        
                        hp.write_map(f"{out_dir}/TQU_{case}_noiseless_nside{meta.nside}_lmax{meta.lmax}_{id_sim:04d}.fits", sim, overwrite=True)

                else:
                    map_Q, map_U = hp.alm2map_spin([alms_E, alms_B], meta.nside, spin=2, lmax=3*meta.nside-1)
                    sim = np.array([map_T, map_Q, map_U])
                    hp.write_map(f"{out_dir}/TQU_noiseless_nside{meta.nside}_lmax{meta.lmax}_{id_sim:04d}.fits", sim, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-processing steps for the pipeline")
    parser.add_argument("--globals", type=str, help="Path to the yaml with global parameters")
    #parser.add_argument("--output_dir", type=str, help="Path to save the data")
    parser.add_argument("--sims", action="store_true", help="Pass to generate the simulations used to compute the transfer function")
    parser.add_argument("--plots", action="store_true", help="Pass to generate plots")

    args = parser.parse_args()
    pre_processer(args)
