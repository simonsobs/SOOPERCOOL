import argparse
from soopercool import BBmeta, utils
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt


def pre_processer(args):
    """
    """
    meta = BBmeta(args.globals)

    # Create beams for each map set
    meta.timer.start("Generating beams")
    utils.get_beam_windows(meta)
    meta.timer.stop("Generating beams")

    # Create bandpower edges / binning_file
    bin_low, bin_high, bin_center = utils.create_binning(meta.nside,
                                                         meta.deltal)
    np.savez(meta.path_to_binning, bin_low=bin_low, bin_high=bin_high,
             bin_center=bin_center)

    if args.plots:
        plt.figure(figsize=(8, 6))
        for i in range(len(bin_low)):
            plt.fill_between([bin_low[i], bin_high[i]], [0, 0], [i+1, i+1],
                             alpha=0.5)
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel('bin index', fontsize=14)
        plt.title('Binning', fontsize=14)
        plt.savefig(meta.path_to_binning.replace('.npz', '.png'))

    lmax_sim = 3*meta.nside - 1

    meta.timer.start("Computing fiducial cls")

    # Create the CMB fiducial cl
    lth, clth = utils.get_theory_cls(
        meta.cosmology,
        lmax=lmax_sim  # ensure cl accuracy up to lmax
    )
    cl_cosmo_fname = meta.save_fiducial_cl(lth, clth, cl_type="cosmo")

    if args.plots:
        for field_pair in ["TT", "TE", "TB", "EE", "EB", "BB"]:
            plt.figure(figsize=(8, 6))
            dlth = lth*(lth + 1) / 2. / np.pi * clth[field_pair]
            plt.loglog(lth, dlth, c='b')
            plt.loglog(lth[dlth < 0], -dlth[dlth < 0], ls='--', c='b')
            plt.xlabel(r'$\ell$', fontsize=14)
            plt.ylabel(r'$\ell(\ell+1)\, C_\ell/(2\pi)$', fontsize=14)
            plt.title(f'cosmo_cls_{field_pair}')
            plt.savefig(cl_cosmo_fname.replace('.npz', f'_{field_pair}.png'))

    # Create the fiducial power law spectra
    # Let's start with the power law used to compute
    # the transfer function
    cl_power_law_tf_estimation = utils.power_law_cl(
        lth, **meta.power_law_pars_tf_est)
    cl_tf_est_fname = meta.save_fiducial_cl(lth, cl_power_law_tf_estimation,
                                            cl_type="tf_est")

    if args.plots:
        for field_pair in ["TT", "TE", "TB", "EE", "EB", "BB"]:
            plt.figure(figsize=(8, 6))
            dlth = cl_power_law_tf_estimation[field_pair]
            plt.loglog(lth, dlth, c='b')
            plt.loglog(lth[dlth < 0], -dlth[dlth < 0], ls='--', c='b')
            plt.xlabel(r'$\ell$', fontsize=14)
            plt.ylabel(r'$C_\ell$', fontsize=14)
            plt.title(f'tf_estimation_{field_pair}')
            plt.savefig(cl_tf_est_fname.replace('.npz',
                                                f'_{field_pair}.png'))

    # Then the power law used to validate the
    # transfer function
    cl_power_law_tf_validation = utils.power_law_cl(
        lth, **meta.power_law_pars_tf_val)
    cl_tf_val_fname = meta.save_fiducial_cl(lth, cl_power_law_tf_validation,
                                            cl_type="tf_val")
    if args.plots:
        for field_pair in ["TT", "TE", "TB", "EE", "EB", "BB"]:
            plt.figure(figsize=(8, 6))
            dlth = cl_power_law_tf_validation[field_pair]
            plt.loglog(lth, dlth, c='b')
            plt.loglog(lth[dlth < 0], -dlth[dlth < 0], ls='--', c='b')
            plt.xlabel(r'$\ell$', fontsize=14)
            plt.ylabel(r'$C_\ell$', fontsize=14)
            plt.title(f'tf_validation_{field_pair}')
            plt.savefig(cl_tf_val_fname.replace('.npz',
                                                f'_{field_pair}.png'))
    meta.timer.stop("Computing fiducial cls")

    if args.sims:
        # Now we iterate over the number of simulations
        # to generate maps for each of them
        hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

        filtering_tags = meta.get_filtering_tags()
        beams = []
        for ftag in filtering_tags:
            which_beam = meta.tags_settings[ftag]["beam"]
            if which_beam is not None:
                _, bl = meta.read_beam(which_beam)
            else:
                bl = None
            beams.append((ftag, bl))

        meta.timer.start("Generating simulations for transfer")
        for id_sim in range(meta.tf_est_num_sims):
            meta.timer.start("Generate `cosmo` and `power_law` "
                             f"simulation n° {id_sim:04d}")
            for cl_type in ["cosmo", "tf_est", "tf_val"]:
                out_dir = getattr(meta, f"{cl_type}_sims_dir")
                os.makedirs(out_dir, exist_ok=True)

                cls = meta.load_fiducial_cl(cl_type=cl_type)
                alms_T, alms_E, alms_B = hp.synalm(
                    [cls[k] for k in hp_ordering],
                    lmax=lmax_sim,
                )

                if cl_type == "tf_est":
                    for ftag, bl in beams:
                            
                        sim_pureE = utils.generate_map_from_alms([alms_T, alms_E, alms_B], meta.nside, pureE=True, bl=bl)
                        sim_pureB = utils.generate_map_from_alms([alms_T, alms_E, alms_B], meta.nside, pureB=True, bl=bl)

                        map_file_pureE = meta.get_map_filename_transfer(
                            id_sim, cl_type, pure_type="pureE",
                            filter_tag=ftag
                        )
                        map_file_pureB = meta.get_map_filename_transfer(
                            id_sim, cl_type, pure_type="pureB",
                            filter_tag=ftag
                        )

                        hp.write_map(map_file_pureE, sim_pureE, overwrite=True,
                                     dtype=np.float32)
                        hp.write_map(map_file_pureB, sim_pureB, overwrite=True,
                                     dtype=np.float32)

                        if args.plots:
                            fname = map_file.replace('.fits', '')
                            title = map_file.split('/')[-1].replace('.fits',
                                                                        '')
                            amp = meta.power_law_pars_tf_est['amp']
                            delta_ell = meta.power_law_pars_tf_est['delta_ell']
                            pl_index = meta.power_law_pars_tf_est['power_law_index'] # noqa
                            ell0 = 0 if pl_index > 0 else 2 * meta.nside
                            var = amp / (ell0 + delta_ell)**pl_index
                            utils.plot_map(
                                sim, fname, vrange_T=10*var**0.5,
                                vrange_P=10*var**0.5, title=title,
                                TQU=True
                            )

                else:
                    for ftag, bl in beams:
                        
                        sim = utils.generate_map_from_alms([alms_T, alms_E, alms_B], meta.nside, bl=bl)
                        map_file = meta.get_map_filename_transfer(
                            id_sim, cl_type,
                            filter_tag=ftag
                        )
                        hp.write_map(map_file, sim, overwrite=True,
                                     dtype=np.float32)
                        if args.plots:
                            fname = map_file.replace('.fits', '')
                            title = map_file.split('/')[-1].replace('.fits', '')
                            if cl_type == "tf_val":
                                amp_T = meta.power_law_pars_tf_val['amp']['TT']
                                amp_E = meta.power_law_pars_tf_val['amp']['EE']
                                delta_ell = meta.power_law_pars_tf_val['delta_ell']
                                pl_index = meta.power_law_pars_tf_val['power_law_index'] # noqa
                                ell0 = 0 if pl_index > 0 else 2 * meta.nside
                                var_T = amp_T / (ell0 + delta_ell)**pl_index
                                var_P = amp_E / (ell0 + delta_ell)**pl_index
                                utils.plot_map(
                                    sim, fname, vrange_T=100*var_T**0.5,
                                    vrange_P=100*var_P**0.5,
                                    title=title, TQU=True
                                )
                            elif cl_type == "cosmo":
                                utils.plot_map(sim, fname, title=title, TQU=True)

            meta.timer.stop("Generate `cosmo` and `power_law` simulation "
                            f"n° {id_sim:04d}")
        meta.timer.stop("Generating simulations for transfer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing steps for the pipeline")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--sims", action="store_true",
                        help="Pass to generate the simulations used to compute the transfer function")  # noqa
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    pre_processer(args)
