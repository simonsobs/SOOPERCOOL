import argparse
import healpy as hp
import numpy as np
import os
from soopercool import BBmeta
from soopercool.utils import get_noise_cls, theory_cls
from soopercool import ps_utils
import pymaster as nmt
import matplotlib.pyplot as plt
import warnings


def pcler(args):
    """
    Compute all decoupled binned pseudo-C_ell estimates needed for
    a specific task in the pipeline related to simulations or data.
    Runs in four modes: "tf_est", "tf_val", "data" and "sims".
    * "tf_est": simulations for the transfer function estimation
    * "tf_val": simulations for the transfer function validation
    * "data": real (or mock) data
    * "sims": simulations needed for the power spectrum covariance estimation
    Parameters
    ----------
    args: dictionary
        Global parameters and command line arguments.
    """
    meta = BBmeta(args.globals)
    mask = meta.read_mask("analysis")
    nmt_binning = meta.read_nmt_binning()
    n_bins = nmt_binning.get_n_bands()
    binary_mask = meta.read_mask("binary")
    fsky = np.mean(binary_mask)

    if args.tf_est or args.tf_val:
        cl_dir = meta.cell_transfer_directory
    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory
    if args.data or args.sims:
        # Set the number of sims to loop over
        Nsims = meta.num_sims if args.sims else 1

        # Load the inverse coupling matrix
        inv_couplings = {}
        for map_set1, map_set2 in meta.get_ps_names_list(type="all",
                                                         coadd=True):
            couplings = np.load(f"{meta.coupling_directory}/couplings_{map_set1}_{map_set2}.npz")  # noqa
            coupling_dict = {
                k1: couplings[f"inv_coupling_{k2}"].reshape([ncl*n_bins,
                                                             ncl*n_bins])
                for k1, k2, ncl in zip(["spin0xspin0", "spin0xspin2",
                                        "spin2xspin0", "spin2xspin2"],
                                       ["spin0xspin0", "spin0xspin2",
                                        "spin0xspin2", "spin2xspin2"],
                                       [1, 2, 2, 4])
            }
            inv_couplings[map_set1, map_set2] = coupling_dict
            if map_set1 != map_set2:
                # the only map set dependance is on the beam
                inv_couplings[map_set2, map_set1] = coupling_dict

        if args.plots:
            el = nmt_binning.get_effective_ells()
            cells_plots = {}

            for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                               coadd=False):
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                split_label = f"split{id_split1}_x_split{id_split2}"
                for pol1, pol2 in ['EE', 'EB', 'BB']:
                    plot_label = f"{map_set1}__{map_set2}__{pol1}{pol2}"
                    has_noise_bias = (id_split1 == id_split2) and \
                        (map_set1 == map_set2)
                    if plot_label not in cells_plots:
                        cells_plots[plot_label] = {}
                    if split_label not in cells_plots[plot_label]:
                        cells_plots[plot_label][split_label] = {}
                    clp = cells_plots[plot_label][split_label]
                    if has_noise_bias:
                        clp['noisy'] = np.zeros((Nsims, n_bins))
                    else:
                        clp['noiseless'] = np.zeros((Nsims, n_bins))

        for id_sim in range(Nsims):
            fields = {}
            # Load namaster fields that will be used
            for map_name in meta.maps_list:
                map_set, id_split = map_name.split("__")

                # Load maps
                map_file = meta.get_map_filename(
                    map_set, id_split, id_sim=id_sim if Nsims > 1 else None
                )
                map_file = map_file.replace(".fits", "_filtered.fits")

                map = hp.read_map(map_file, field=[0, 1, 2])

                # Include beam in namaster fields to deconvolve it
                field_spin0 = nmt.NmtField(mask, [map[0]])
                field_spin2 = nmt.NmtField(mask, [map[1], map[2]],
                                           purify_b=meta.pure_B)

                fields[map_set, id_split] = {
                    "spin0": field_spin0,
                    "spin2": field_spin2
                }

            for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                               coadd=False):
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                pcls = ps_utils.get_coupled_pseudo_cls(
                    fields[map_set1, id_split1],
                    fields[map_set2, id_split2],
                    nmt_binning
                )

                decoupled_pcls = ps_utils.decouple_pseudo_cls(
                    pcls, inv_couplings[map_set1, map_set2]
                )

                sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                np.savez(f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz",  # noqa
                         **decoupled_pcls, lb=nmt_binning.get_effective_ells())

                if args.plots:
                    split_label = f"split{id_split1}_x_split{id_split2}"
                    for field_pair in ['EE', 'EB', 'BB']:
                        plot_label = f"{map_set1}__{map_set2}__{field_pair}"
                        has_noise_bias = (id_split1 == id_split2) and \
                            (map_set1 == map_set2)

                        clp = cells_plots[plot_label][split_label]
                        if has_noise_bias:
                            clp['noisy'][id_sim] = \
                                decoupled_pcls[field_pair]
                        else:
                            clp['noiseless'][id_sim] = \
                                decoupled_pcls[field_pair]

        if args.plots:
            plot_dir_rel = meta.sims_directory_rel if Nsims > 1 \
                else meta.map_directory_rel
            plot_dir = meta.plot_dir_from_output_dir(plot_dir_rel)

            for plot_label in cells_plots:
                plt.clf()
                plt.figure(figsize=(6, 4))
                plt.title(plot_label)

                # Load theory CMB spectra
                map_set1, map_set2, field_pair = plot_label.split("__")
                el_th, cl_th_dict = theory_cls(meta.cosmology,
                                               meta.lmax, lmin=meta.lmin)
                cl_th = cl_th_dict[field_pair]
                if meta.null_e_modes and "E" in field_pair:
                    cl_th = np.zeros_like(cl_th)
                do_theory_plot = True
                for split_label in cells_plots[plot_label]:
                    cells_splits = cells_plots[plot_label][split_label]

                    if do_theory_plot:
                        # Plot noiseless theory power spectra
                        plt.plot(el_th,
                                 el_th*(el_th+1)/2./np.pi*(cl_th),
                                 lw=1, ls='--', c='darkred',
                                 label="theory (noiseless)")
                        if 'noisy' in cells_splits:
                            # Plot noisy theory with beam-deconvolved noise
                            _, nl_th_beam_dict = get_noise_cls(
                                fsky, meta.lmax, lmin=meta.lmin,
                                sensitivity_mode='baseline',
                                oof_mode='optimistic',
                                is_beam_deconvolved=True
                            )
                            freq_tag = meta.freq_tag_from_map_set(map_set1)
                            n_splits = meta.n_splits_from_map_set(map_set1)
                            nl_th_beam = nl_th_beam_dict["P"][freq_tag] *\
                                float(n_splits)
                            pre = el_th*(el_th+1)/(2*np.pi)
                            plt.plot(el_th, pre*(cl_th+nl_th_beam),
                                     lw=1, ls='-.', c='k',
                                     label="theory (noisy)")
                        do_theory_plot = False

                    # Plot splits power spectra
                    for has_noise_bias, cells_sims in cells_splits.items():
                        cells_sum = 0
                        for id_sim in range(Nsims):
                            cells = cells_sims[id_sim]
                            cells_sum += cells
                            color = 'k' if (has_noise_bias == "noisy") \
                                else 'tab:red'
                            if Nsims > 1:
                                plt.plot(el, el*(el+1)/2./np.pi*cells,
                                         lw=0.3, c=color, alpha=0.5)
                        plt.plot(el, el*(el+1)/2./np.pi*cells_sum/Nsims,
                                 lw=1, c=color, label=split_label)

                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel(r'$\ell(\ell+1)\,C_\ell/2\pi$', fontsize=14)
                    plt.xlabel(r'$\ell$', fontsize=14)
                    plt.legend(bbox_to_anchor=(1, 1))
                    plt.savefig(
                        os.path.join(
                            plot_dir,
                            f"cells_{map_set1}_{map_set2}_{field_pair}.png"
                        ), bbox_inches='tight'
                    )

    if args.tf_est:

        filtering_tags = meta.get_filtering_tags()
        filtering_tag_pairs = meta.get_independent_filtering_pairs()
        for id_sim in range(meta.tf_est_num_sims):

            fields = {ftag: {
                "filtered": {},
                "unfiltered": {}
                } for ftag in filtering_tags
            }

            for ftag in filtering_tags:
                for pure_type in ["pureE", "pureB"]:
                    map_file = meta.get_map_filename_transfer(
                        id_sim,
                        "tf_est",
                        pure_type=pure_type,
                        filter_tag=ftag
                    )
                    map_file_filtered = map_file.replace(".fits", "_filtered.fits")

                    map = hp.read_map(map_file, field=[0, 1, 2])
                    map_filtered = hp.read_map(map_file_filtered, field=[0, 1, 2])

                    # TO-DO: filter temperature only once !
                    field = {
                        "spin0": nmt.NmtField(mask, map[:1]),
                        "spin2": nmt.NmtField(mask, map[1:],
                                              purify_b=meta.tf_est_pure_B)
                    }
                    field_filtered = {
                        "spin0": nmt.NmtField(mask, map_filtered[:1]),
                        "spin2": nmt.NmtField(mask, map_filtered[1:],
                                              purify_b=meta.tf_est_pure_B)
                    }

                    fields[ftag]["unfiltered"][pure_type] = field
                    fields[ftag]["filtered"][pure_type] = field_filtered

            for ftag1, ftag2 in filtering_tag_pairs:
                pcls_mat_filtered = ps_utils.get_pcls_mat_transfer(
                    fields[ftag1]["filtered"],
                    nmt_binning, fields2=fields[ftag2]["filtered"]
                )
                pcls_mat_unfiltered = ps_utils.get_pcls_mat_transfer(
                    fields[ftag1]["unfiltered"],
                    nmt_binning, fields2=fields[ftag2]["unfiltered"]
                )

                np.savez(f"{cl_dir}/pcls_mat_tf_est_{ftag1}x{ftag2}_filtered_{id_sim:04d}.npz", # noqa
                         **pcls_mat_filtered)
                np.savez(f"{cl_dir}/pcls_mat_tf_est_{ftag1}x{ftag2}_unfiltered_{id_sim:04d}.npz", # noqa
                         **pcls_mat_unfiltered)

    if args.tf_val:
        filtering_tag_pairs = meta.get_independent_filtering_pairs()
        filtering_tags = meta.get_filtering_tags()

        inv_couplings = {}
        pure_str = "_pure" if meta.tf_est_pure_B else ""

        for ftype in ["filtered", "unfiltered"]:

            inv_couplings[ftype] = {}

            for ftag1, ftag2 in filtering_tag_pairs:

                cross_name = f"{ftag1}x{ftag2}"
                couplings = np.load(f"{meta.coupling_directory}/couplings_{cross_name}_{ftype}{pure_str}.npz")  # noqa
                inv_couplings[ftype][ftag1, ftag2] = {
                    k1: couplings[f"inv_coupling_{k2}"].reshape([ncl*n_bins,
                                                                ncl*n_bins])
                    for k1, k2, ncl in zip(["spin0xspin0", "spin0xspin2",
                                            "spin2xspin0", "spin2xspin2"],
                                            ["spin0xspin0", "spin0xspin2",
                                            "spin0xspin2", "spin2xspin2"],
                                        [1, 2, 2, 4])
                }

        for id_sim in range(meta.tf_est_num_sims):
            for cl_type in ["tf_val", "cosmo"]:
                for ftype in ["filtered", "unfiltered"]:
                    fields = {}
                    for ftag in filtering_tags:
                        map_file = meta.get_map_filename_transfer(
                            id_sim, cl_type=cl_type,
                            filter_tag=ftag
                        )
                        if ftype == "filtered":
                            map_file = map_file.replace(".fits", "_filtered.fits")

                        map = hp.read_map(map_file, field=[0, 1, 2])
                        
                        fields[ftag] = {
                            "spin0": nmt.NmtField(mask, map[:1]),
                            "spin2": nmt.NmtField(mask, map[1:],
                                                  purify_b=meta.tf_est_pure_B)
                        }

                    for ftag1, ftag2 in filtering_tag_pairs:

                        pcls = ps_utils.get_coupled_pseudo_cls(
                            fields[ftag1], fields[ftag2],
                            nmt_binning
                        )

                        decoupled_pcls = ps_utils.decouple_pseudo_cls(
                            pcls, inv_couplings[ftype][ftag1, ftag2]
                        )

                        print(decoupled_pcls["TT"])

                    np.savez(f"{cl_dir}/pcls_{cl_type}_{ftag1}x{ftag2}_{id_sim:04d}_{ftype}.npz",  # noqa
                             **decoupled_pcls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated power spectra if True.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--tf_est", action="store_true")
    mode.add_argument("--tf_val", action="store_true")

    args = parser.parse_args()

    if (args.tf_est and args.plots) or (args.tf_val and args.plots):
        warnings.warn("Both --tf_[...] and --plots are set to True. "
                      "This is not implemented yet. Set --plot to False")
        args.plots = False

    pcler(args)
