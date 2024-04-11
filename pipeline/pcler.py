import argparse
import healpy as hp
import numpy as np
from soopercool import BBmeta, ps_utils
import pymaster as nmt
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

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    if args.tf_est or args.tf_val:
        cl_dir = meta.cell_transfer_directory
    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory
    if args.data or args.sims:
        # Load the inverse coupling matrix
        meta.timer.start("couplings")
        inv_couplings_beamed = meta.get_inverse_couplings(beamed=True)
        meta.timer.stop(
            "couplings",
            text_to_output="Load inverse coupling matrix for beamed sims",
            verbose=args.verbose
        )
    if args.data or args.sims:
        # Set the number of sims to loop over
        Nsims = meta.num_sims if args.sims else 1

        for id_sim in range(Nsims):
            meta.timer.start("pcler")
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
                    pcls, inv_couplings_beamed[map_set1, map_set2]
                )

                sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                np.savez(f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz",  # noqa
                         **decoupled_pcls, lb=nmt_binning.get_effective_ells())
            meta.timer.stop(
                "pcler",
                text_to_output=f"Compute mock data C_ell #{id_sim}",
                verbose=args.verbose
            )

        if args.plots:
            meta.timer.start("plots")
            import matplotlib.pyplot as plt

            type = "sims" if args.sims else "data"
            lb = nmt_binning.get_effective_ells()
            nl = nmt_binning.lmax + 1
            ell = np.arange(nl)
            cls_dict = {}
            map_split_pairs_list = []

            for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                               coadd=False):
                m1, s1 = map_name1.split("__")
                m2, s2 = map_name2.split("__")
                split_pair = f"split{s1}_x_split{s2}"
                map_pair = f"{m1}__{m2}"
                map_split_pairs_list += [(map_pair, split_pair)]

                for fp in field_pairs:
                    cls_dict[map_pair, fp, split_pair] = []
                for id_sim in range(Nsims):
                    sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                    cls = np.load(f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz")  # noqa
                    for fp in field_pairs:
                        cls_dict[map_pair, fp, split_pair] += [cls[fp]]

            # Compute mean and std
            cls_mean_dict = {
                (map_pair, field_pair, split_pair):
                np.mean(cls_dict[map_pair, field_pair, split_pair], axis=0)
                for (map_pair, field_pair, split_pair) in cls_dict
            }
            cls_std_dict = {
                (map_pair, field_pair, split_pair):
                np.std(cls_dict[map_pair, field_pair, split_pair], axis=0)
                for (map_pair, field_pair, split_pair) in cls_dict
            }

            # Load and bin theory spectra
            bp_win = np.load(
                f"{meta.coupling_directory}/couplings_unfiltered.npz"
            )
            cls_theory_unbinned = meta.load_fiducial_cl("cosmo")
            cls_theory_binned = ps_utils.get_binned_cls(bp_win,
                                                        cls_theory_unbinned)
            nls_theory_unbinned = meta.load_fiducial_cl("noise")

            # Plot power spectra
            plot_dir_rel = getattr(meta, f"cell_{type}_directory_rel")
            plot_dir = meta.plot_dir_from_output_dir(plot_dir_rel)

            for map_pair, split_pair in map_split_pairs_list:
                plt.figure(figsize=(16, 16))
                grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)

                # Plot noise for auto-spectra only
                map_set_1, map_set_2 = map_pair.split("__")
                split_1, split_2 = split_pair.split("_x_")
                auto = map_set_1 == map_set_2 and split_1 == split_2

                for i1, i2 in [(i, j) for i in range(3) for j in range(3)]:
                    # Define subplots
                    main = plt.subplot(grid[3*i1:3*(i1 + 1) - 1, i2])
                    sub = plt.subplot(grid[3*(i1 + 1) - 1, i2])

                    f1, f2 = "TEB"[i1], "TEB"[i2]
                    spec = f2 + f1 if i1 > i2 else f1 + f2
                    no_e = 0. if meta.null_e_modes and "E" in spec else 1.
                    n_splits = meta.n_splits_from_map_set(map_set_1)

                    # Plot fiducial theory and noise spectra
                    cl_th = no_e * cls_theory_unbinned[spec]
                    nl_th = (
                        (n_splits
                         * nls_theory_unbinned[map_set_1][spec])
                        if auto else 0.
                    )

                    rescaling = ell*(ell + 1)/(2*np.pi)
                    main.plot(ell, rescaling*cl_th, "k-",
                              label="theory signal")
                    if auto:
                        main.plot(ell, rescaling*(nl_th + cl_th), "k:",
                                  label="theory isotropic noise")

                    # Plot split combinations (decoupled)
                    offset = 0.5
                    rescaling = lb*(lb + 1.)/(2*np.pi)

                    main.errorbar(
                        lb - offset,
                        rescaling*cls_mean_dict[map_pair, spec, split_pair],
                        rescaling*cls_std_dict[map_pair, spec, split_pair],
                        color="navy", marker=".", markerfacecolor="white",
                        label=f"{map_pair}\n{split_pair}", ls="None"
                    )

                    if f1 == f2:
                        main.set_yscale("log")

                    if type == "sims":
                        # Plot residuals
                        residual = ((cls_mean_dict[map_pair, spec, split_pair]
                                    - cls_theory_binned[spec])
                                    / cls_std_dict[map_pair, spec, split_pair])

                        sub.axhspan(-2, 2, color="gray", alpha=0.2)
                        sub.axhspan(-1, 1, color="gray", alpha=0.7)

                        sub.axhline(0, color="k")
                        sub.plot(lb - offset,
                                 residual * np.sqrt(Nsims),
                                 color="navy", marker=".",
                                 markerfacecolor="white", ls="None")

                    # Multipole range
                    main.set_xlim(2, meta.lmax)
                    sub.set_xlim(*main.get_xlim())

                    # Suplot y range
                    sub.set_ylim((-5., 5.))

                    # Cosmetix
                    main.set_title(f1 + f2, fontsize=14)
                    if spec == "TT":
                        main.legend(fontsize=13)
                    main.set_xticklabels([])
                    if i1 != 2:
                        sub.set_xticklabels([])
                    else:
                        sub.set_xlabel(r"$\ell$", fontsize=13)

                    if i2 == 0:
                        if isinstance(rescaling, float):
                            main.set_ylabel(r"$C_\ell$", fontsize=13)
                        else:
                            main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$",
                                            fontsize=13)
                        sub.set_ylabel(
                            r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$",  # noqa
                            fontsize=13
                        )

                plt.savefig(f"{plot_dir}/decoupled_pcls_nobeam_{type}_{map_pair}_{split_pair}.pdf", # noqa
                            bbox_inches="tight")
            meta.timer.stop(
                "plots",
                text_to_output="Plot decoupled C_ells for mock data",
                verbose=args.verbose
            )

    if args.tf_est:

        filtering_tags = meta.get_filtering_tags()
        filtering_tag_pairs = meta.get_independent_filtering_pairs()
        for id_sim in range(meta.tf_est_num_sims):
            meta.timer.start("pcler_tf_est")

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
                    map_file_filtered = map_file.replace(".fits",
                                                         "_filtered.fits")

                    map = hp.read_map(map_file, field=[0, 1, 2])
                    map_filtered = hp.read_map(map_file_filtered,
                                               field=[0, 1, 2])

                    # TO-DO: include pureT simulation type in the future
                    field = {
                        "spin0": nmt.NmtField(mask, map[:1]),
                        "spin2": nmt.NmtField(mask, map[1:],
                                              purify_b=meta.pure_B)
                    }
                    field_filtered = {
                        "spin0": nmt.NmtField(mask, map_filtered[:1]),
                        "spin2": nmt.NmtField(mask, map_filtered[1:],
                                              purify_b=meta.pure_B)
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
            meta.timer.stop(
                "pcler_tf_est",
                text_to_output=f"Compute C_ell #{id_sim} for TF estimation",
                verbose=args.verbose
            )

    if args.tf_val:
        filtering_tag_pairs = meta.get_independent_filtering_pairs()
        filtering_tags = meta.get_filtering_tags()
        meta.timer.start("couplings_tf_val")
        inv_couplings = {}

        for ftype in ["filtered", "unfiltered"]:

            inv_couplings[ftype] = {}

            for ftag1, ftag2 in filtering_tag_pairs:

                cross_name = f"{ftag1}x{ftag2}"
                couplings = np.load(f"{meta.coupling_directory}/couplings_{cross_name}_{ftype}.npz")  # noqa
                inv_couplings[ftype][ftag1, ftag2] = {
                    k1: couplings[f"inv_coupling_{k2}"].reshape([ncl*n_bins,
                                                                ncl*n_bins])
                    for k1, k2, ncl in zip(["spin0xspin0", "spin0xspin2",
                                            "spin2xspin0", "spin2xspin2"],
                                           ["spin0xspin0", "spin0xspin2",
                                            "spin0xspin2", "spin2xspin2"],
                                           [1, 2, 2, 4])
                }
        meta.timer.stop(
            "couplings_tf_val",
            text_to_output="Loading inverse coupling matrix for validation",
            verbose=args.verbose
        )
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
                            map_file = map_file.replace(".fits",
                                                        "_filtered.fits")

                        map = hp.read_map(map_file, field=[0, 1, 2])

                        fields[ftag] = {
                            "spin0": nmt.NmtField(mask, map[:1]),
                            "spin2": nmt.NmtField(mask, map[1:],
                                                  purify_b=meta.pure_B)
                        }

                    for ftag1, ftag2 in filtering_tag_pairs:

                        pcls = ps_utils.get_coupled_pseudo_cls(
                            fields[ftag1], fields[ftag2],
                            nmt_binning
                        )

                        decoupled_pcls = ps_utils.decouple_pseudo_cls(
                            pcls, inv_couplings[ftype][ftag1, ftag2]
                        )

                        np.savez(f"{cl_dir}/pcls_{cl_type}_{ftag1}x{ftag2}_{id_sim:04d}_{ftype}.npz",  # noqa
                                 **decoupled_pcls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated power spectra if True.")
    parser.add_argument("--verbose", action="store_true")

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
