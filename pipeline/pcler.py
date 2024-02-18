import argparse
import healpy as hp
import numpy as np
from soopercool import BBmeta
import pymaster as nmt
from itertools import product
import warnings


def get_coupled_pseudo_cls(fields1, fields2, nmt_binning):
    """
    Compute the binned coupled pseudo-C_ell estimates from two
    (spin-0 or spin-2) NaMaster fields and a multipole binning scheme.
    Parameters
    ----------
    fields1, fields2 : NmtField
        Spin-0 or spin-2 fields to correlate.
    nmt_binning : NmtBin
        Multipole binning scheme.
    """
    spins = list(fields1.keys())

    pcls = {}
    for spin1 in spins:
        for spin2 in spins:

            f1 = fields1[spin1]
            f2 = fields2[spin2]

            coupled_cell = nmt.compute_coupled_cell(f1, f2)
            coupled_cell = coupled_cell[:, :nmt_binning.lmax+1]

            pcls[f"{spin1}x{spin2}"] = nmt_binning.bin_cell(coupled_cell)
    return pcls


def decouple_pseudo_cls(coupled_pseudo_cells, coupling_inv):
    """
    Decouples the coupled pseudo-C_ell estimators computed between two fields
    of spin 0 or 2. Returns decoupled binned power spectra labeled by field
    pairs (e.g. 'TT', 'TE', 'EE', 'EB', 'BB' etc.).
    Parameters
    ----------
    coupled_pseudo_cells : dict with keys f"spin{s1}xspin{s2}",
        items array-like. Coupled pseudo-C_ell estimators.
    coupling_inv : array-like
        Inverse binned bandpower coupling matrix.
    """
    decoupled_pcls = {}
    for spin_comb, coupled_pcl in coupled_pseudo_cells.items():
        n_bins = coupled_pcl.shape[-1]
        decoupled_pcl = coupling_inv[spin_comb] @ coupled_pcl.flatten()
        if spin_comb == "spin0xspin0":
            size = 1
        elif spin_comb in ["spin0xspin2", "spin2xspin0"]:
            size = 2
        elif spin_comb == "spin2xspin2":
            size = 4
        decoupled_pcl = decoupled_pcl.reshape((size, n_bins))

        decoupled_pcls[spin_comb] = decoupled_pcl

    decoupled_pcls = field_pairs_from_spins(decoupled_pcls)

    return decoupled_pcls


def field_pairs_from_spins(cls_in_dict):
    """
    Reorders power spectrum dictionary with a given input spin
    pair into pairs of output (pseudo-)scalar fields on the sky
    (T, E, or B).

    Parameters
    ----------
    cls_in_dict: dictionary
    """
    cls_out_dict = {}

    field_spin_mapping = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin0": ["ET", "BT"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }

    for spin_pair in cls_in_dict:
        for index, field_pair in enumerate(field_spin_mapping[spin_pair]):

            cls_out_dict[field_pair] = cls_in_dict[spin_pair][index]

    return cls_out_dict


def get_pcls_mat_transfer(fields, nmt_binning):
    """
    Compute coupled binned pseudo-C_ell estimates from
    pure-E and pure-B transfer function estimation simulations,
    and cast them into matrix shape.

    Parameters
    ----------
    fields: dictionary of NmtField objects (keys "pureE", "pureB")
    nmt_binning: NmtBin object
    """
    n_bins = nmt_binning.get_n_bands()
    pcls_mat_00 = np.zeros((1, 1, n_bins))
    pcls_mat_02 = np.zeros((2, 2, n_bins))
    pcls_mat_22 = np.zeros((4, 4, n_bins))

    index = 0
    cases = ["pureE", "pureB"]
    for index, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls = get_coupled_pseudo_cls(fields[pure_type1],
                                      fields[pure_type2],
                                      nmt_binning)
        pcls_mat_22[index] = pcls["spin2xspin2"]
        pcls_mat_02[cases.index(pure_type2)] = pcls["spin0xspin2"]

    pcls_mat_00[0] = pcls["spin0xspin0"]

    return {"spin0xspin0": pcls_mat_00,
            "spin0xspin2": pcls_mat_02,
            "spin2xspin2": pcls_mat_22}


def get_binned_cls(bp_win_dict, cls_dict_unbinned):
    """
    """
    nl = np.shape(list(bp_win_dict.values())[0])[-1]
    cls_dict_binned = {}

    for spin_comb in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat = bp_win_dict[f"bp_win_{spin_comb}"]
        if spin_comb == "spin0xspin0":
            cls_vec = np.array([cls_dict_unbinned["TT"][:nl]]).reshape(1, nl)
        elif spin_comb == "spin0xspin2":
            cls_vec = np.array([cls_dict_unbinned["TE"][:nl],
                                cls_dict_unbinned["TB"][:nl]])
        elif spin_comb == "spin2xspin2":
            cls_vec = np.array([cls_dict_unbinned["EE"][:nl],
                                cls_dict_unbinned["EB"][:nl],
                                cls_dict_unbinned["EB"][:nl],
                                cls_dict_unbinned["BB"][:nl]])

        cls_dict_binned[spin_comb] = np.einsum("ijkl,kl", bpw_mat, cls_vec)

    return field_pairs_from_spins(cls_dict_binned)


def get_validation_power_spectra(meta, id_sim, mask, nmt_binning,
                                 inv_couplings):
    """
    This function computes transfer validation power spectra given an
    input simulation ID, mask and binning scheme, and stores them to disk.
    """
    map_set_pairs = (meta.get_ps_names_list(type="all", coadd=True)
                     if meta.validate_beam else [(None, None)])
    filter_flags = (["filtered"] if meta.validate_beam
                    else ["filtered", "unfiltered"])

    for cl_type in ["tf_val", "cosmo"]:
        for filter_flag in filter_flags:
            for map_sets in map_set_pairs:
                map_files = [
                    meta.get_map_filename_transfer2(
                        id_sim, cl_type=cl_type, map_set=ms
                    ) for ms in map_sets
                ]

                if filter_flag == "filtered":
                    map_files = [mf.replace(".fits", "_filtered.fits")
                                 for mf in map_files]

                maps = [hp.read_map(m, field=[0, 1, 2])
                        for m in map_files]

                field = [{
                    "spin0": nmt.NmtField(mask, map[:1]),
                    "spin2": nmt.NmtField(mask, map[1:],
                                          purify_b=meta.tf_est_pure_B)
                } for map in maps]

                pcls = get_coupled_pseudo_cls(field[0], field[1], nmt_binning)

                if meta.validate_beam:
                    decoupled_pcls = decouple_pseudo_cls(
                        pcls, inv_couplings[map_sets[0], map_sets[1]]
                    )
                else:
                    decoupled_pcls = decouple_pseudo_cls(
                        pcls, inv_couplings[filter_flag]
                    )
                cl_prefix = f"pcls_{cl_type}_{id_sim:04d}"
                cl_suffix = (f"_{map_sets[0]}_{map_sets[1]}"
                             if meta.validate_beam else f"_{filter_flag}")
                cl_name = cl_prefix + cl_suffix

                np.savez(f"{meta.cell_transfer_directory}/{cl_name}.npz",
                         **decoupled_pcls)


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
                pcls = get_coupled_pseudo_cls(fields[map_set1, id_split1],
                                              fields[map_set2, id_split2],
                                              nmt_binning)

                decoupled_pcls = decouple_pseudo_cls(
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
            cls_theory_binned = get_binned_cls(bp_win, cls_theory_unbinned)
            nls_theory_unbinned = meta.load_fiducial_cl("noise")
            freq_bands = nls_theory_unbinned["freq_bands"]

            ch_idx = {}
            for i, f in enumerate(freq_bands):
                for map_set in meta.map_sets:
                    if f"f{str(int(f)).zfill(3)}" in map_set:
                        ch_idx[map_set] = i

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
                         * nls_theory_unbinned[spec][ch_idx[map_set_1]])
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
        for id_sim in range(meta.tf_est_num_sims):
            meta.timer.start("pcler_tf_est")
            fields = {"filtered": {}, "unfiltered": {}}
            for pure_type in ["pureE", "pureB"]:
                map_file = meta.get_map_filename_transfer2(
                    id_sim,
                    "tf_est",
                    pure_type=pure_type
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

                fields["unfiltered"][pure_type] = field
                fields["filtered"][pure_type] = field_filtered

            pcls_mat_filtered = get_pcls_mat_transfer(fields["filtered"],
                                                      nmt_binning)
            pcls_mat_unfiltered = get_pcls_mat_transfer(fields["unfiltered"],
                                                        nmt_binning)

            np.savez(f"{cl_dir}/pcls_mat_tf_est_filtered_{id_sim:04d}.npz",
                     **pcls_mat_filtered)
            np.savez(f"{cl_dir}/pcls_mat_tf_est_unfiltered_{id_sim:04d}.npz",
                     **pcls_mat_unfiltered)
            meta.timer.stop(
                "pcler_tf_est",
                text_to_output=f"Compute C_ell #{id_sim} for TF estimation",
                verbose=args.verbose
            )

    if args.tf_val:
        meta.timer.start("couplings_tf_val")
        inv_couplings = meta.get_inverse_couplings(beamed=meta.validate_beam)
        meta.timer.stop(
            "couplings_tf_val",
            text_to_output="Loading inverse coupling matrix for validation",
            verbose=args.verbose
        )

        for id_sim in range(meta.tf_est_num_sims):
            meta.timer.start("pcler_tf_val")

            get_validation_power_spectra(
                meta, id_sim, mask, nmt_binning, inv_couplings=inv_couplings
            )

            meta.timer.stop(
                "pcler_tf_val",
                text_to_output=f"Compute C_ell #{id_sim} for TF validation",
                verbose=args.verbose
            )


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
