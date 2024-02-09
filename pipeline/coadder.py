import argparse
import numpy as np
import os
from soopercool import BBmeta
from soopercool.utils import theory_cls, get_noise_cls
import matplotlib.pyplot as plt


def coadder(args):
    """
    """
    meta = BBmeta(args.globals)
    nmt_binning = meta.read_nmt_binning()
    n_bins = nmt_binning.get_n_bands()
    field_pairs = ["EE", "EB", "BB"]  # TODO: ADD T*, *T and BE
    binary_mask = meta.read_mask("binary")
    fsky = np.mean(binary_mask)

    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory

    # Set the number of sims to loop over
    Nsims = meta.num_sims if args.sims else 1

    # Set the type of split-pair-averaged power spectra to generate.
    types = ["cross"]
    if args.all:
        types.append("all")
    if args.auto:
        types.append("auto")

    for type in types:
        if args.plots:
            cells_plots = []

        # Load split C_ells
        for id_sim in range(Nsims):
            cells_save = {}
            for map_name1, map_name2 in meta.get_ps_names_list(type=type,
                                                               coadd=False):
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                n_pairs = meta.get_n_split_pairs_from_map_sets(
                    map_set1, map_set2, type=type
                )
                sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                cells_dict = np.load(
                    f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz"  # noqa
                )
                for id_field_pair, field_pair in enumerate(field_pairs):
                    cells_label = f"{map_set1}__{map_set2}__{field_pair}"
                    if cells_label not in cells_save:
                        cells_save[cells_label] = np.zeros(n_bins)
                    cells_save[cells_label] += cells_dict[field_pair]/n_pairs

            for cells_label in cells_save:
                np.savez(
                    f"{cl_dir}/decoupled_{type}_pcls_nobeam_{cells_label}{sim_label}.npz",  # noqa
                    clb=cells_save[cells_label],
                    lb=nmt_binning.get_effective_ells()
                )
            if args.plots:
                cells_plots.append(cells_save)

        if args.plots:
            plot_dir_rel = meta.sims_directory_rel if Nsims > 1 \
                else meta.map_directory_rel
            plot_dir = meta.plot_dir_from_output_dir(plot_dir_rel)
            el = nmt_binning.get_effective_ells()

            for cells_label in cells_plots[0]:
                # Load and plot theory CMB spectra
                plt.figure(figsize=(6, 4))
                plt.title(cells_label)
                map_set1, map_set2, field_pair = cells_label.split("__")
                el_th, cl_th_dict = theory_cls(meta.cosmology,
                                               meta.lmax, lmin=meta.lmin)
                cl_th = cl_th_dict[field_pair]
                if meta.null_e_modes and "E" in field_pair:
                    cl_th = np.zeros_like(cl_th)
                plt.plot(el_th,
                         el_th*(el_th+1)/2./np.pi*(cl_th),
                         lw=1, ls='--', c='darkred',
                         label="theory (noiseless)")

                if type == "auto" or type == "all":
                    # Plot noisy theory with beam-deconvolved noise
                    _, nl_th_beam_dict = get_noise_cls(
                        fsky, meta.lmax, lmin=meta.lmin,
                        sensitivity_mode='baseline',
                        oof_mode='optimistic',
                        is_beam_deconvolved=True
                    )
                    freq_tag = meta.freq_tag_from_map_set(map_set1)
                    n_splits = meta.n_splits_from_map_set(map_set1)
                    nl_th_beam = nl_th_beam_dict["P"][freq_tag]*float(n_splits)
                    plt.plot(el_th,
                             el_th*(el_th+1)/2./np.pi*(cl_th+nl_th_beam),
                             lw=1, ls='-.', c='k',
                             label="theory (noisy)")

                # Plot split-pair-averaged power spectra
                cells_sum = 0
                for id_sim in range(Nsims):
                    cells = cells_plots[id_sim][cells_label]
                    cells_sum += cells
                    if Nsims > 1:
                        plt.plot(el, el*(el+1)/2./np.pi*cells,
                                 lw=0.3, c='tab:red', alpha=0.5)
                plt.plot(el, el*(el+1)/2./np.pi*cells_sum/Nsims,
                         lw=1, c='blue', label=cells_label)

                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel(r'$\ell(\ell+1)\,C_\ell/2\pi$', fontsize=14)
                plt.xlabel(r'$\ell$', fontsize=14)
                plt.legend(bbox_to_anchor=(1, 1))
                plt.savefig(
                    os.path.join(
                        plot_dir,
                        f"cells_{type}_{cells_label}.png"
                    ), bbox_inches='tight'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--auto", action="store_true",
                        help="Save auto-split power spectra.")
    parser.add_argument(
        "--all", action="store_true",
        help="Save coadded auto- and cross-split power spectra.")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated power spectra if True.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")

    args = parser.parse_args()

    coadder(args)
