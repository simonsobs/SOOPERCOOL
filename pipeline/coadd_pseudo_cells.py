from soopercool import BBmeta
from itertools import product
import numpy as np
import argparse
import pymaster as nmt


def main(args):
    """
    This script is used to coadd the cross-split power spectra
    (e.g. SAT1_f093__0 x SAT_f093__1) into cross-map-set power
    spectra (e.g. SAT1_f093 x SAT_f093). It will produce both
    cross and auto map-set power spectra from which we derive
    the noise power spectra.
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells"

    plot_dir = f"{out_dir}/plots/cells"
    BBmeta.make_dir(plot_dir)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    lb = nmt_bins.get_effective_ells()
    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    if do_plots:
        import healpy as hp
        import matplotlib.pyplot as plt

        lmax = 3*meta.nside - 1
        ll = np.arange(lmax + 1)
        field_pairs_theory = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
        colors = {"cross": "navy", "auto": "darkorange", "noise": "r"}

        plot_dir = f"{out_dir}/plots/cells"
        BBmeta.make_dir(plot_dir)

        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False)
    }

    cross_split_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    # Load split C_ells

    # Initialize output dictionary
    cells_coadd = {
        "cross": {
            (ms1, ms2): {
                fp: [] for fp in field_pairs
            } for ms1, ms2 in cross_map_set_list
        },
        "auto": {
            (ms1, ms2): {
                fp: [] for fp in field_pairs
            } for ms1, ms2 in cross_map_set_list
        }
    }

    # Loop over all map set pairs
    for map_name1, map_name2 in cross_split_list:

        map_set1, _ = map_name1.split("__")
        map_set2, _ = map_name2.split("__")

        cells_dict = np.load(
            f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}.npz"
        )

        if (map_name1, map_name2) in ps_names["cross"]:
            type = "cross"
        elif (map_name1, map_name2) in ps_names["auto"]:
            type = "auto"

        for field_pair in field_pairs:

            cells_coadd[type][map_set1, map_set2][field_pair] += [cells_dict[field_pair]] # noqa

    # Average the cross-split power spectra
    cells_coadd["noise"] = {}
    for map_set1, map_set2 in cross_map_set_list:
        cells_coadd["noise"][(map_set1, map_set2)] = {}
        for field_pair in field_pairs:
            for type in ["cross", "auto"]:
                if len(cells_coadd[type][map_set1, map_set2][field_pair]) != 0:
                    cells_coadd[type][map_set1, map_set2][field_pair] = \
                        np.mean(
                            cells_coadd[type][map_set1, map_set2][field_pair],
                            axis=0
                        )

            if len(cells_coadd["auto"][map_set1, map_set2][field_pair]) == 0:
                cells_coadd["noise"][map_set1, map_set2][field_pair] = np.zeros_like(cells_coadd["cross"][map_set1, map_set2][field_pair])  # noqa
                cells_coadd["auto"][map_set1, map_set2][field_pair] = np.zeros_like(cells_coadd["cross"][map_set1, map_set2][field_pair])  # noqa
            else:
                cells_coadd["noise"][(map_set1, map_set2)][field_pair] = \
                    cells_coadd["auto"][map_set1, map_set2][field_pair] - \
                    cells_coadd["cross"][map_set1, map_set2][field_pair]

        for type in ["cross", "auto", "noise"]:
            cells_to_save = {
                fp: cells_coadd[type][map_set1, map_set2][fp]
                for fp in field_pairs
            }
            np.savez(
                f"{cells_dir}/decoupled_{type}_pcls_{map_set1}_x_{map_set2}.npz",  # noqa
                lb=lb,
                **cells_to_save
            )

    if do_plots:
        for map_set1, map_set2 in cross_map_set_list:
            nu1 = meta.freq_tag_from_map_set(map_set1)
            nu2 = meta.freq_tag_from_map_set(map_set2)

            for fp in field_pairs:

                plt.figure(figsize=(10, 8))
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.ylabel(r"$C_\ell^\mathrm{%s} \; [\mu K_\mathrm{CMB}^2]$" % fp, # noqa
                           fontsize=15)

                for type in ["cross", "auto", "noise"]:
                    plt.plot(lb, cells_coadd[type][(map_set1, map_set2)][fp],
                             label=type,
                             marker="o", lw=0.7, mfc="w", c=colors[type])
                if fp in field_pairs_theory:
                    ifp = field_pairs_theory[fp]
                    cmb_cl = hp.read_cl(fiducial_cmb)[ifp, :lmax+1]
                    dust_cl = hp.read_cl(
                        fiducial_dust.format(nu1=nu1, nu2=nu2)
                    )[ifp, :lmax+1]
                    synch_cl = hp.read_cl(
                        fiducial_synch.format(nu1=nu1, nu2=nu2)
                    )[ifp, :lmax+1]
                    clth = cmb_cl + dust_cl + synch_cl

                    plt.plot(ll, clth, c="k", ls="--", label="Theory")

                plt.legend(fontsize=14)
                plt.title(f"{map_set1} x {map_set2} | {fp}", fontsize=15)

                plt.xlim(0, 2*meta.nside)

                if fp == fp[::-1]:
                    plt.yscale("log")
                    if fp == "TT":
                        plt.ylim(1e-2, 1e9)
                    elif fp in ["EE", "BB"]:
                        plt.ylim(1e-6, 1e3)

                else:
                    plt.axhline(0, color="k", linestyle="--")
                    if fp in ["EB", "BE"]:
                        plt.ylim(-0.01, 0.01)
                    else:
                        plt.ylim(-4, 4)

                plt.savefig(
                    f"{plot_dir}/pcls_{map_set1}_{map_set2}_{fp}.png",
                    bbox_inches="tight"
                )
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not make plots")

    args = parser.parse_args()

    main(args)
