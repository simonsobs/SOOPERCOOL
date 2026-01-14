from soopercool import BBmeta
from soopercool import mpi_utils as mpi
from itertools import product
import numpy as np
import argparse
import os


def main(args):
    """
    This script is used to coadd the cross-bundle power spectra
    (e.g. SAT1_f093__0 x SAT_f093__1) into cross-map-set power
    spectra (e.g. SAT1_f093 x SAT_f093). It will produce both
    cross and auto map-set power spectra from which we derive
    the noise power spectra.
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory
    cells_dir = {}
    map_types = ["signal", "noise", "coadd"]
    for typ in map_types:
        cells_dir[typ] = f"{out_dir}/cells_sims/{typ}"

    nmt_bins = meta.read_nmt_binning()
    lmax = nmt_bins.lmax

    lb = nmt_bins.get_effective_ells()
    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]
    nsims = meta.covariance["cov_num_sims"]

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False)
    }

    cross_bundle_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    if do_plots:
        import healpy as hp
        import matplotlib.pyplot as plt

        mask = lb < meta.lmax
        field_pairs_theory = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}
        colors = {"cross": "navy", "auto": "darkorange", "noise": "r"}
        mst = {"cross": "x", "auto": "o", "noise": "v"}

        plot_dir = f"{out_dir}/plots/cells_sims"
        BBmeta.make_dir(plot_dir)

        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]

    # Load bundle C_ells
    cells_dict_sims = {
        type: {
            (clt, ms1, ms2): {
                fp: [] for fp in field_pairs
            } for clt, (ms1, ms2) in product(map_types, cross_map_set_list)
        } for type in ["cross", "auto", "noise"]
    }

    # MPI parallelization
    rank, size, comm = mpi.init(True)
    mpi_shared_list = [(c, j)
                       for (c, j) in product(map_types,
                                             [i for i in range(nsims)])]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for clt, id_sim in local_mpi_list:

        # Initialize temporary dictionary
        cells_coadd = {
            type: {
                (ms1, ms2): {
                    fp: [] for fp in field_pairs
                } for ms1, ms2 in cross_map_set_list
            } for type in ["cross", "auto"]
        }

        # Loop over all map set pairs
        for map_name1, map_name2 in cross_bundle_list:
            if verbose:
                print(f"# {id_sim} | {clt} | {map_name1} x {map_name2}")

            map_set1, _ = map_name1.split("__")
            map_set2, _ = map_name2.split("__")

            cells_dict = np.load(
                f"{cells_dir[clt]}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz"  # noqa
            )

            if (map_name1, map_name2) in ps_names["cross"]:
                type = "cross"
            elif (map_name1, map_name2) in ps_names["auto"]:
                type = "auto"

            for field_pair in field_pairs:

                cells_coadd[type][map_set1, map_set2][field_pair] += [
                    cells_dict[field_pair]
                ]

        # Average the cross-bundle power spectra
        cells_coadd["noise"] = {}
        for map_set1, map_set2 in cross_map_set_list:
            cells_coadd["noise"][(map_set1, map_set2)] = {}
            for field_pair in field_pairs:
                for type in ["cross", "auto"]:
                    cells_coadd[type][map_set1, map_set2][field_pair] =\
                        np.mean(
                            cells_coadd[type][map_set1, map_set2][field_pair],
                            axis=0
                        )

                cells_coadd["noise"][map_set1, map_set2][field_pair] = \
                    cells_coadd["auto"][map_set1, map_set2][field_pair] -\
                    cells_coadd["cross"][map_set1, map_set2][field_pair]

            for type in ["cross", "auto", "noise"]:
                cells_to_save = {
                    fp: cells_coadd[type][map_set1, map_set2][fp]
                    for fp in field_pairs
                }
                for fp in field_pairs:
                    cells_dict_sims[type][clt, map_set1, map_set2][fp] += [
                        np.array(cells_to_save[fp]).squeeze()
                    ]
                np.savez(
                    f"{cells_dir[clt]}/decoupled_{type}_pcls_{map_set1}_x_{map_set2}_{id_sim:04d}.npz",  # noqa
                    lb=lb,
                    **cells_to_save
                )

    # Plot mean and standard deviation over simulations
    if do_plots:
        conv = 1e12 if args.units_K else 1
        types = ["cross", "auto", "noise"]
        if rank == 0:
            if size > 1:
                for i in range(1, size):
                    cells_dict = comm.recv(source=i, tag=11)
                    for clt, type, (m1, m2), fp in product(map_types,
                                                           types,
                                                           cross_map_set_list,
                                                           field_pairs):
                        cells_dict_sims[type][clt, m1, m2][fp] += [
                            np.array(
                                cells_dict[type][clt, m1, m2][fp]
                            ).squeeze()
                        ]
            cells_dict_std = {
                type: {
                    (clt, m1, m2): {
                        fp: np.std(
                                np.array(
                                    cells_dict_sims[type][(clt, m1, m2)][fp],
                                    dtype=float
                                ), axis=0
                            )
                        for fp in field_pairs
                    } for clt, (m1, m2) in product(map_types,
                                                   cross_map_set_list)
                } for type in types
            }
            cells_dict_mean = {
                type: {
                    (clt, m1, m2): {
                        fp: np.mean(
                                np.array(
                                    cells_dict_sims[type][(clt, m1, m2)][fp],
                                    dtype=float
                                ), axis=0
                            )
                        for fp in field_pairs
                    } for clt, (m1, m2) in product(map_types,
                                                   cross_map_set_list)
                } for type in types
            }
        else:
            comm.send(cells_dict_sims, dest=0, tag=11)

        if rank != 0:
            return

        for map_set1, map_set2 in cross_map_set_list:
            nu1 = meta.freq_tag_from_map_set(map_set1)
            nu2 = meta.freq_tag_from_map_set(map_set2)

            clb_th = None
            if fiducial_cmb:
                cmb_cl = hp.read_cl(fiducial_cmb)[:, :lmax+1]
                napp = nmt_bins.lmax + 1 - cmb_cl.shape[1]
                if napp > 0:
                    cmb_cl = np.concatenate(
                        [cmb_cl, np.zeros((cmb_cl.shape[0], napp))], axis=1
                    )
                cmb_clb = nmt_bins.bin_cell(cmb_cl)[:, mask]
                clb_th = cmb_clb
            if fiducial_dust:
                if not os.path.isfile(fiducial_dust.format(nu1=nu1, nu2=nu2)):
                    nu1, nu2 = nu2, nu1
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu1, nu2=nu2)
                )[:, :lmax+1]
                napp = nmt_bins.lmax + 1 - dust_cl.shape[1]
                if napp > 0:
                    dust_cl = np.concatenate(
                        [dust_cl, np.zeros((dust_cl.shape[0], napp))], axis=1
                    )
                dust_clb = nmt_bins.bin_cell(dust_cl)[:, mask]
                if clb_th is not None:
                    clb_th += dust_clb
                else:
                    clb_th = dust_clb
            if fiducial_synch:
                if not os.path.isfile(fiducial_dust.format(nu1=nu1, nu2=nu2)):
                    nu1, nu2 = nu2, nu1
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu1, nu2=nu2)
                )[:, :lmax+1]
                napp = nmt_bins.lmax + 1 - synch_cl.shape[1]
                if napp > 0:
                    synch_cl = np.concatenate(
                        [synch_cl, np.zeros((synch_cl.shape[0], napp))], axis=1
                    )
                synch_clb = nmt_bins.bin_cell(synch_cl)[:, mask]
                if clb_th is not None:
                    clb_th += synch_clb
                else:
                    clb_th = synch_clb
            beam1, beam2 = (
                nmt_bins.bin_cell(meta.read_beam(ms)[1][:lmax+1])[mask]
                for ms in [map_set1, map_set2]
            )

            for fp in field_pairs:
                if fp in field_pairs_theory and clb_th is not None:
                    ifp = field_pairs_theory[fp]
                    f, (main, sub) = plt.subplots(
                        2, 1, gridspec_kw={'height_ratios': [3, 1]},
                        figsize=(10, 9), sharex=True
                    )
                    y = clb_th[ifp] * beam1 * beam2
                    res = cells_dict_mean["cross"][("coadd",
                                                    map_set1,
                                                    map_set2)][fp][mask] - y
                    res /= cells_dict_std["cross"][("coadd",
                                                    map_set1,
                                                    map_set2)][fp][mask]
                    res *= np.sqrt(nsims)
                    sub.set_xlabel(r"$\ell$", fontsize=15)
                    sub.set_ylabel("residual", fontsize=12)
                    sub.axhspan(-3, 3, facecolor="grey", alpha=0.2)
                    sub.axhspan(-2, 2, facecolor="grey", alpha=0.2)
                    sub.axhline(0, color="k", linestyle="--")
                    sub.plot(lb[mask], res, c=colors["cross"])
                    main.plot(lb[mask], y, c="k", ls="--",
                              label="Theory")
                    sub.tick_params(
                        axis='x', which="both", bottom=True, top=True,
                        labeltop=False, direction="in"
                    )
                    sub.set_ylim(-5, 5)
                    main.set_xlim(0, meta.lmax)
                for typ in types:
                    main.errorbar(
                        lb,
                        conv*cells_dict_mean[typ][("coadd", map_set1,
                                                   map_set2)][fp],
                        yerr=conv*cells_dict_std[typ][("coadd", map_set1,
                                                       map_set2)][fp],
                        label=typ, marker=mst[typ], lw=0.7,
                        c=colors[typ]
                    )
                else:
                    f, main = plt.subplots(1, 1, figsize=(10, 8))
                    main.set_xlabel(r"$\ell$", fontsize=15)
                    main.set_xlim(0, meta.lmax)

                if fp == fp[::-1]:
                    main.set_yscale("log")
                    if fp == "TT":
                        main.set_ylim(1e-4, 1e7)
                    elif fp == "EE":
                        main.set_ylim(1e-8, 1e1)
                    elif fp == "BB":
                        main.set_ylim(1e-10, 1e1)
                elif fp in ["TE", "ET"]:
                    main.set_yscale("log")
                    main.set_ylim(1e-8, 1e3)
                else:
                    main.axhline(0, color="k", linestyle="--")
                    if fp in ["EB", "BE"]:
                        main.set_ylim(-0.01, 0.01)
                    else:
                        main.set_ylim(-4, 4)
                main.set_ylabel(
                    r"$C_\ell^\mathrm{%s} \; [\mu K_\mathrm{CMB}^2]$" % fp,
                    fontsize=15
                )
                main.tick_params(
                    axis='x', which="both", bottom=True, top=True,
                    labeltop=False, direction="in"
                )
                main.legend(fontsize=14)
                f.suptitle(f"{map_set1} x {map_set2} | {fp}", fontsize=15,
                           y=0.93)
                plt.subplots_adjust(hspace=0)
                plt.savefig(
                    f"{plot_dir}/pcls_{map_set1}_{map_set2}_{fp}.png",
                    bbox_inches="tight"
                )
                plt.close('all')
                if args.verbose:
                    print(" PLOT "
                          f"{plot_dir}/pcls_{map_set1}_{map_set2}_{fp}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coadd the cross-bundle power spectra for simulations"
    )
    parser.add_argument(
        "--globals",
        help="Path to the soopercool parameter file"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true", help="Do not make any plots"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode."
    )
    parser.add_argument(
        "--units_K", action="store_true",
        help="For plotting only. Assume Cls are in K^2, otherwise muK^2."
    )
    args = parser.parse_args()

    main(args)
