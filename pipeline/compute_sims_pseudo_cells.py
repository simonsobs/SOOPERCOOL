from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from soopercool import mpi_utils as mpi
from itertools import product
import os
import argparse
import numpy as np
import pymaster as nmt
from pixell import enmap


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_sims"
    couplings_dir = f"{out_dir}/couplings"
    sims_dir = f"{out_dir}/cov_sims"
    nsims = meta.covariance["cov_num_sims"]

    BBmeta.make_dir(cells_dir)

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1,
                                     is_Dell=meta.compute_Dl)
    n_bins = nmt_bins.get_n_bands()
    ps_pairs = meta.get_ps_names_list(type="all", coadd=False)

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        inv_couplings_beamed[ms1, ms2] = np.load(f"{couplings_dir}/couplings_{ms1}_{ms2}.npz")["inv_coupling"].reshape([n_bins*9, n_bins*9])  # noqa

    decoupled_pcls = {
        ps: []
        for ps in meta.get_ps_names_list(type="all", coadd=False)
    }
    cls_dict_sims = {pn: {} for pn in ps_pairs}

    if do_plots:
        import healpy as hp
        import matplotlib.pyplot as plt

        lmax = 600  # meta.lmax
        lb = nmt_bins.get_effective_ells()
        lmax_bins = nmt_bins.get_ell_max(nmt_bins.get_n_bands() - 1)
        lb_msk = lb < lmax + 10
        cb2db = lb*(lb+1)/2./np.pi
        field_pairs = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}

        cl_plot_dir = f"{out_dir}/plots/cells_sims"
        map_plot_dir = f"{out_dir}/plots/maps"
        BBmeta.make_dir(cl_plot_dir)
        BBmeta.make_dir(map_plot_dir)

        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]

    rank, size, comm = mpi.init(True, logger=None)
    id_start = meta.covariance["cov_id_start"]
    nsims = meta.covariance["cov_num_sims"]

    # Initialize tasks for MPI sharing
    mpi_shared_list = [id_sim for id_sim in range(id_start, id_start+nsims)]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim in local_mpi_list:
        base_dir = f"{sims_dir}/{id_sim:04d}"

        # Create namaster fields
        fields = {}
        for map_name in meta.maps_list:
            map_set, id_bundle = map_name.split("__")
            map_fname = f"{base_dir}/cov_sims_{map_set}_bundle{id_bundle}.fits"

            if verbose:
                print(f"Namaster field for sim {id_sim+1} | "
                      f"({map_set}, bundle {id_bundle})")

            m = mu.read_map(map_fname,
                            pix_type=meta.pix_type,
                            fields_hp=[0, 1, 2],
                            car_template=meta.car_template,
                            convert_K_to_muK=False)  # Sim maps are in muK_CMB

            wcs = None
            if meta.pix_type == "car":
                # # This is a patch. Reproject mask and map onto template
                # geometry.
                tshape, twcs = enmap.read_map_geometry(meta.car_template)
                if twcs != m.wcs:
                    shape, wcs = enmap.overlap(m.shape, m.wcs, tshape, twcs)
                else:
                    shape, wcs = tshape, twcs
                if mask.wcs != wcs:
                    shape, wcs = enmap.overlap(mask.shape, mask.wcs,
                                               shape, wcs)
                if not (m.wcs == mask.wcs == twcs):
                    flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)
                    mask = enmap.insert(flat_template.copy()[0], mask)
                    m = enmap.insert(flat_template.copy(), m)
            field_spin0 = nmt.NmtField(mask, m[:1], wcs=wcs)
            field_spin2 = nmt.NmtField(mask, m[1:], wcs=wcs,
                                       purify_b=meta.pure_B)

            fields[map_set, id_bundle] = {
                "spin0": field_spin0,
                "spin2": field_spin2
            }

        for map_name1, map_name2 in ps_pairs:
            fname = f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz"  # noqa
            map_set1, id_bundle1 = map_name1.split("__")
            map_set2, id_bundle2 = map_name2.split("__")

            if verbose:
                print(f"Coupled C_ells for sim {id_sim+1} | "
                      f"({map_set1}, bundle {id_bundle1}) x "
                      f"({map_set2}, bundle {id_bundle2})")
            if os.path.isfile(fname) and args.no_overwrite:
                cls_dict = np.load(fname)
                for fp, cls in cls_dict.items():
                    if fp not in cls_dict_sims[(map_name1, map_name2)]:
                        cls_dict_sims[(map_name1, map_name2)][fp] = []
                    cls_dict_sims[(map_name1, map_name2)][fp] += [cls]
                continue

            pcls = pu.get_coupled_pseudo_cls(
                fields[map_set1, id_bundle1],
                fields[map_set2, id_bundle2],
                nmt_bins
            )
            if do_plots:
                plt.loglog(lb[lb_msk], pcls["spin2xspin2"][0][lb_msk],
                           label="EE")
                plt.legend()
                plt.savefig(
                    f"{cl_plot_dir}/pcl_EE_{map_name1}_x_{map_name2}.png"
                )
                print(f"{cl_plot_dir}/pcl_EE_{map_name1}_x_"
                      f"{map_name2}_{id_sim:04}.png")
                plt.close()

            decoupled_pcls = pu.decouple_pseudo_cls(
                    pcls, inv_couplings_beamed[map_set1, map_set2]
                    )
            for fp, cls in decoupled_pcls.items():
                if fp not in cls_dict_sims[(map_name1, map_name2)]:
                    cls_dict_sims[(map_name1, map_name2)][fp] = []
                cls_dict_sims[(map_name1, map_name2)][fp] += [np.array(cls).squeeze()]  # noqa

            np.savez(fname, **decoupled_pcls, lb=nmt_bins.get_effective_ells())

    # Plot mean and standard deviation over simulations
    if do_plots:
        conv = 1e12 if args.units_K else 1
        if rank == 0:
            if size > 1:
                for i in range(1, size):
                    cls_dict = comm.recv(source=i, tag=11)
                    for (m1, m2), fp in product(ps_pairs, field_pairs):
                        cls_dict_sims[(m1, m2)][fp] += [np.array(cls_dict[(m1, m2)][fp]).squeeze()]  # noqa
            cls_dict_std = {
                (m1, m2): {
                    fp: np.std(np.atleast_2d(cls_dict_sims[(m1, m2)][fp]), axis=0)  # noqa
                    for fp in field_pairs
                } for m1, m2 in ps_pairs
            }
            cls_dict_mean = {
                (m1, m2): {
                    fp: np.mean(np.atleast_2d(cls_dict_sims[(m1, m2)][fp]), axis=0)  # noqa
                    for fp in field_pairs
                } for m1, m2 in ps_pairs
            }
        else:
            comm.send(cls_dict_sims, dest=0, tag=11)

        if rank != 0:
            return
        for m1, m2 in ps_pairs:
            map_set1, id_bundle1 = m1.split("__")
            map_set2, id_bundle2 = m2.split("__")
            nu1 = meta.freq_tag_from_map_set(map_set1)
            nu2 = meta.freq_tag_from_map_set(map_set2)

            clb_th = None
            if fiducial_cmb:
                cmb_cl = hp.read_cl(fiducial_cmb)[:, :lmax_bins+1]
                cmb_clb = nmt_bins.bin_cell(cmb_cl)[:, lb_msk]
                clb_th = cmb_clb
            if fiducial_dust:
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu1, nu2=nu2)
                )[:, :lmax_bins+1]
                dust_clb = nmt_bins.bin_cell(dust_cl)[:, lb_msk]
                if clb_th:
                    clb_th += dust_clb
                else:
                    clb_th = dust_clb
            if fiducial_synch:
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu1, nu2=nu2)
                )[:, :lmax_bins+1]
                synch_clb = nmt_bins.bin_cell(synch_cl)[:, lb_msk]
                if clb_th:
                    clb_th += synch_clb
                else:
                    clb_th = synch_clb

            for i, fp in enumerate(field_pairs):
                plt.title(f"({map_set1}, bundle {id_bundle1}) "
                          f"x ({map_set2}, bundle {id_bundle2}) | {fp}",
                          fontsize=9)
                plt.errorbar(
                    lb[lb_msk],
                    conv*cb2db[lb_msk]*cls_dict_mean[(m1, m2)][fp][lb_msk],
                    yerr=conv*cb2db[lb_msk]*cls_dict_std[(m1, m2)][fp][lb_msk],
                    c="navy", marker=".", mfc="w", capsize=3, ls="",
                    label=f"{fp} ({nsims} sims)"
                )
                if clb_th is not None:
                    plt.plot(lb[lb_msk], cb2db[lb_msk]*clb_th[i],
                             c="k", ls="--", label="Theory")
                plt.yscale("log")
                plt.ylabel(
                    r"$D_\ell^\mathrm{%s} \; [\mu K_\mathrm{CMB}^2]$" % fp,
                    fontsize=15
                )
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.legend(fontsize=13)
                plt.savefig(f"{cl_plot_dir}/pcls_{map_set1}_"
                            f"bundle{id_bundle1}_{map_set2}_"
                            f"bundle{id_bundle2}_{fp}.png")
                plt.close()
        if args.verbose:
            print(f" PLOTS: {cl_plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--globals",
        help="Path to the global parameter file."
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Do not make plots."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode."
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite spectra if existing."
    )
    parser.add_argument(
        "--units_K", action="store_true",
        help="For plotting only. Assume Cls are in K^2, otherwise muK^2."
    )
    args = parser.parse_args()
    main(args)
