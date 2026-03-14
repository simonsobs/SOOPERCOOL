import argparse
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
from pixell import enmap
from soopercool import utils as su
from soopercool import map_utils as mu
from soopercool import ps_utils as pu
from soopercool import mpi_utils as mpi
from soopercool import sim_utils
from copy import deepcopy
from scipy.stats import chi2


def main(args):
    """
    Validate mask against power spectrum bias on mask-purified simulations.
    Simulation types include CMB-like (Planck cosmology with AL=1, r=0) and a
    30 arcmin beam, white-noise-like, and power-law (alpha=-2).

    Produces plots of EE and BB and their chi2 against simulations.
    Prints a warning message if any of the tests has a PTE above 0.95.

    Parameters
    ----------
    out_dir: Output directory used for plots and temporary files
    mask_fname: File name of the mask to be tested (in CAR or HEALPix format)
    binning_fname: Path to (soopercool) binning file in .npz format
    lmax: (Optional) maximum multipole to use. Defaults to 300.
    num_sims: (Optional) Number of sims to use. Defaults to 20.
    """
    out_dir = args.out_dir

    # MPI related initialization
    rank, size, comm = mpi.init(True)
    num_sims = args.num_sims

    # Read mask
    try:
        mask = enmap.read_map(args.mask_fname)
        shape, wcs = enmap.read_map_geometry(args.mask_fname)
    except:  # noqa
        mask = hp.read_map(args.mask_fname)
        shape, wcs = (mask.shape, None)
    pix_type = mu._get_pix_type(args.mask_fname)
    lmax_sim = mu.lmax_from_map(mask, pix_type=pix_type)

    #############
    # Simulations
    #############

    # Make input Cls
    fwhm_amin = 30.
    lmax = 300 if args.lmax is None else args.lmax
    bins = pu.read_nmt_binning(args.binning_fname, lmax, compute_Dl=False)
    lb = bins.get_effective_ells()

    if args.lmax is not None:
        lmax = args.lmax
    ls = np.arange(lmax_sim+1)
    cl_types = ["cmb", "plaw", "noise"]
    cls = {}
    cls["cmb"] = np.array([
        su.get_theory_cls(lmax=lmax_sim, fwhm_amin=fwhm_amin,
                          verbose=False)[1][fp]
        for fp in ["TT", "TE", "EE", "BB"]
    ])
    cls["cmb_bonly"] = deepcopy(cls["cmb"])
    cls["cmb_bonly"][:3] *= 0.
    cls["noise"] = np.ones(4*(lmax_sim+1), dtype=np.float64).reshape(4, -1)
    cls["noise_bonly"] = deepcopy(cls["noise"])
    cls["noise_bonly"][:3] *= 0.
    cls["plaw"] = (deepcopy(cls["noise"])*(1./(ls + 0.01)**2)[None, :] *
                   hp.gauss_beam(np.deg2rad(fwhm_amin/60.), lmax=lmax_sim)**2)
    cls["plaw_bonly"] = deepcopy(cls["plaw"])
    cls["plaw_bonly"][:3] *= 0.

    # Make input sims
    if rank == 0:
        print("Generating maps")
    if pix_type == "car":
        map_temp = enmap.zeros((3,) + shape, wcs=wcs)
    else:
        map_temp = np.zeros((3,) + shape)

    # MPI preparation
    mpi_shared_list = [(id_sim, cl_type)
                       for id_sim in range(num_sims) for cl_type in cls]
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, cl_type in local_mpi_list:
        print(f"Maps | sim {id_sim+1}/{num_sims} | {cl_type}")
        alms = hp.synalm(cls[cl_type], lmax=lmax_sim)
        mapTQU = sim_utils.get_map_from_alms(alms, map_temp)
        mapQU = mapTQU.copy()[1:]
        fn = f"{out_dir}/mapQU_{cl_type}_{id_sim:04}.fits"
        mu.write_map(fn, mapQU, pix_type=pix_type)
    comm.barrier()

    ###############
    # Power spectra
    ###############
    if rank == 0:
        print("Preparing NaMaster workspaces")
        mask = mu.read_map(args.mask_fname, pix_type=None,
                           geometry=(shape, wcs))
        f = nmt.NmtField(mask, None, wcs=wcs, spin=2, lmax=lmax)
        f_pure = nmt.NmtField(mask, None, spin=2, wcs=wcs, purify_b=True,
                              lmax=lmax)
        wsp = nmt.NmtWorkspace(f, f, bins)
        wsp_pure = nmt.NmtWorkspace(f_pure, f_pure, bins)
        wsp.write_to(f"{out_dir}/wsp_nopure.fits")
        wsp_pure.write_to(f"{out_dir}/wsp_pure.fits")
    comm.barrier()

    wsp = nmt.NmtWorkspace()
    wsp.read_from(f"{out_dir}/wsp_nopure.fits")
    wsp_pure = nmt.NmtWorkspace()
    wsp_pure.read_from(f"{out_dir}/wsp_pure.fits")
    cls22 = {cl_typ: np.array([cl[2], 0.*cl[2], 0.*cl[2], cl[3]], dtype=np.float64)  # noqa: E501
             for cl_typ, cl in cls.items()}
    clth = {f"{cl_type}_{case}": w.decouple_cell(w.couple_cell(cls22[cl_type]))
            for cl_type in cl_types
            for (case, w) in zip(["nopure", "pure"], [wsp, wsp_pure])}
    comm.barrier()

    if rank == 0:
        print("Computing PCLs")
    kwargs = {"wcs": wcs, "lmax": lmax, "lmax_mask": lmax}

    # MPI preparation
    mpi_shared_list = [(id_sim, cl_type)
                       for id_sim in range(num_sims)
                       for cl_type in cl_types]
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, cl_type in local_mpi_list:
        print(f"PCLs | sim {id_sim+1}/{num_sims} | {cl_type}")
        kwargs_map = {"pix_type": pix_type, "fields_hp": (0, 1)}
        map = mu.read_map(
            f"{out_dir}/mapQU_{cl_type}_{id_sim:04}.fits", **kwargs_map)
        map_bonly = mu.read_map(
            f"{out_dir}/mapQU_{cl_type}_bonly_{id_sim:04}.fits", **kwargs_map)
        f = nmt.NmtField(mask, map, purify_b=False, **kwargs)
        fpure = nmt.NmtField(mask, map, purify_b=True, **kwargs)
        fbonly = nmt.NmtField(mask, map_bonly, purify_b=False, **kwargs)
        cl = wsp.decouple_cell(nmt.compute_coupled_cell(f, f))
        clpure = wsp_pure.decouple_cell(nmt.compute_coupled_cell(fpure, fpure))
        clbonly = wsp.decouple_cell(nmt.compute_coupled_cell(fbonly, fbonly))
        np.savez_compressed(f"{out_dir}/clb_{cl_type}_{id_sim:04}.npz",
                            nopure=cl, pure=clpure, bonly=clbonly)
    comm.barrier()
    if rank != 0:
        return

    clb = {f"{cl_type}_{case}": []
           for cl_type in cl_types for case in ["nopure", "pure", "bonly"]}

    for cl_type in cl_types:
        for case in ["nopure", "pure", "bonly"]:
            for id_sim in range(num_sims):
                clb[f"{cl_type}_{case}"] += [
                    np.load(f"{out_dir}/clb_{cl_type}_{id_sim:04}.npz")[case]]
    clb = {case: np.array(cb) for case, cb in clb.items()}

    #######
    # Plots
    #######
    failed_count = 0
    for cl_type in ["cmb", "plaw", "noise"]:
        for pols in ["EE", "BB"]:
            _, (main, sub, std) = plt.subplots(
                3, 1, sharex=True, height_ratios=(3, 1, 2), figsize=(5, 8))
            icls = {"EE": 2, "BB": 3}[pols]
            msk = np.logical_and(ls < lmax, ls > 2)
            iclb = {"EE": 0, "BB": 3}[pols]
            main.plot(ls[msk], cls[cl_type][icls, msk], label="Input", c="k")
            cases = {"BB": ["nopure", "pure", "bonly"],
                     "EE": ["nopure", "pure"]}[pols]
            for case in cases:
                clab = f"{cl_type}_{case}"
                clab_th = f"{cl_type}_" + {"bonly": "pure",
                                           "nopure": "nopure",
                                           "pure": "pure"}[case]
                c = {"bonly": "navy",
                     "pure": "darkorange",
                     "nopure": "r"}[case]
                off = {"bonly": -2, "pure": 0, "nopure": 2}[case]
                chisq = np.sum(((np.mean(clb[clab], axis=0)[iclb] - clth[clab_th][iclb])**2)/np.var(clb[clab], axis=0)[iclb])  # noqa: E501
                pte = chi2.sf(chisq, df=len(lb))
                main.errorbar(
                    lb+off,
                    np.mean(clb[clab], axis=0)[iclb],
                    yerr=np.std(clb[clab], axis=0)[iclb],
                    label=fr"{case}: $\chi^2/n_{{\rm dof}}={{{chisq:.1f}}}/{{{len(lb)}}}$",  # noqa: E501
                    marker=".", ls="", c=c)
                sub.plot(
                    lb+off,
                    (np.mean(clb[clab], axis=0) - clth[clab_th])[iclb]/np.std(clb[clab], axis=0)[iclb]*np.sqrt(num_sims),  # noqa: E501
                    c=c)
                std.plot(
                    lb+off,
                    np.std(clb[clab], axis=0)[iclb],
                    c=c)
                main.plot(lb, clth[clab_th][iclb], ls="--", c=c)
                main.set_title(f"{cl_type} ({num_sims} simulations)")
                if pte < 0.05:
                    failed_count += 1
                    print(f"FAILED: {cl_type}_{case} (PTE {pte:.1e})")
            sub.axhline(0, color="k")
            sub.set_ylabel("Bias on mean")
            sub.set_ylim((-5, 5))
            main.set_yscale("log")
            main.set_ylabel(fr"$C_\ell^{{{pols}}}$")
            std.set_xlabel(r"$\ell$")
            std.set_ylabel("STD")
            std.set_yscale("log")
            main.legend()
            plt.savefig(f"{out_dir}/mask_validation_{cl_type}_{pols}.png",
                        bbox_inches="tight")
            print(f"{out_dir}/mask_validation_{cl_type}_{pols}.png")
            plt.close()
    if failed_count == 0:
        print("All tests passed.")
    else:
        print(f"WARNING: {failed_count} out of 6 tests failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--mask_fname",
                        type=str,
                        help="Filename of the mask .fits to test")
    parser.add_argument("--binning_fname",
                        type=str,
                        help="Filename of the soopercool binning .npz")
    parser.add_argument("--lmax",
                        type=int,
                        default=None,
                        help="Maximum multipole (optional)")
    parser.add_argument("--num_sims",
                        type=int,
                        default=20,
                        help="Number of simulations")
    args = parser.parse_args()

    main(args)
