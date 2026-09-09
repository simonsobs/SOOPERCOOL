import argparse
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt
from pixell import enmap
from soopercool import utils as su
from soopercool import coupling_utils as cu
from soopercool import fft_utils as sfft
from soopercool import map_utils as mu
from soopercool import ps_utils as pu
from soopercool import mpi_utils as mpi
from soopercool import sim_utils
from soopercool import BBmeta
from soopercool import fft_utils as sfft
from copy import deepcopy
from scipy.stats import chi2
from itertools import product
import os
import matplotlib.ticker as mticker


def main(args):
    """
    Generate noise-only CMB sims, white-noise sims, and power-law sims,
    apply a k-space filter, and compare the (TF*MCM)-decoupled spectra against
    theory. Compare to unfiltered MCM-decoupled spectra and decoupled spectra
    from E-mode-free simulations.

    Produces plots of EE and BB and their chi2 against simulations.
    Runs 12 tests in total (npols x nsim_flavors x n_filtered).
    Prints a warning message if any of the tests has a PTE above 0.95.

    Computes k-space transfer fucntion on the fly, from a (required) set of
    TF estimation simulations read from the directory provided by the yaml.
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    
    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()
    n_bins = nmt_bins.get_n_bands()
    lmax = meta.lmax
    lmax_sim = min(2*lmax, lmax+500)
    pol_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    tf_settings = meta.transfer_settings

    out_dir = meta.output_directory
    couplings_dir = f"{out_dir}/couplings"
    tf_dir = tf_settings["transfer_directory"]
    plot_dir = f"{out_dir}/plots/cells_tf_val"
    val_sims_dir = f"{out_dir}/tf_validation_sims"
    BBmeta.make_dir(plot_dir)
    BBmeta.make_dir(tf_dir)

    pcls_tf_est_dir = f"{out_dir}/cells_tf_est"
    BBmeta.make_dir(pcls_tf_est_dir)

    num_est_sims = tf_settings["tf_est_num_sims"]
    num_val_sims = tf_settings["tf_val_num_sims"]

    # MPI related initialization
    rank, size, comm = mpi.init(True)

    # Read mask
    mask = mu.read_map(
        meta.masks["analysis_mask"],
        pix_type=meta.pix_type,
        car_template=meta.car_template
    )
    mask_binary = mu.binarize_mask(mask)
    shape, wcs = meta.get_geometry()
    lmax_res = mu.lmax_from_map(mask, pix_type=meta.pix_type)

    # NOTE: this is a hardcoded filter for now.
    kspace_tag = "kx20"
    kspace_pars = {"dkx": 20., "dky":0., "type": "cosine"}

    ##############################
    # Transfer function estimation
    ##############################
    if rank == 0:
        print("Estimating k-space filter TF")

    if tf_settings["tf_est_beams_list"]:
        sim_dirs = {
            beam: tf_settings.unfiltered_map_dir[beam]
            for beam in tf_settings["tf_est_beam_list"]
        }
        sim_templates = {
            beam: tf_settings.unfiltered_map_template[beam]
            for beam in tf_settings["tf_est_beam_list"]
        }
        beams = {}
        for beam_label in tf_settings["tf_est_beams_list"]:
            _, bl = meta.read_beam(beam_label, lmax=lmax_sim)
            beams[beam_label] = bl
    else:
        if rank == 0:
            print("Using Gaussian beam of FWHM 30 arcmin and low pass at "
                  "ell=650")
        # The default beam is a 30-arcminute Gaussian beam bandlimited at
        # lmax=650
        beam = (su.bandlim_sine2(np.arange(lmax_sim+1), 650, 50) * 
                su.beam_gaussian(np.arange(lmax_sim+1), 30.*np.pi/180./60.))
        beams = {"fwhm30": beam}

        # Read TF estimation sims from disk
        sim_dirs = {
            "fwhm30": list(tf_settings["unfiltered_map_dir"].values())[0]}
        sim_templates = {
            "fwhm30": list(tf_settings["unfiltered_map_template"].values())[0]}

    # MPI: parallelize over sim IDs and TF estimation beams
    mpi_shared_list = [(id_sim, beam_label)
                       for id_sim in range(num_est_sims)
                       for beam_label in beams]
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, beam_label in local_mpi_list:
        if verbose:
            print(" TF estimation power spectra for id_sim "
                  f"{id_sim} | {beam_label}")

        # We add "nofilt" to make clear that the only filter is a kspace filter
        lab = f"nofilt_{beam_label}_{kspace_tag}"
        out_f = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{lab}_x_{lab}_filtered_{id_sim:04d}.npz"  # noqa: E501
        out_unf = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{lab}_x_{lab}_unfiltered_{id_sim:04d}.npz"  # noqa: E501
        if os.path.isfile(out_f) and os.path.isfile(out_unf):
            continue

        fields = {"filtered": {}, "unfiltered": {}}

        for pure_type in ["pureT", "pureE", "pureB"]:

            map_fn = sim_templates[beam_label].format(
                id_sim=id_sim, pure_type=pure_type
            )
            map_fn = f"{sim_dirs[beam_label]}/{map_fn}"
            map = mu.read_map(
                map_fn,
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
            )
            # k-space filter the sims in-place
            map_filtered = sfft.kspace_filter(
                map * mask_binary,
                pix_type=meta.pix_type,
                **kspace_pars
            )

            field_unfiltered = {
                "spin0": nmt.NmtField(
                    mask,
                    map[:1],
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax
                ),
                "spin2": nmt.NmtField(
                    mask,
                    map[1:],
                    purify_b=meta.pure_B,
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax
                )
            }
            field_filtered = {
                "spin0": nmt.NmtField(
                    mask,
                    map_filtered[:1],
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax
                ),
                "spin2": nmt.NmtField(
                    mask,
                    map_filtered[1:],
                    purify_b=meta.pure_B,
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax
                )
            }

            fields["unfiltered"][pure_type] = field_unfiltered
            fields["filtered"][pure_type] = field_filtered
        pcls_mat_filtered = pu.get_pcls_mat_transfer(
            fields["filtered"],
            nmt_bins,
            fields2=fields["filtered"],
            return_unbinned=False
        )
        pcls_mat_unfiltered = pu.get_pcls_mat_transfer(
            fields["unfiltered"],
            nmt_bins,
            fields2=fields["unfiltered"],
            return_unbinned=False
        )
        if id_sim == mpi_shared_list[0][0]:
            fplt = f"{plot_dir}/pcls_mat_tf_est_{lab}_x_{lab}_{id_sim:04d}.pdf"
            pu.plot_pcls_mat_transfer(
                pcls_mat_unfiltered, pcls_mat_filtered, lb, fplt,
                lmax=meta.lmax
            )
        np.savez(out_f, pcls_mat=pcls_mat_filtered)
        np.savez(out_unf, pcls_mat=pcls_mat_unfiltered)
    comm.barrier()

    # Compute TF and save to disk
    if rank == 0:
        # We skip cross-TFs for different beams for simplicity
        tf_pairs = [2*[(f"nofilt_{beam_label}", kspace_tag),]
                    for beam_label in beams]
        pcls_mat_dict = cu.read_pcls_matrices(
            pcls_tf_est_dir, tf_pairs,
            num_est_sims,
            tf_settings["sim_id_start"]
        )

        # Average the pseudo-cl matrices
        pcls_mat_filtered_mean = cu.average_pcls_matrices(
            pcls_mat_dict,
            tf_pairs,
            filtered=True
        )
        pcls_mat_unfiltered_mean = cu.average_pcls_matrices(
            pcls_mat_dict,
            tf_pairs,
            filtered=False
        )

        # Compute and save the transfer functions
        trans = cu.get_transfer_dict(
            pcls_mat_filtered_mean,
            pcls_mat_unfiltered_mean,
            pcls_mat_dict,
            tf_pairs
        )
        full_tf = {}
        for ftag1, ftag2 in tf_pairs:
            lab1 = f"{ftag1[0]}_{ftag1[1]}"
            lab2 = f"{ftag2[0]}_{ftag2[1]}"
            tf = trans[ftag1, ftag2]
            np.savez(
                f"{tf_dir}/transfer_function_{lab1}_x_{lab2}.npz",
                **tf
            )
            full_tf[ftag1, ftag2] = tf["full_tf"]

        plot_dir = f"{out_dir}/plots/transfer_functions"
        BBmeta.make_dir(plot_dir)

        for ftag1, ftag2 in tf_pairs:
            lab1 = f"{ftag1[0]}_{ftag1[1]}"
            lab2 = f"{ftag2[0]}_{ftag2[1]}"
            tf_dict = np.load(
                f"{tf_dir}/transfer_function_{lab1}_x_{lab2}.npz")

            su.plot_transfer_function(
                lb, tf_dict, meta.lmin, meta.lmax,
                pol_pairs,
                file_name=f"{plot_dir}/transfer_{lab1}_x_{lab2}.pdf"
            )
        print(f"Saved TF plots under {plot_dir}")
    comm.barrier()

    ####################
    # TF validation sims
    ####################

    # Make input Cls
    # TF validation beams. Choose Gaussian 30 arcmin beam if
    # "tf_val_beams_list" is None or empty
    beams = {"fwhm30": hp.gauss_beam(fwhm=30.*np.pi/180./60., lmax=lmax_sim)}

    if tf_settings["tf_val_beams_list"]:
        beams = {}
        for beam_label in tf_settings["tf_val_beams_list"]:
            _, bl = meta.read_beam(beam_label, lmax=lmax_sim)
            beams[beam_label] = bl

    lth = np.arange(lmax_sim+1)
    cl_types = [lab+suf
                for lab in ["cmb", "plaw", "noise"]
                for suf in ["", "_bonly"]]
    f_types = ["filtered", "unfiltered"]

    # Input power spectra. White noise is not beam-convolved.
    cls = {}
    for beam in beams:
        bl = beams[beam]
        cls["cmb", beam] = np.array([
            su.get_theory_cls(lmax=lmax_sim, verbose=False)[1][fp] * bl**2
            for fp in ["TT", "TE", "EE", "BB"]
        ])
        cls["cmb_bonly", beam] = deepcopy(cls["cmb", beam])
        cls["cmb_bonly", beam][:3] *= 0.
        cls["noise", beam] = np.ones(4*(lmax_sim+1),
                                     dtype=np.float64).reshape(4, -1)
        cls["plaw", beam] = np.array([
            su.power_law_cl(lth,
                            **tf_settings["power_law_pars_tf_est"])[fp] * bl**2
            for fp in ["TT", "TE", "EE", "BB"]
        ])
        cls["plaw_bonly", beam] = deepcopy(cls["plaw", beam])
        cls["plaw_bonly", beam][:3] *= 0.
        cls["noise_bonly", beam] = deepcopy(cls["noise", beam])
        cls["noise_bonly", beam][:3] *= 0.

    # Make validation input sims
    if rank == 0:
        print("Generating TF validation sims")
    if meta.pix_type == "car":
        map_temp = enmap.zeros((3,) + shape, wcs=wcs)
    else:
        map_temp = np.zeros((3,) + shape)

    # MPI: parallelize over sim IDs, input CL shapes, validation beams, and
    # (filtered, unfiltered)
    mpi_shared_list = [(id_sim, cl_type, beam, isfilt)
                       for id_sim in range(num_val_sims)
                       for cl_type, beam in cls
                       for isfilt in f_types]
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, cl_type, beam, isfilt in local_mpi_list:
        print(f"Maps | sim {id_sim+1}/{num_val_sims} | {cl_type} | {beam} | {isfilt}")  # noqa: E501
        np.random.seed(id_sim+1000)
        alms = hp.synalm(cls[cl_type, beam], lmax=lmax_sim)
        mapTQU = sim_utils.get_map_from_alms(alms, map_temp)
        if isfilt == "filtered":
            mapTQU = sfft.kspace_filter(
                mask_binary * mapTQU,
                pix_type=meta.pix_type,
                **kspace_pars
            )
        fn = f"{val_sims_dir}/mapTQU_{cl_type}_{beam}_{isfilt}_{id_sim:04}.fits"  # noqa: E501
        mu.write_map(fn, mapTQU, pix_type=meta.pix_type)
    comm.barrier()

    #############################
    # TF validation power spectra
    #############################
    if rank == 0:
        print("Computing TF validation power spectra")

    # Load MCMs, transfer functions and compute coupling matrices
    # This avoid saving all products to disk and save disk space.
    mcm = cu.read_mcm(
        f"{couplings_dir}/mcm.npz",
        full_mcm=True
    )

    def ftag_from_map_set(ms):
        return f"nofilt_{ms}"

    def kspace_tag_from_map_set(ms):
        return kspace_tag
    
    bpwins = {}
    icoup = {}
    lmin_tf = {}

    for beam in beams:
        # Load specific validation TFs, not the cross-mapset ones.
        transfer, transfer_std = cu.load_transfer_function(
            meta.transfer_settings["transfer_directory"],
            beam, beam,
            ftag_from_map_set,
            kspace_tag_from_map_set,
            nmt_bins,
            return_std=True
        )

        # TF range
        # We cut every low-ell bin whose BB->BB TF is measured at less than
        # 2 sigma. We also cut all bins centered below ell of 30.
        tf_zscore = transfer[-1, -1] / transfer_std[-1, -1]
        good = tf_zscore > 2.
        if np.any(~good):
            lmin_tf[beam] = max((lb[~good][-1] + lb[good][0])/2., 30)
        else:
            lmin_tf[beam] = max((lb[0])/2., 30)

        (bpwins[beam, "filtered"],
            icoup[beam, "filtered"]) = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=transfer,
            compute_Dl=meta.compute_Dl,
            beam=None  # np.outer(beams[beam][:lmax+1], beams[beam][:lmax+1])
        )
        (bpwins[beam, "unfiltered"],
        icoup[beam, "unfiltered"]) = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=None,
            compute_Dl=meta.compute_Dl,
            beam=None  # np.outer(beams[beam][:lmax+1], beams[beam][:lmax+1])
        )

    cls_dict = {
        (cl_typ, beam): {
            fp: np.array([cl[0], cl[1], 0*cl[0], cl[1], 0*cl[0],
                          cl[2], 0*cl[2], 0*cl[2], cl[3]], dtype=np.float64)[i]
            for i, fp in enumerate(pol_pairs)
        }
        for (cl_typ, beam), cl in cls.items()
    }
    clth = {
        f"{cl_type}_{beam}_{isfilt}": pu.bin_theory_cls(
            cls_dict[cl_type, beam], bpwins[beam, isfilt]
        )
        for cl_type in cl_types
        for beam in beams
        for isfilt in f_types
    }
    kwargs = {"wcs": wcs, "lmax": meta.lmax, "lmax_mask": meta.lmax}

    for id_sim, cl_type, beam, isfilt in local_mpi_list:
        print(f"PCLs | sim {id_sim+1}/{num_val_sims} | {cl_type} | {beam} | {isfilt}")  # noqa: E501
        kwargs_map = {"pix_type": meta.pix_type, "fields_hp": (0, 1, 2)}
        map = mu.read_map(
            f"{val_sims_dir}/mapTQU_{cl_type}_{beam}_{isfilt}_{id_sim:04}.fits", **kwargs_map)  # noqa: E501

        # Compute decoupled power spectra
        # NOTE: we don't calculate the purified and non-purified version, 
        # just the one that is indicated in the config. Wwe don't loop over
        # the map sets, only over the sim types.
        field =  {
            "spin0": nmt.NmtField(mask, map[:1], **kwargs),
            "spin2": nmt.NmtField(
                mask,
                map[1:], purify_b=meta.pure_B, **kwargs)
        }
        pcls = pu.get_coupled_pseudo_cls(field, field, nmt_bins)
        clbs = pu.decouple_pseudo_cls(
            pcls, icoup[beam, isfilt].reshape([n_bins*9, n_bins*9]))

        np.savez_compressed(
            f"{out_dir}/clb_{cl_type}_{beam}_{isfilt}_{id_sim:04}.npz", cl=clbs
        )
    comm.barrier()
    if rank != 0:
        return

    clb = {
        f"{cl_type}_{beam}_{isfilt}_{pols}": []
        for cl_type in cl_types
        for beam in beams
        for isfilt in f_types
        for pols in ["EE", "BB"]
    }

    for cl_type, beam in product(cl_types, beams):
        for isfilt in f_types:
            for id_sim in range(num_val_sims):
                for pols in ["EE", "BB"]:
                    clab = f"{cl_type}_{beam}_{isfilt}"
                    clb[clab+f"_{pols}"] += [
                        np.load(
                            f"{out_dir}/clb_{clab}_{id_sim:04}.npz",
                            allow_pickle=True
                        )["cl"].item()[pols]
                    ]
    clb = {case: np.array(cb) for case, cb in clb.items()}

    #####################
    # TF validation plots
    #####################
    if rank == 0:
        print("Making plots")

    plot_dir = f"{out_dir}/plots/cells_tf_val_kspace"
    BBmeta.make_dir(plot_dir)

    pte_log = open(f"{plot_dir}/pte.log", "w")

    failed_count = 0

    # Loop over input simulation configutations
    for cl_type, beam in product(["cmb", "plaw", "noise"], beams):
        for pols in ["EE", "BB"]:
            fig, (main, sub, std) = plt.subplots(
                3, 1, sharex=True, height_ratios=(3, 2, 1), figsize=(5, 8))
            icls = {"EE": 2, "BB": 3}[pols]
            msk = np.logical_and(lth < lmax, lth > 2)
            for ax in [main, sub, std]:
                ax.axvspan(xmin=lb[0]/2., xmax=lmin_tf[beam],
                           color="k", alpha=0.2)
                ax.axvspan(xmin=lmax_res, xmax=lb[-1], color="k", alpha=0.2)
            lb_msk = np.logical_and(lb < lmax_res, lb > lmin_tf[beam])
            main.plot(lth[msk], cls[cl_type, beam][icls, msk],
                      label="Theory", c="k")
            cases = {"BB": ["", "_bonly"], "EE": [""]}[pols]

            # Loop over polarizations and filtered vs unfiltered sims
            for case, isfilt in product(cases, f_types):
                clab = f"{cl_type}{case}_{beam}_{isfilt}_{pols}"
                clab_th = f"{cl_type}{case}_{beam}_{isfilt}"
                c = {"_bonly": "navy", "": "r"}[case]
                ls = {"unfiltered": "-", "filtered": "--"}[isfilt]
                m = {"unfiltered": ".", "filtered": "x"}[isfilt]
                off = {"": -2, "_bonly": 2}[case]
                chisq = np.sum(((np.mean(clb[clab][:, lb_msk], axis=0) - \
                                 clth[clab_th][pols][lb_msk])**2)/np.var(clb[clab][:, lb_msk], axis=0))  # noqa: E501
                pte = chi2.sf(chisq, df=sum(lb_msk))
                if pte < 0.05:
                        failed_count += 1
                        print(f"FAILED: {cl_type}{case}_{beam} {isfilt} {pols} (PTE {pte:.1e})")  # noqa: E501
                        pte_log.write(f"FAILED: {cl_type}{case}_{beam} {isfilt} {pols} (PTE {pte:.1e})\n")  # noqa: E501
                caselab = {"": {True: "mask-purif.", False: "non-purif."}[meta.pure_B], "_bonly": "B-mode-only"}[case]  # noqa: E501
                caselab += ", " + {"filtered": "filt.", "unfiltered": "unfilt."}[isfilt]  # noqa: E501
                res = (np.mean(clb[clab], axis=0) - clth[clab_th][pols])/np.std(clb[clab], axis=0)*np.sqrt(num_val_sims)  # noqa: E501
                main.errorbar(
                    lb+off,
                    np.mean(clb[clab], axis=0),
                    yerr=np.std(clb[clab], axis=0),
                    label=fr"{caselab}: $\chi^2/n_{{\rm dof}}={{{chisq:.1f}}}/{{{sum(lb_msk)}}}$",  # noqa: E501
                    marker=m, ls="", c=c)
                sub.plot(
                    lb+off,
                    res,
                    c=c, ls="", marker=m)
                std.plot(
                    lb+off,
                    np.std(clb[clab], axis=0),
                    c=c, ls=ls)

                # Suplot y range
                ymin = -5
                ymax = 5
                sub.set_ylim((ymin, ymax))

                # Mark points whose central value is outside the visible range
                for xi, yi in zip(lb, res):
                    if yi > ymax:
                        sub.annotate(
                            f'{yi:.1f}',
                            xy=(xi, ymax),          # arrow tip at top boundary
                            xytext=(xi, ymax - 1),  # text inside plot
                            ha='center',
                            va='top',
                            color=c,
                            fontsize=8,
                            arrowprops=dict(
                                arrowstyle='->',
                                lw=1.5,
                                ls=ls,
                                color=c,
                            )
                        )
                    elif yi < ymin:
                        sub.annotate(
                            f'{yi:.1f}',
                            xy=(xi, ymin),
                            xytext=(xi, ymin + 1),
                            ha='center',
                            va='bottom',
                            color=c,
                            fontsize=8,
                            arrowprops=dict(
                                arrowstyle='->',
                                lw=1.5,
                                ls=ls,
                                color=c,
                            )
                        )
                main.plot(lb, clth[clab_th][pols], ls=ls, c=c)
            main.set_title(
                f"{cl_type} {pols} ({num_val_sims} sims, beam {beam})")
            sub.axhline(0, color="k")
            sub.set_ylabel("Bias/error on mean")
            std.plot([], [], "k-", label="unfiltered")
            std.plot([], [], "k--", label="filtered")
            main.set_yscale("log")
            main.set_ylabel(fr"$C_\ell^{{{pols}}}$")
            std.set_xlabel(r"$\ell$")
            std.set_ylabel("Error on mean")
            std.set_yscale("log")
            std.set_xlim((0, meta.lmax))
            main.legend(frameon=False)
            std.legend(frameon=False)

            # Show units of error on 1 experiment as twin y axis
            for iax, ax1 in enumerate([sub, std]):
                ax2 = ax1.twinx()
                ax2.set_ylabel(["Bias/error", "Error"][iax])
                ax2.set_ylim(ax1.get_ylim())

                # apply a function formatter
                formatter = mticker.FuncFormatter(
                    lambda x, pos: '{:.2e}'.format(x/np.sqrt(num_val_sims)))
                ax2.yaxis.set_major_formatter(formatter)

            fig.align_ylabels([main, sub, std])
            plt.savefig(
                f"{plot_dir}/TF_validation_{cl_type}_{beam}_{pols}.pdf",
                bbox_inches="tight")
            plt.close()

    pte_log.close()
    print(f"TF validation plots saved under {plot_dir}")
    if failed_count == 0:
        print("All tests passed.")
    else:
        print(f"WARNING: {failed_count} out of 18 tests failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer function validation"
    )
    parser.add_argument(
        "--globals",
        help='Path to yaml with global parameters'
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
