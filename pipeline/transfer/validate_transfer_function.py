import argparse
import numpy as np
from soopercool import BBmeta
import matplotlib.pyplot as plt
from soopercool import ps_utils as pu
from soopercool import coupling_utils as cu
import soopercool.utils as su
import os


def main(args):
    """
    This script compares (TF*MCM)-decoupled power spectra from filtered
    simulations with MCM-decoupled power spectra from unfiltered simulations.

    The paths to read from must be given in the yaml under
    transfer['validation']. Both filtered and unfiltered sims must exist on
    disk. This supports any type of filtered simulations.

    If you don't have access to a set of validation simulations, consider
    running `validate_transfer_function_kspace.py` for a quick and simple,
    k-space-only validation. This might be useful e.g. to explore interactions
    between kspace filtering (and associated large-scale mode loss) and
    specific sky masking choices.
    """
    meta = BBmeta(args.globals)

    if "validation" not in meta.transfer_settings:
        raise KeyError(
            "SOOPERCOOL config yaml must point to existing TF validation"
            "sims under transfer['validation']."
        )

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()
    lb_msk = lb < meta.lmax
    cb2db = lb*(lb+1)/2/np.pi

    nsims = meta.transfer_settings["tf_val_num_sims"]

    out_dir = meta.output_directory
    couplings_dir = f"{out_dir}/couplings"
    plot_dir = f"{out_dir}/plots/cells_tf_val"
    BBmeta.make_dir(plot_dir)

    ps_pairs = meta.get_ps_names_list(type="all", coadd=True)
    fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    # Load MCMs, transfer functions and compute coupling matrices
    # This avoid saving all products to disk and save disk space.
    mcm = cu.read_mcm(
        f"{couplings_dir}/mcm.npz",
        full_mcm=True
    )
    bpwins = {"filtered": {}, "unfiltered": {}}
    tfs = {}
    tfs_std = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):

        bpwin_fn = f"{couplings_dir}/bp_win_{ms1}_x_{ms2}.npz"
        if os.path.isfile(bpwin_fn):
            bpwins[ms1, ms2] = np.load(bpwin_fn)["bp_win"]
            continue

        _, bl1 = su.read_beam_from_file(
            "/".join([
                meta.beam_dir_from_map_set(ms1),
                meta.beam_file_from_map_set(ms1)
            ]),
            lmax=meta.lmax
        )
        _, bl2 = su.read_beam_from_file(
            "/".join([
                meta.beam_dir_from_map_set(ms2),
                meta.beam_file_from_map_set(ms2)
            ]),
            lmax=meta.lmax
        )
        beam = np.outer(bl1, bl2)

        transfer, transfer_std = cu.load_transfer_function(
            meta.transfer_settings["transfer_directory"],
            ms1, ms2,
            meta.filtering_tag_from_map_set,
            meta.kspace_tag_from_map_set,
            nmt_bins,
            return_std=True
        )
        tfs[ms1, ms2] = transfer
        tfs_std[ms1, ms2] = transfer_std
        bpwins["filtered"][ms1, ms2], _ = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=transfer,
            compute_Dl=meta.compute_Dl,
            beam=beam
        )
        bpwins["unfiltered"][ms1, ms2], _ = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=None,
            compute_Dl=meta.compute_Dl,
            beam=beam
        )

    # Then we read the decoupled spectra
    # both for the filtered and unfiltered cases
    cl_dir = f"{out_dir}/cells_tf_val"
    ftypes = ["filtered", "unfiltered"]

    cls_dict = {
        (ftype, ms1, ms2, fp): []
        for fp in fields
        for ftype in ftypes
        for ms1, ms2 in ps_pairs
    }

    for ftype in ftypes:
        for ms1, ms2 in ps_pairs:
            preproc_ftag1 = meta.filtering_tag_from_map_set(ms1)
            kspace_tag1 = meta.kspace_tag_from_map_set(ms1)
            preproc_ftag2 = meta.filtering_tag_from_map_set(ms2)
            kspace_tag2 = meta.kspace_tag_from_map_set(ms2)
            for id_sim in range(nsims):
                cls = np.load(f"{cl_dir}/cls_tf_val_{preproc_ftag1}_{kspace_tag1}_x_{preproc_ftag2}_{kspace_tag2}_{ftype}_{id_sim:04d}.npz")  # noqa: E501
                for fp in fields:
                    cls_dict[ftype, ms1, ms2, fp] += [cls[fp]]

    # Compute mean and std
    cls_mean_dict = {
        (ftype, ms1, ms2, fp):
        np.mean(cls_dict[ftype, ms1, ms2, fp], axis=0)
        for ftype in ftypes
        for fp in fields
        for ms1, ms2 in ps_pairs
    }
    cls_std_dict = {
        (ftype, ms1, ms2, fp):
        np.std(cls_dict[ftype, ms1, ms2, fp], axis=0)
        for ftype in ftypes
        for fp in fields
        for ms1, ms2 in ps_pairs
    }

    # Compute the bandpower-convolved theory spectra for (un)filtered sims.
    cls_theory = meta.load_fiducial_cl()
    cls_theory_binned = {"filtered": {}, "unfiltered": {}}

    for ftype in ["filtered", "unfiltered"]:
        for ms1, ms2 in ps_pairs:
            cls_theory_binned[ftype][ms1, ms2] = pu.bin_theory_cls(
                cls_theory[ms1, ms2], bpwins[ftype][ms1, ms2]
            )

    # Make plots
    for ms1, ms2 in ps_pairs:
        plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)

        for id1, f1 in enumerate("TEB"):
            for id2, f2 in enumerate("TEB"):
                main = plt.subplot(grid[3*id1:3*(id1+1)-1, id2])
                sub = plt.subplot(grid[3*(id1+1)-1, id2])

                spec = f2 + f1 if id1 > id2 else f1 + f2

                # Plot theory
                main.plot(
                    lb[lb_msk],
                    cb2db[lb_msk]*cls_theory_binned["unfiltered"][ms1, ms2][spec][lb_msk],  # noqa: E501
                    color="darkorange", ls="--", alpha=0.6
                )
                main.plot(
                    lb[lb_msk],
                    cb2db[lb_msk]*cls_theory_binned["filtered"][ms1, ms2][spec][lb_msk],  # noqa: E501
                    color="navy", ls="--", alpha=0.6
                )
                main.plot([], [], "k.", label="Simulations")
                main.plot([], [], "k--", alpha=0.6, label="Theory")
                offset = 2

                # Plot filtered & unfiltered (decoupled)
                main.errorbar(
                    lb[lb_msk]-offset,
                    cb2db[lb_msk]*cls_mean_dict["unfiltered", ms1, ms2, spec][lb_msk],  # noqa: E501
                    cb2db[lb_msk]*cls_std_dict["unfiltered", ms1, ms2, spec][lb_msk],  # noqa: E501
                    color="navy",
                    marker=".",
                    markerfacecolor="white",
                    label=r"masked, decoupled",
                    ls="None"
                )
                main.errorbar(
                    lb[lb_msk]+offset,
                    cb2db[lb_msk]*cls_mean_dict["filtered", ms1, ms2, spec][lb_msk],  # noqa: E501
                    cb2db[lb_msk]*cls_std_dict["filtered", ms1, ms2, spec][lb_msk],  # noqa: E501
                    color="darkorange",
                    marker=".",
                    markerfacecolor="white",
                    label=r"masked, filtered, decoupled",
                    ls="None"
                )
                if f1 == f2:
                    main.set_yscale("log")

                # Plot residuals
                res_unf = (cls_mean_dict["unfiltered", ms1, ms2, spec] -
                           cls_theory_binned["unfiltered"][ms1, ms2][spec])
                res_unf /= ((cls_std_dict["unfiltered", ms1, ms2, spec]
                             / np.sqrt(nsims)))
                res_f = (cls_mean_dict["filtered", ms1, ms2, spec] -
                         cls_theory_binned["filtered"][ms1, ms2][spec])
                res_f /= ((cls_std_dict["filtered", ms1, ms2, spec]
                           / np.sqrt(nsims)))

                sub.axhspan(-3, 3, color="k", alpha=0.2)
                sub.axhspan(-2, 2, color="k", alpha=0.2)

                sub.axhline(0, color="k")
                sub.plot(
                    lb[lb_msk]-offset, res_unf[lb_msk], c="navy", ls="",
                    marker="."
                )
                sub.plot(
                    lb[lb_msk]+offset, res_f[lb_msk], c="darkorange", ls="",
                    marker="."
                )

                # Multipole range
                main.set_xlim(2, meta.lmax)
                sub.set_xlim(*main.get_xlim())

                # Suplot y range
                ymin = -5
                ymax = 5
                sub.set_ylim((ymin, ymax))

                # Mark points whose central value is outside the visible range
                for color, res, offset in zip(["navy", "darkorange"],
                                              [res_unf, res_f],
                                              [-5, 5]):
                    for xi, yi in zip(lb[lb_msk]+offset, res[lb_msk]):
                        if yi > ymax:
                            sub.annotate(
                                f'{yi:.1f}',
                                xy=(xi, ymax),  # arrow tip at top boundary
                                xytext=(xi, ymax - 1),  # text inside plot
                                ha='center',
                                va='top',
                                color=color,
                                fontsize=8,
                                arrowprops=dict(
                                    arrowstyle='->',
                                    lw=1.5,
                                    color=color,
                                )
                            )

                        elif yi < ymin:
                            sub.annotate(
                                f'{yi:.1f}',
                                xy=(xi, ymin),
                                xytext=(xi, ymin + 1),
                                ha='center',
                                va='bottom',
                                color=color,
                                fontsize=8,
                                arrowprops=dict(
                                    arrowstyle='->',
                                    lw=1.5,
                                    color=color,
                                )
                            )

                # TF range
                # We cut every low-ell bin whose TF is negative or measured
                # at less than 2 sigma.
                # We also cut all bins centered below ell of 30.
                transfer = tfs[ms1, ms2][fields.index(f1+f2),
                                         fields.index(f1+f2), :]
                transfer_std = tfs_std[ms1, ms2][fields.index(f1+f2),
                                                 fields.index(f1+f2), :]
                tf_zscore = transfer / transfer_std
                good = tf_zscore > 2.

                lmin = max((lb[~good][-1] + lb[good][0])/2., 30)
                main.axvspan(xmin=lb[0]/2., xmax=lmin, color="k", alpha=0.3)
                sub.axvspan(xmin=lb[0]/2., xmax=lmin, color="k", alpha=0.3)

                # Cosmetix
                main.set_title(f1+f2, fontsize=14)
                if spec == "TT":
                    main.legend(fontsize=12, frameon=False)
                main.set_xticklabels([])
                if id1 != 2:
                    sub.set_xticklabels([])
                else:
                    sub.set_xlabel(r"$\ell$", fontsize=13)

                if id2 == 0:
                    main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$", fontsize=13)
                    sub.set_ylabel(
                        r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$",
                        fontsize=13
                    )
        plt.savefig(f"{plot_dir}/cls_{ms1}_x_{ms2}.pdf", bbox_inches="tight")
    print(f"Validation plots saved at {plot_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer function validation"
    )
    parser.add_argument(
        "--globals",
        help='Path to yaml with global parameters'
    )
    args = parser.parse_args()

    main(args)
