import argparse
import numpy as np
from soopercool import BBmeta
import matplotlib.pyplot as plt
from soopercool import coupling_utils as cu

# TODO:
# Read in beam for validation and correct C_ell estimator by it. Currently we
# assume a bean FWHM of 30 arcmin by default.


def read_transfer(transfer_file):
    """
    """
    tfd = np.load(transfer_file)

    tf_dict = {}
    spin_pairs = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }
    for spin_pair, fields in spin_pairs.items():
        for id1, f1 in enumerate(fields):
            for id2, f2 in enumerate(fields):
                tf_dict[f1, f2] = tfd[f"tf_{spin_pair}"][id1, id2]
                tf_dict[f1, f2, "std"] = tfd[f"tf_std_{spin_pair}"][id1, id2]
    return tf_dict


def main(args):
    """
    Compare decoupled spectra (including deconvolved form the TF)
    to the unfiltered spectra (deconvolved from mode coupling only)
    Compare coupled spectra (with TF + MCM)
    """
    meta = BBmeta(args.globals)

    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()
    lb_msk = lb < meta.lmax
    cb2db = lb*(lb+1)/2/np.pi

    nsims = meta.transfer_settings["tf_val_num_sims"]

    out_dir = meta.output_directory
    plot_dir = f"{out_dir}/plots/cells_tf_val"
    BBmeta.make_dir(plot_dir)

    ps_pairs = meta.get_ps_names_list(type="all", coadd=True)
    fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

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
            for id_sim in range(nsims):
                cls = np.load(f"{cl_dir}/cls_tf_val_{ms1}_x_{ms2}_{ftype}_{id_sim:04d}.npz")  # noqa
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
    cls_theory = meta.load_fiducial_cl()

    cls_theory_binned = {"filtered": {}, "unfiltered": {}}
    _, bpwf = meta.get_inverse_couplings(return_bpwf=True)

    for type in ["filtered", "unfiltered"]:
        for ms1, ms2 in ps_pairs:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)
            cls_theory_binned[type][ms1, ms2] = cu.bin_theory_cls(
                cls_theory[ms1, ms2], bpwf[type][ftag1, ftag2]
            )

    for ms1, ms2 in ps_pairs:
        ftag1 = meta.filtering_tag_from_map_set(ms1)
        ftag2 = meta.filtering_tag_from_map_set(ms2)

        plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)

        for id1, f1 in enumerate("TEB"):
            for id2, f2 in enumerate("TEB"):
                # Define subplots
                main = plt.subplot(grid[3*id1:3*(id1+1)-1, id2])
                sub = plt.subplot(grid[3*(id1+1)-1, id2])

                spec = f2 + f1 if id1 > id2 else f1 + f2

                # Plot theory
                main.plot(
                    lb[lb_msk],
                    cb2db[lb_msk]*cls_theory_binned["unfiltered"][ms1, ms2][spec][lb_msk],  # noqa
                    color="k", ls="--", label="theory"
                )
                main.plot(
                    lb[lb_msk],
                    cb2db[lb_msk]*cls_theory_binned["filtered"][ms1, ms2][spec][lb_msk],  # noqa
                    color="k", ls="--", alpha=0.3
                )
                offset = 0.5

                # Plot filtered & unfiltered (decoupled)
                main.errorbar(
                    lb[lb_msk]-offset,
                    cb2db[lb_msk]*cls_mean_dict["unfiltered", ms1, ms2, spec][lb_msk],  # noqa
                    cb2db[lb_msk]*cls_std_dict["unfiltered", ms1, ms2, spec][lb_msk],  # noqa
                    color="navy",
                    marker=".",
                    markerfacecolor="white",
                    label=r"masked, decoupled",
                    ls="None"
                )
                main.errorbar(
                    lb[lb_msk]+offset,
                    cb2db[lb_msk]*cls_mean_dict["filtered", ms1, ms2, spec][lb_msk],  # noqa
                    cb2db[lb_msk]*cls_std_dict["filtered", ms1, ms2, spec][lb_msk],  # noqa
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
                    lb[lb_msk]-offset, res_unf[lb_msk], c="navy", ls="-"
                )
                sub.plot(
                    lb[lb_msk]+offset, res_f[lb_msk], c="darkorange", ls="-"
                )

                # Multipole range
                main.set_xlim(2, meta.lmax)
                sub.set_xlim(*main.get_xlim())

                # Suplot y range
                sub.set_ylim((-5, 5))

                # Cosmetix
                main.set_title(f1+f2, fontsize=14)
                if spec == "TT":
                    main.legend(fontsize=13)
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

            plt.savefig(f"{plot_dir}/cls_{ms1}_x_{ms2}.pdf",
                        bbox_inches="tight")


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
