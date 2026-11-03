import argparse
import os
from soopercool import BBmeta
from soopercool import coupling_utils as cu
from soopercool import utils
import numpy as np

FIELD_PAIRS = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


def band_limited_cl(meta, nl, bandlim_ell, bandlim_width):
    """
    Input C_l of band-limited TF-estimation sims: the power law of
    transfer_settings.power_law_pars_tf_est times
    bandlim_sine2(ell, bandlim_ell, bandlim_width)^2, with ell < 2 set to 0.
    """
    ell = np.arange(nl)
    pl = utils.power_law_cl(
        ell, **meta.transfer_settings["power_law_pars_tf_est"]
    )["TT"]
    cl = pl * utils.bandlim_sine2(ell, bandlim_ell, bandlim_width) ** 2
    cl[:2] = 0.
    return ell, cl


def transfer_on_valid_bins(tf_ordering, valid):
    """
    TF estimator that solves only on the bins flagged in `valid` (least
    squares for MT) and copies the last valid bin to the higher bins.
    Above the band limit of the TF sims the MT design matrix has zero
    columns, so inv(D^T D) is singular, and TM has no information either.
    """
    last = np.where(valid)[0].max()

    def tf_valid(pcls_mat_filt, cl, mcm, nmt_binning, compute_Dl=False):
        n_bins = nmt_binning.get_n_bands()
        if tf_ordering == "MT":
            # get_transfer_with_error_MT, with only the valid TF columns
            nl = nmt_binning.lmax + 1
            phiC = np.zeros((n_bins, nl))
            for i in range(n_bins):
                idx = nmt_binning.get_ell_list(i)
                phiC[i, idx] = cl[idx]
            bMphiC = cu.bin_mcm_left(
                np.einsum("DiCj,pj->DiCp", mcm, phiC), nmt_binning
            )
            mcm_binned = cu.bin_mcm_right(
                cu.bin_mcm_left(mcm, nmt_binning), nmt_binning,
                compute_Dl=compute_Dl
            )
            size = mcm_binned.shape[0]
            mcm_binned_inv = np.linalg.inv(
                mcm_binned.reshape([size*n_bins, size*n_bins])
            ).reshape([size, n_bins, size, n_bins])
            D = np.einsum(
                "AkDm,DmCp->AkCp", mcm_binned_inv, bMphiC
            )[..., valid]
            nv = valid.sum()
            cls = np.einsum("AkBp,NBpC->NAkC", mcm_binned_inv,
                            np.transpose(pcls_mat_filt, [0, 2, 3, 1]))
            D_inv = np.linalg.pinv(D.reshape([size*n_bins, size*nv]))
            T_mean = (
                D_inv @ np.mean(cls, axis=0).reshape([size*n_bins, size])
            ).reshape([size, nv, size])
            T_std = np.std(
                np.einsum("ab,Nbc->Nac", D_inv,
                          cls.reshape([-1, size*n_bins, size])),
                axis=0
            ).reshape([size, nv, size])
            T_std /= np.sqrt(np.array(pcls_mat_filt).shape[0])
            tm = np.zeros((size, size, n_bins))
            ts = np.zeros((size, size, n_bins))
            tm[..., valid] = np.transpose(T_mean, [0, 2, 1])
            ts[..., valid] = np.transpose(T_std, [0, 2, 1])
        else:
            tm, ts = cu.get_transfer_with_error_TM(
                pcls_mat_filt, cl, mcm, nmt_binning, compute_Dl
            )
        tm[..., last+1:] = tm[..., last:last+1]
        ts[..., last+1:] = ts[..., last:last+1]
        return tm, ts

    return tf_valid


def main(args):
    """
    Compute the filtering transfer function as in Hervias et al. 2025.

    Requires a set of filtered and unfiltered pure-type power spectrum
    matrices, obtained with 'compute_pseudo_cells_tf_estimation.py'.

    Saves the full set of cross-filtering-tag transfer functions (as defined
    under 'map_sets' in the yaml) under tf_settings["transfer_directory"].

    By default the TF-estimation sims are taken to be band-limited (as the
    SAT ISO v4 TF sims): the input C_l is the power law times the sine^2
    band limit (--bandlim-ell, --bandlim-width), written to
    tf_settings["power_law_c_ell"] if missing and checked against it
    otherwise, and the TF is solved only on bins with
    ell_max <= --lmax-valid (higher bins take the last valid value).
    --no-bandlim reads the input C_l from tf_settings["power_law_c_ell"]
    and solves the TF on all bins.
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_tf_est"
    do_plot = not args.no_plots

    tf_settings = meta.transfer_settings
    tf_dir = tf_settings["transfer_directory"]
    BBmeta.make_dir(tf_dir)

    nmt_bins = meta.read_nmt_binning()
    nl = nmt_bins.lmax + 1
    lb = nmt_bins.get_effective_ells()

    filtering_pairs = meta.get_independent_filtering_pairs()

    pcls_mat_dict = cu.read_pcls_matrices(
        cells_dir, filtering_pairs,
        tf_settings["tf_est_num_sims"],
        tf_settings["sim_id_start"]
    )

    # Compute and save the transfer functions
    mcm = cu.read_mcm(
        f"{out_dir}/couplings/mcm.npz",
        full_mcm=True
    )
    if args.no_bandlim:
        cl = np.load(
            tf_settings["power_law_c_ell"]
        )
        cl = cl["cl"][:nl]

        trans = cu.get_transfer_dict(
            pcls_mat_dict,
            filtering_pairs,
            cl,
            mcm,
            nmt_bins,
            compute_Dl=meta.compute_Dl,
            tf_ordering=tf_settings["tf_ordering"]
        )
    else:
        trans = band_limited_transfer(
            args, meta, nmt_bins, pcls_mat_dict, filtering_pairs, mcm
        )
    full_tf = {}
    for ftag1, ftag2 in filtering_pairs:
        lab1 = f"{ftag1[0]}_{ftag1[1]}"
        lab2 = f"{ftag2[0]}_{ftag2[1]}"
        tf = trans[ftag1, ftag2]
        np.savez(
            f"{tf_dir}/transfer_function_{lab1}_x_{lab2}.npz",
            **tf
        )
        full_tf[ftag1, ftag2] = tf["full_tf"]

    if do_plot:
        plot_dir = f"{out_dir}/plots/transfer_functions"
        BBmeta.make_dir(plot_dir)

        for ftag1, ftag2 in filtering_pairs:
            lab1 = f"{ftag1[0]}_{ftag1[1]}"
            lab2 = f"{ftag2[0]}_{ftag2[1]}"
            tf_dict = np.load(f"{tf_dir}/transfer_function_{lab1}_x_{lab2}.npz")  # noqa: E501

            utils.plot_transfer_function(
                lb, tf_dict, meta.lmin, meta.lmax,
                ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"],
                file_name=f"{plot_dir}/transfer_{lab1}_x_{lab2}.pdf"
            )
        print(f"Saved TF plots under {plot_dir}")


def band_limited_transfer(args, meta, nmt_bins, pcls_mat_dict,
                          filtering_pairs, mcm):
    """
    Transfer functions for band-limited TF-estimation sims (see main).
    Returns the same dictionary as coupling_utils.get_transfer_dict.
    """
    tf_settings = meta.transfer_settings
    tf_ordering = tf_settings["tf_ordering"]
    nl = nmt_bins.lmax + 1
    n_bins = nmt_bins.get_n_bands()
    lb = nmt_bins.get_effective_ells()

    ell, cl = band_limited_cl(meta, nl, args.bandlim_ell, args.bandlim_width)
    cl_file = tf_settings["power_law_c_ell"]
    if os.path.isfile(cl_file):
        cl_saved = np.load(cl_file)["cl"][:nl]
        if not np.allclose(cl_saved, cl, rtol=1e-12, atol=0):
            raise ValueError(
                f"{cl_file} differs from the band-limited power law"
            )
    else:
        BBmeta.make_dir(os.path.dirname(cl_file))
        np.savez(cl_file, l=ell, cl=cl)
        print(f"Wrote input C_l to {cl_file}")

    valid = np.array([
        nmt_bins.get_ell_list(i).max() <= args.lmax_valid
        for i in range(n_bins)
    ])
    last = np.where(valid)[0].max()
    print(f"tf_ordering={tf_ordering}; TF solved on bins 0..{last} "
          f"(ell <= {nmt_bins.get_ell_list(last).max()}), "
          f"bins {last+1}..{n_bins-1} copied from bin {last}", flush=True)

    # Input C_l check: mean unfiltered pure-X pseudo-Cl vs binned M C_l
    bmc = np.einsum("CkBj,j->BCk", cu.bin_mcm_left(mcm, nmt_bins), cl)
    for ft1, ft2 in filtering_pairs:
        unf = np.mean(pcls_mat_dict[ft1, ft2]["unfiltered"], axis=0)
        for fp in ("TT", "EE", "BB"):
            i = FIELD_PAIRS.index(fp)
            r = unf[i, i] / bmc[i, i]
            sel = (lb > 30) & (lb < 600)
            print(f"  C_l check {ft1} x {ft2} {fp}: unfiltered/(M C_l) "
                  f"median {np.median(r[sel]):.4f}, max |dev| ell 30-600 "
                  f"{np.abs(r[sel] - 1).max():.4f}", flush=True)
            if abs(np.median(r[sel]) - 1) > 0.03:
                raise ValueError(
                    "Input C_l does not match the unfiltered TF sims"
                )

    tf_func = transfer_on_valid_bins(tf_ordering, valid)
    trans = {}
    for ftag1, ftag2 in filtering_pairs:
        tf, tferr = tf_func(
            pcls_mat_dict[ftag1, ftag2]["filtered"], cl, mcm, nmt_bins,
            meta.compute_Dl
        )
        trans[ftag1, ftag2] = {}
        for i, fp1 in enumerate(FIELD_PAIRS):
            for j, fp2 in enumerate(FIELD_PAIRS):
                trans[ftag1, ftag2][f"{fp2}_to_{fp1}"] = tf[i, j]
                trans[ftag1, ftag2][f"{fp2}_to_{fp1}_std"] = tferr[i, j]
        trans[ftag1, ftag2]["full_tf"] = tf
        trans[ftag1, ftag2]["full_tf_std"] = tferr
    return trans


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not plot transfer function.")
    parser.add_argument("--no-bandlim", action="store_true",
                        help="TF-estimation sims are not band-limited: read "
                             "the input C_l from power_law_c_ell and solve "
                             "the TF on all bins.")
    parser.add_argument("--bandlim-ell", type=float, default=650.,
                        help="Centre of the sine^2 band limit.")
    parser.add_argument("--bandlim-width", type=float, default=50.,
                        help="Width of the sine^2 band limit.")
    parser.add_argument("--lmax-valid", type=int, default=700,
                        help="Largest ell_max of a bin on which the TF is "
                             "solved.")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
