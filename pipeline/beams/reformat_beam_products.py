from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import argparse
import os


def read_beam_product(beam_fname):
    """
    """
    data = np.load(beam_fname)

    bl = np.abs(data["Bl"])
    norm = (2 * np.pi / data["omega"])

    cov_stat = data["analytic_cov_Bl"]
    cov_theta1 = data["theta1_cov_Bl"]
    cov_lmax = data["lmax_cov_Bl"]

    cov = cov_stat + cov_theta1 + cov_lmax

    bl = bl * norm
    cov = cov * norm**2

    return bl, cov, data["fwhm"]


def plot_beam(bl, cov, fwhm=None, title="", save_fname=None):
    """
    """
    plt.figure(figsize=(8, 6))
    l, = plt.plot(np.arange(len(bl)), bl)
    plt.fill_between(
        np.arange(len(bl)),
        bl - np.sqrt(np.diag(cov)),
        bl + np.sqrt(np.diag(cov)),
        alpha=0.5,
        color=l.get_color()
    )
    if fwhm is not None:
        plt.plot(
            np.arange(len(bl)),
            hp.gauss_beam(np.deg2rad(fwhm/60), lmax=len(bl) - 1),
            ls="--",
            color=l.get_color()
        )
    plt.xlim(0, 1200)
    plt.ylim(0, np.max(bl+np.sqrt(np.diag(cov))))
    plt.xlabel(r"$\ell$", fontsize=15)
    plt.ylabel(r"$b_\ell$", fontsize=15)
    plt.title(title, fontsize=15)
    if save_fname is not None:
        plt.savefig(save_fname, bbox_inches="tight")
    plt.close()


def plot_corr(cov, title="", save_fname=None):
    """
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(
        cov / np.outer(
            np.sqrt(np.diag(cov)),
            np.sqrt(np.diag(cov))
        ),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )
    plt.colorbar()
    plt.title(title, fontsize=15)
    if save_fname is not None:
        plt.savefig(save_fname, bbox_inches="tight")
    plt.close()


def main(args):
    """
    This utility script read raw beam products
    per telescope, band, wafer_slot
    and reformat them into a standard
    format to be used in the rest of
    the pipeline.

    Command line arguments are
    --beam-dir:
        Path to the raw beam products
    --beam-tmpl:
        Template for raw file names with
        placeholders {tel}, {band}, {wafer}.
        We assume this is a .npz file
    --out-dir:
        Where to save the outputs
        i.e. reformatted beams and
        associated covariances
    --telescopes:
        List of telescopes. Should be
        comma separated string as
        "satp1,satp3"
    --bands:
        List of bands. Should be
        comma separated string as
        "f090,f150"
    --wafers:
        List of wafers. Should be
        comma separated string as
        "ws0,ws1,ws2,ws3,ws4,ws5,ws6"
    --wafer-weights:
        Should be a comma separated string of
        weights corresponding to the wafers.
        If not provided, will assume equal weights.
    """
    beam_dir = args.beam_dir
    beam_tmpl = args.beam_tmpl
    out_dir = args.out_dir
    telescopes = args.telescopes.split(",")
    bands = args.bands.split(",")
    wafers = args.wafers.split(",")

    if args.wafer_weights is not None:
        weights = args.wafer_weights.split(",")
        weights = np.array(weights, dtype=float)
    else:
        weights = np.ones(len(wafers))

    weights /= np.sum(weights)
    if len(weights) != len(wafers):
        raise ValueError("Weights must match number of wafers")

    os.makedirs(f"{out_dir}/plots", exist_ok=True)

    for tel in telescopes:
        for band in bands:

            bl_coadd = 0
            cov_coadd = 0

            for wafer, weight in zip(wafers, weights):
                print(wafer, weight)
                beam_fname = beam_tmpl.format(
                    tel=tel,
                    band=band,
                    wafer=wafer
                )
                try:
                    bl, cov, fwhm = read_beam_product(
                        f"{beam_dir}/{beam_fname}"
                    )
                except FileNotFoundError:
                    print(f"File {beam_dir}/{beam_fname} not found. Skipping.")
                    continue
                bl_coadd += weight * bl
                cov_coadd += weight**2 * cov

                plot_beam(
                    bl,
                    cov,
                    fwhm,
                    title=f"{tel} {band} {wafer}",
                    save_fname=f"{out_dir}/plots/beam_{tel}_{band}_{wafer}.png", # noqa
                )
                plot_corr(
                    cov,
                    title=f"{tel} {band} {wafer}",
                    save_fname=f"{out_dir}/plots/beam_cov_{tel}_{band}_{wafer}.png", # noqa
                )

            plot_beam(
                bl_coadd,
                cov_coadd,
                fwhm=None,
                title=f"{tel} {band} coadd",
                save_fname=f"{out_dir}/plots/beam_{tel}_{band}_coadd.png",
            )

            # Keep the first 20 eigenmodes
            # such that we can store it in a
            # compact way
            lambda_k, V_k = eigsh(cov_coadd, k=20, which="LA")
            V_k = np.sqrt(lambda_k) * V_k
            res = (V_k @ V_k.T - cov_coadd) / cov_coadd * 100
            max, med = np.max(np.abs(res)), np.median(np.abs(res))
            print(
                f"Max cov diff for {tel} {band}: {max:.8f} %"
            )
            print(
                f"Median cov diff for {tel} {band}: {med:.8f} %"
            )

            plot_corr(
                cov_coadd,
                title=f"{tel} {band} coadd",
                save_fname=f"{out_dir}/plots/beam_cov_{tel}_{band}_coadd.png",
            )
            ratio = cov_coadd / np.outer(bl_coadd, bl_coadd)
            ratio = ratio.diagonal()
            ratio = ratio[np.arange(len(ratio)) < 650]
            ratio = np.sqrt(ratio)

            to_save = np.hstack([
                np.arange(len(bl_coadd))[:, None],
                bl_coadd[:, None],
                V_k
            ])
            np.savetxt(
                f"{out_dir}/beam_{tel}_{band}_coadd.dat",
                to_save
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam-dir",
        help="Path to the raw beam products",
        type=str
    )
    parser.add_argument(
        "--beam-tmpl",
        help="Template for raw file names",
        type=str,
    )
    parser.add_argument(
        "--out-dir",
        help="Path to reformatted beam products",
        type=str
    )
    parser.add_argument(
        "--telescopes",
        help="List of telescopes",
        type=str,
        default="satp1,satp3"
    )
    parser.add_argument(
        "--bands",
        help="List of bands",
        type=str,
        default="f090,f150"
    )
    parser.add_argument(
        "--wafers",
        help="List of wafers",
        type=str,
        default="ws0,ws1,ws2,ws3,ws4,ws5,ws6"
    )
    parser.add_argument(
        "--wafer-weights",
        help="Path to the wafer weights file",
        default=None
    )

    args = parser.parse_args()
    main(args)
