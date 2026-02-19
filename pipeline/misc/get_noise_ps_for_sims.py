import argparse
from soopercool import BBmeta
import numpy as np
from itertools import product
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import soopercool.map_utils as mu


def smooth_array(arr, kernel_size):
    ker = np.ones(kernel_size) / kernel_size
    return np.convolve(arr, ker, mode='same')


def interp_array(x, y):
    """
    """
    return interp1d(x, y, fill_value='extrapolate')


def noise_fit(lb, nlb, ls):
    """
    """

    p0 = [50., -2.]
    Nwhite = np.median(nlb[lb > 250])

    def noise_curve(ls, lknee, alpha_knee):
        return Nwhite * (1. + (ls/lknee)**alpha_knee)

    lknee, alpha_knee = curve_fit(
        noise_curve, lb, nlb, p0=p0, bounds=([2, -4.], [250, -1.])
    )[0]
    nl_best = noise_curve(ls, int(lknee), alpha_knee)

    return nl_best, Nwhite, int(lknee), alpha_knee


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory

    cells_dir = f"{out_dir}/cells"
    plot_dir = f"{out_dir}/plots/noise"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()

    lmax = mu.lmax_from_map(meta.masks["analysis_mask"],
                            pix_type=meta.pix_type)
    nl = lmax + 1
    ell = np.arange(nl)
    lmin_fit = 60 if args.lmin_fit is None else args.lmin_fit
    lmax_fit = 60 if args.lmax_fit is None else args.lmax_fit

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    # Only estimate noise from SAT data
    map_set_list = [
        ms for ms in meta.map_sets_list
        if "sat" in meta.exp_tag_from_map_set(ms).lower()
    ]
    cross_map_set_list = [
        (ms1, ms2)
        for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True)
        if ("sat" in meta.exp_tag_from_map_set(ms1).lower()
            and "sat" in meta.exp_tag_from_map_set(ms2).lower())
    ]

    # Load beams to correct the mode coupling matrix
    beams = {}
    for map_set in map_set_list:
        beam_dir = meta.beam_dir_from_map_set(map_set)
        beam_file = meta.beam_file_from_map_set(map_set)

        l, bl = np.loadtxt(f"{beam_dir}/{beam_file}", unpack=True)
        beams[map_set] = bl[:nl]

    for map_set1, map_set2 in cross_map_set_list:
        if verbose:
            print(f"{map_set1} {map_set2}")

        noise_dict = np.load(f"{cells_dir}/decoupled_noise_pcls_{map_set1}_x_{map_set2}.npz")  # noqa

        for field_pair in field_pairs:
            nb = noise_dict[field_pair]

            # Ensure positivity
            msk = np.array(nb > 0) & np.array(lb >= lmin_fit) & np.array(lb < lmax_fit)  # noqa

            try:
                # Fit white noise and 1/f model
                nl_fit, Nw, lk, alk = noise_fit(lb[msk], nb[msk], ell)
            except ValueError:  # fit gone wrong
                print(f"Fit did not converge: {map_set1}x{map_set2} "
                      f"| {field_pair}")
                continue

            msg = fr"$N_w={Nw:.2e}$, $\ell_k={lk}$, $\alpha_k={alk:.1f}$"

            if do_plots:
                plt.figure(figsize=(10, 8))
                plt.title(fr"{map_set1} x {map_set2} - {field_pair}")
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.ylabel(r"$N_\ell^\mathrm{%s}$" % field_pair, fontsize=15)
                plt.plot(lb, nb, c="navy", ls="", marker=".", alpha=0.2,
                         label="Original")
                plt.plot(lb[msk], nb[msk], c="navy", ls="", marker=".",
                         label="Fitted to")
                plt.axhline(2*Nw, color="r", ls=":")
                plt.axvline(lk, color="r", ls=":")
                plt.plot(ell, nl_fit, c="r", ls="--", label=fr"Fit {msg}")
                plt.legend(fontsize=14)
                plt.yscale("log")
                plt.xlim(0, meta.lmax)
                lims = (1e-1, 1e7) if field_pair == "TT" else (1e-5, 1e2)
                plt.ylim(lims)
                plt.savefig(f"{plot_dir}/noise_{map_set1}_x_{map_set2}_{field_pair}.png") # noqa
                plt.close()

        if do_plots:
            print(f"  Plots: {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get N_ell parameters from data"
    )
    parser.add_argument("--globals", help="Path to the paramfile")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--no-plots", help="Plot the results",
                        action="store_true")
    parser.add_argument("--lmin_fit", type=int, default=None,
                        help="Minimum ell used for fit")
    parser.add_argument("--lmax_fit", type=int, default=None,
                        help="Maximum ell used for fit")

    args = parser.parse_args()

    main(args)
