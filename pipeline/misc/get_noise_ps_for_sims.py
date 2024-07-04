import argparse
from soopercool import BBmeta
import pymaster as nmt
import numpy as np
from itertools import product
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def smooth_array(arr, kernel_size):
    ker = np.ones(kernel_size) / kernel_size
    return np.convolve(arr, ker, mode='same')


def interp_array(x, y):
    """
    """
    return interp1d(x, y, fill_value='extrapolate')


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    # verbose = args.verbose

    out_dir = meta.output_directory

    cells_dir = f"{out_dir}/cells"
    noise_dir = f"{out_dir}/noise_interp"
    BBmeta.make_dir(noise_dir)
    plot_dir = f"{out_dir}/plots/noise_interpolation"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    lb = nmt_bins.get_effective_ells()

    nl = 3 * meta.nside
    ell = np.arange(nl)
    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    # Load beams to correct the mode coupling matrix
    beams = {}
    for map_set in meta.map_sets_list:
        beam_dir = meta.beam_dir_from_map_set(map_set)
        beam_file = meta.beam_file_from_map_set(map_set)

        l, bl = np.loadtxt(f"{beam_dir}/{beam_file}", unpack=True)
        beams[map_set] = bl[:nl]

    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    for map_set1, map_set2 in cross_map_set_list:

        noise_dict = np.load(f"{cells_dir}/decoupled_noise_pcls_{map_set1}_x_{map_set2}.npz") # noqa
        plt.figure()
        plt.plot(lb, noise_dict["EE"])
        plt.title(f"{map_set1} x {map_set2} - EE")
        plt.yscale("log")
        plt.show()

        nb1, nb2 = (meta.n_bundles_from_map_set(map_set1),
                    meta.n_bundles_from_map_set(map_set2))
        interp_noise = {}
        for field_pair in field_pairs:
            nb = noise_dict[field_pair]
            n_int = interp_array(lb, nb)
            nl = n_int(ell) * beams[map_set1] * beams[map_set2]

            interp_noise[field_pair] = nl * np.sqrt(nb1) * np.sqrt(nb2)

            np.savez(f"{noise_dir}/nl_{map_set1}_x_{map_set2}.npz",
                     ell=ell, **interp_noise)

            if do_plots:
                plt.figure(figsize=(10, 8))
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.ylabel(r"$N_\ell^\mathrm{%s}$" % field_pair, fontsize=15)
                plt.plot(lb, nb, label="Original")
                plt.plot(ell, nl, ls="--",
                         label="Interpolated + Beam deconvolved")
                plt.legend()
                plt.xlabel(r"$\ell$")
                plt.ylabel(r"$C_{\ell}$")
                plt.title(f"{map_set1} x {map_set2} - {field_pair}")
                plt.yscale("log")
                plt.xlim(0, 2 * meta.nside)
                plt.savefig(f"{plot_dir}/noise_interp_{map_set1}_x_{map_set2}_{field_pair}.png") # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mask mode coupling matrices"
    )
    parser.add_argument("--globals", help="Path to the paramfile")
    parser.add_argument("--no-plots", help="Plot the results",
                        action="store_true")

    args = parser.parse_args()

    main(args)
