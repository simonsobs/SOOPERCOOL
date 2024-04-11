import matplotlib.pyplot as plt
from soopercool import BBmeta
import argparse
from itertools import product
import sacc
import numpy as np
from soopercool import ps_utils


def load_bpwins(coupling_file):
    """
    """
    bp_win = np.load(coupling_file)
    bpw_mat = {}
    for spin_pair in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat[spin_pair] = bp_win[f"bp_win_{spin_pair}"]

    return bpw_mat


def binned_theory_from_unbinned(clth, bpw_mat):
    """
    """
    clth_binned_dict = {}
    for spin_pair, modes in zip(
        ["spin0xspin0", "spin0xspin2", "spin2xspin2"],
        [["TT"], ["TE", "TB"], ["EE", "EB", "BE", "BB"]]
    ):

        clth_vec = np.concatenate(
            [clth[mode] for mode in modes]
        ).reshape(len(modes), -1)
        clth_binned = np.einsum("ijkl,kl", bpw_mat[spin_pair], clth_vec)
        clth_binned_dict[spin_pair] = clth_binned

    clth_binned = ps_utils.field_pairs_from_spins(clth_binned_dict)

    to_update = []
    for k, v in clth_binned.items():
        if not (k[::-1] in clth_binned):
            to_update.append((k[::-1], v))

    for k, v in to_update:
        clth_binned[k] = v

    return clth_binned


def multipole_min_from_tf(tf_file, snr_cut=3.):
    """
    """
    tf = np.load(tf_file)
    idx_bad_tf = {}
    for spin_pair in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        tf_mean = tf[f"tf_{spin_pair}"][0, 0]
        tf_std = tf[f"tf_std_{spin_pair}"][0, 0]
        snr = tf_mean / tf_std
        idx = np.where(snr < 3.)[0]
        idx_bad_tf[spin_pair] = idx.max()

    idx_bad_tf["spin2xspin0"] = idx_bad_tf["spin0xspin2"]

    return idx_bad_tf


def plot_spectrum(lb, cb, cb_err, title, ylabel, xlim,
                  add_theory=False, lth=None, clth=None,
                  cbth=None, save_file=None):
    """
    """
    plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(4, 1, wspace=0, hspace=0)
    if add_theory:
        main = plt.subplot(grid[:-1])
        sub = plt.subplot(grid[-1])
        sub.set_xlabel(r"$\ell$")
        sub.set_ylabel(r"$\Delta C_\ell / (\sigma / \sqrt{N_\mathrm{sims}})$")
    else:
        main = plt.subplot(grid[:])
        main.set_xlabel(r"$\ell$")

    main.set_ylabel(r"$\ell (\ell + 1) C_\ell^{%s} / 2\pi$" % ylabel)
    main.set_title(title)

    fac = lb * (lb + 1) / (2 * np.pi)

    if add_theory:
        fac_th = lth * (lth + 1) / (2 * np.pi)
        main.plot(lth, fac_th * clth, c="darkgray", ls="-.", lw=2.6)

    main.errorbar(
        lb, fac * cb, yerr=fac * cb_err, marker="o", ls="None",
        markerfacecolor="white", markeredgecolor="navy",
        elinewidth=1.75, ecolor="navy", markeredgewidth=1.75
    )

    main.set_xlim(*xlim)

    if add_theory:
        sub.axhspan(-3, 3, color="gray", alpha=0.1)
        sub.axhspan(-2, 2, color="gray", alpha=0.4)
        sub.axhspan(-1, 1, color="gray", alpha=0.8)

        sub.plot(lb, (cb - cbth) / cb_err, marker="o", ls="None",
                 markerfacecolor="white", markeredgecolor="navy",
                 markeredgewidth=1.75)

        sub.set_xlim(*xlim)
        sub.set_ylim(-4.5, 4.5)

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()


def sacc_plotter(args):
    """
    This script will read the spectra and covariance
    stored in the `sacc` files and plot the power
    spectra.
    """
    meta = BBmeta(args.globals)
    sacc_dir = meta.sacc_directory
    coupling_dir = meta.coupling_directory

    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]
    ps_names = meta.get_ps_names_list(type="all", coadd=True)

    spins = {"T": 0, "E": 2, "B": 2}
    types = {"T": "0", "E": "e", "B": "b"}

    Nsims = meta.num_sims if args.sims else 1

    psth = meta.load_fiducial_cl(cl_type="cosmo")
    ps_th = {}
    for field_pair in field_pairs:
        if field_pair in psth:
            ps_th[field_pair] = psth[field_pair]
        else:
            ps_th[field_pair] = psth[field_pair[::-1]]

    # Transfer function
    idx_bad_tf = {}
    bpw_mats = {}

    for ftag1, ftag2 in meta.get_independent_filtering_pairs():
        idx = multipole_min_from_tf(
            f"{coupling_dir}/transfer_function_{ftag1}x{ftag2}.npz",
            snr_cut=3
        )
        idx_bad_tf[ftag1, ftag2] = idx

        bpw_file = f"couplings_{ftag1}x{ftag2}_unfiltered.npz"
        bpw_mats[ftag1, ftag2] = load_bpwins(f"{coupling_dir}/{bpw_file}")

    # Bandpower window functions
    fields_to_spin = {
        "T": "spin0",
        "E": "spin2",
        "B": "spin2"
    }

    clth_binned = {
        (ftag1, ftag2): binned_theory_from_unbinned(ps_th, bpw_mat)
        for (ftag1, ftag2), bpw_mat in bpw_mats.items()
    }

    plot_data = {
        (ms1, ms2, fp): {
            "x": None,
            "y": [],
            "err": None,
            "x_th": None,
            "y_th": None,
            "th_binned": None,
            "title": None,
            "ylabel": None
        } for ms1, ms2 in ps_names
        for fp in field_pairs
    }

    if args.sims:
        # Load theory
        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                mask_th = (psth["l"] <= 2 * meta.nside - 1)
                x_th, y_th = psth["l"][mask_th], psth[fp][mask_th]

                s1, s2 = fields_to_spin[fp[0]], fields_to_spin[fp[1]]

                idx_bad = idx_bad_tf[ftag1, ftag2][f"{s1}x{s2}"]
                mask = ((np.arange(len(lb)) > idx_bad) &
                        (lb <= 2 * meta.nside - 1))
                th_binned = clth_binned[ftag1, ftag2][fp][mask]

                plot_data[ms1, ms2, fp]["x_th"] = x_th
                plot_data[ms1, ms2, fp]["y_th"] = y_th
                plot_data[ms1, ms2, fp]["th_binned"] = th_binned

    for id_sim in range(Nsims):
        sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""

        s = sacc.Sacc.load_fits(f"{sacc_dir}/cl_and_cov_sacc{sim_label}.fits")

        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                f1, f2 = fp
                s1, s2 = fields_to_spin[f1], fields_to_spin[f2]

                ell, cl, cov = s.get_ell_cl(
                    f"cl_{types[f1]}{types[f2]}",
                    f"{ms1}_s{spins[f1]}",
                    f"{ms2}_s{spins[f2]}",
                    return_cov=True)

                idx_bad = idx_bad_tf[ftag1, ftag2][f"{s1}x{s2}"]
                mask = (
                    (np.arange(len(ell)) > idx_bad) &
                    (ell <= 2 * meta.nside - 1))
                x, y, err = ell, cl, np.sqrt(cov.diagonal())
                x, y, err = x[mask], y[mask], err[mask]

                plot_data[ms1, ms2, fp]["x"] = x
                plot_data[ms1, ms2, fp]["y"].append(y)
                plot_data[ms1, ms2, fp]["err"] = err
                plot_data[ms1, ms2, fp]["title"] = f"{ms1} x {ms2} - {fp}"
                plot_data[ms1, ms2, fp]["ylabel"] = fp

    plot_dir = meta.plot_dir_from_output_dir(
            meta.cell_sims_directory_rel if args.sims
            else meta.cell_data_directory_rel
    )

    for ms1, ms2 in ps_names:
        for fp in field_pairs:

            plot_data[ms1, ms2, fp]["y"] = np.mean(
                plot_data[ms1, ms2, fp]["y"], axis=0
            )
            plot_data[ms1, ms2, fp]["err"] /= np.sqrt(Nsims)

            plot_name = f"{ms1}_{ms2}_{fp}.pdf"

            plot_spectrum(
                plot_data[ms1, ms2, fp]["x"],
                plot_data[ms1, ms2, fp]["y"],
                plot_data[ms1, ms2, fp]["err"],
                title=plot_data[ms1, ms2, fp]["title"],
                ylabel=plot_data[ms1, ms2, fp]["ylabel"],
                xlim=(30, 2 * meta.nside - 1),
                add_theory=args.sims,
                lth=plot_data[ms1, ms2, fp]["x_th"],
                clth=plot_data[ms1, ms2, fp]["y_th"],
                cbth=plot_data[ms1, ms2, fp]["th_binned"],
                save_file=f"{plot_dir}/{plot_name}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sacc plotter')
    parser.add_argument("--globals", type=str,
                        help="Path to the global configuration file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--data", action="store_true")
    args = parser.parse_args()
    sacc_plotter(args)
