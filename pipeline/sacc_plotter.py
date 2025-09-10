import matplotlib.pyplot as plt
from soopercool import BBmeta
import argparse
from itertools import product
import sacc
import numpy as np
import healpy as hp
import pymaster as nmt
import os
import glob


def load_fiducial_cmb_any(path, lmax):
    """
    Load CMB TT/EE/BB/TE up to lmax from either:
      - FITS readable by healpy.read_cl (shape [>=4, ell])
      - CAMB ASCII .dat with columns [ell, TT, EE, BB, TE] (standard)
    Returns a dict with keys 'TT','EE','BB','TE' each length lmax+1.
    """
    import os
    path = str(path)
    out = {k: np.zeros(lmax+1) for k in ["TT", "EE", "BB", "TE"]}

    if path.lower().endswith((".fits", ".fit", ".fits.gz")):
        arr = hp.read_cl(path)  # shape [spec, ell]
        # Expect at least 4 rows in order TT, EE, BB, TE
        out["TT"][:arr.shape[1]] = arr[0, :lmax+1]
        out["EE"][:arr.shape[1]] = arr[1, :lmax+1]
        out["BB"][:arr.shape[1]] = arr[2, :lmax+1]
        out["TE"][:arr.shape[1]] = arr[3, :lmax+1]
        return out

    # ASCII / .dat (CAMB): columns [ell, TT, EE, BB, TE]
    data = np.loadtxt(path)
    # tolerate files with header lines
    if data.ndim == 1:
        # single-line edge case (unlikely)
        data = data[None, :]
    # find columns by count
    # expected: >=5 columns
    if data.shape[1] < 5:
        raise ValueError(f"{path} doesn't look like CAMB lensedCls: need columns [ell, TT, EE, BB, TE].")

    ell = data[:, 0].astype(int)
    TT = data[:, 1]
    EE = data[:, 2]
    BB = data[:, 3]
    TE = data[:, 4]

    # clamp to 2..lmax (we keep indices aligned to ell value)
    m = (ell >= 0) & (ell <= lmax)
    ell = ell[m]
    out["TT"][ell] = TT[m]
    out["EE"][ell] = EE[m]
    out["BB"][ell] = BB[m]
    out["TE"][ell] = TE[m]
    return out


def load_bpwins(coupling_file):
    bp_win = np.load(coupling_file)
    return bp_win["bp_win"]


def binned_theory_from_unbinned(clth, bpw_mat, lb, lmax):
    modes = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    clth_vec = np.concatenate([clth[mode] for mode in modes]).reshape(len(modes), -1)
    msk = np.arange(bpw_mat.shape[-1]) <= lmax
    bpw_mat_crop = bpw_mat[:, :, :, msk]
    clth_binned = np.einsum("ijkl,kl->ij", bpw_mat_crop, clth_vec)
    cl_out = {m: clth_binned[i] for i, m in enumerate(modes)}
    return cl_out


def multipole_min_from_tf(tf_file, field_pairs, snr_cut=3.0):
    tf = np.load(tf_file)
    idx_bad_tf = {}
    for fp in field_pairs:
        name = f"{fp}_to_{fp}"
        snr = tf[name] / tf[f"{name}_std"]
        idx = np.where(snr < snr_cut)[0]
        idx_bad_tf[fp] = idx.max() if idx.size > 0 else 0
    return idx_bad_tf


def plot_spectrum(lb, cb, cb_err, title, ylabel, xlim,
                  cb_data=None, cb_data_err=None, add_theory=False,
                  lth=None, clth=None, cbth=None, save_file=None):
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
    offset = 0 if cb_data is None else 1

    if add_theory and (lth is not None) and (clth is not None):
        fac_th = lth * (lth + 1) / (2 * np.pi)
        main.plot(lth, fac_th * clth, c="darkgray", ls="-.", lw=2.0, label="theory")

    main.errorbar(
        lb - offset, fac * cb, yerr=fac * cb_err, marker="o", ls="None",
        markerfacecolor="white", markeredgecolor="navy", label="sims" if cb_data is not None else "data",
        elinewidth=1.5, ecolor="navy", markeredgewidth=1.5
    )

    if cb_data is not None:
        main.errorbar(
            lb + offset, fac * cb_data, yerr=fac * cb_data_err, marker="o",
            ls="None", markerfacecolor="white", markeredgecolor="darkorange",
            elinewidth=1.5, ecolor="darkorange", markeredgewidth=1.5,
            label="data"
        )

    main.legend(fontsize=12)
    main.set_xlim(*xlim)

    if add_theory and (cbth is not None):
        sub.axhspan(-3, 3, color="gray", alpha=0.1)
        sub.axhspan(-2, 2, color="gray", alpha=0.25)
        sub.axhspan(-1, 1, color="gray", alpha=0.5)

        sub.plot(
            lb - offset, (cb - cbth) / cb_err, marker="o", ls="None",
            markerfacecolor="white", markeredgecolor="navy", markeredgewidth=1.5
        )
        if cb_data is not None and cb_data_err is not None:
            sub.plot(
                lb + offset, (cb_data - cbth) / cb_data_err,
                marker="o", ls="None",
                markerfacecolor="white", markeredgecolor="darkorange",
                markeredgewidth=1.5
            )

        sub.set_xlim(*xlim)
        sub.set_ylim(-4.5, 4.5)

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()
    plt.close()


def main(args):
    """
    Read spectra + covariance from SACC and make plots.
    Works in two modes automatically detected:
      • data-only (analytic covariance): uses saccs/cl_and_cov_sacc.fits
      • sims present: averages cl_and_cov_sacc_*.fits and overlays data
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    out_dir = meta.output_directory
    sacc_dir = f"{out_dir}/saccs"
    couplings_dir = f"{out_dir}/couplings"
    if "couplings_directory" in meta.covariance:
        if meta.covariance["couplings_directory"] is not None:
            couplings_dir = meta.covariance["couplings_directory"]

    plot_dir = f"{out_dir}/plots/sacc_spectra"
    BBmeta.make_dir(plot_dir)

    # Binning
    binning = np.load(meta.binning_file)
    nmt_binning = nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)
    lb = nmt_binning.get_effective_ells()
    lmax = meta.lmax

    # Beams for theory curves
    beams = {ms: meta.read_beam(ms)[1][:lmax + 1] for ms in meta.map_sets_list}

    field_pairs = [m1 + m2 for m1, m2 in product("TEB", repeat=2)]
    ps_names = meta.get_ps_names_list(type="all", coadd=True)
    types = {"T": "0", "E": "e", "B": "b"}

    # Detect per-sim SACC files (if any)
    sim_files = sorted(glob.glob(f"{sacc_dir}/cl_and_cov_sacc_*.fits"))
    Nsims = len(sim_files)

    # Load TF SNR cut (not used directly in plots, but kept for completeness)
    idx_bad_tf = {}
    transfer_dir = meta.transfer_settings["transfer_directory"]
    for ftag1, ftag2 in meta.get_independent_filtering_pairs():
        idx = multipole_min_from_tf(
            f"{transfer_dir}/transfer_function_{ftag1}_x_{ftag2}.npz",
            field_pairs=field_pairs,
            snr_cut=3.0,
        )
        idx_bad_tf[ftag1, ftag2] = idx

    # Bandpower windows
    bpw_mats = {}
    for ms1, ms2 in ps_names:
        bpw_file = f"couplings_{ms1}_{ms2}.npz"
        bpw_mats[ms1, ms2] = load_bpwins(f"{couplings_dir}/{bpw_file}")

    # Theory
    plot_sims = {
        (ms1, ms2, fp): {
            "x": None,
            "y": [],
            "err": None,
            "x_th": None,
            "y_th": None,
            "th_binned": None,
            "title": None,
            "ylabel": None,
        }
        for ms1, ms2 in ps_names
        for fp in field_pairs
    }

    # --- THEORY SETUP ---
    fiducial_cmb   = meta.covariance.get("fiducial_cmb", None)
    fiducial_dust  = meta.covariance.get("fiducial_dust", None)
    fiducial_synch = meta.covariance.get("fiducial_synch", None)

    def _path_exists(x):
        try:
            return isinstance(x, (str, os.PathLike)) and os.path.exists(x)
        except Exception:
            return False

    have_cmb = _path_exists(fiducial_cmb)
    have_dust = isinstance(fiducial_dust, str) and ("{nu1}" in fiducial_dust and "{nu2}" in fiducial_dust)
    have_syn = isinstance(fiducial_synch, str) and ("{nu1}" in fiducial_synch and "{nu2}" in fiducial_synch)

    # Beams only needed if we're going to draw theory
    beams = {ms: meta.read_beam(ms)[1][:lmax+1] for ms in meta.map_sets_list} if (have_cmb or have_dust or have_syn) else None

    for ms1, ms2 in ps_names:
        nu1 = meta.freq_tag_from_map_set(ms1)
        nu2 = meta.freq_tag_from_map_set(ms2)

        clth = None
        if have_cmb and beams is not None:
            # load CMB from either FITS or CAMB .dat
            cmb = load_fiducial_cmb_any(fiducial_cmb, lmax)

            clth = {}
            for fp in ["TT","EE","BB","TE"]:
                cl = cmb[fp].copy()
                # add dust/sync if provided (per-frequency files)
                if have_dust:
                    df = fiducial_dust.format(nu1=nu1, nu2=nu2)
                    if os.path.exists(df):
                        dspec = hp.read_cl(df)[:, :lmax+1]
                        # dspec rows assumed [TT,EE,BB,TE]
                        idx = {"TT":0,"EE":1,"BB":2,"TE":3}[fp]
                        cl += dspec[idx]
                if have_syn:
                    sf = fiducial_synch.format(nu1=nu1, nu2=nu2)
                    if os.path.exists(sf):
                        sspec = hp.read_cl(sf)[:, :lmax+1]
                        idx = {"TT":0,"EE":1,"BB":2,"TE":3}[fp]
                        cl += sspec[idx]
                # apply beams for display
                cl *= beams[ms1] * beams[ms2]
                clth[fp] = cl

            clth["EB"] = np.zeros(lmax+1)
            clth["TB"] = np.zeros(lmax+1)
            clth["l"]  = np.arange(lmax+1)

        # Bin theory (if available)
        cl_th = {}
        clth_binned = None
        if clth is not None:
            for fp in field_pairs:
                cl_th[fp] = clth[fp] if fp in clth else clth[fp[::-1]]
            clth_binned = binned_theory_from_unbinned(cl_th, bpw_mats[(ms1, ms2)], lb, lmax)

        for fp in field_pairs:
            if clth is not None:
                mask_th = clth["l"] <= lmax
                x_th = clth["l"][mask_th]
                y_th = cl_th[fp][mask_th]
                mask = lb <= lmax
                th_binned = clth_binned[fp][mask]
                plot_sims[ms1, ms2, fp]["x_th"] = x_th
                plot_sims[ms1, ms2, fp]["y_th"] = y_th
                plot_sims[ms1, ms2, fp]["th_binned"] = th_binned
            else:
                plot_sims[ms1, ms2, fp]["x_th"] = None
                plot_sims[ms1, ms2, fp]["y_th"] = None
                plot_sims[ms1, ms2, fp]["th_binned"] = None


    # Load data SACC (if present)
    plot_data = {(ms1, ms2, fp): {"x": None, "y": None, "err": None}
                 for ms1, ms2 in ps_names for fp in field_pairs}
    fname_data = f"{sacc_dir}/cl_and_cov_sacc.fits"
    if os.path.isfile(fname_data):
        s = sacc.Sacc.load_fits(fname_data)
        for ms1, ms2 in ps_names:
            for fp in field_pairs:
                f1, f2 = fp
                ell, cl, cov = s.get_ell_cl(f"cl_{types[f1]}{types[f2]}",
                                            ms1, ms2, return_cov=True)
                mask = ell <= lmax
                x, y, err = ell[mask], cl[mask], np.sqrt(cov.diagonal())[mask]
                plot_data[ms1, ms2, fp]["x"] = x
                plot_data[ms1, ms2, fp]["y"] = y
                plot_data[ms1, ms2, fp]["err"] = err
    else:
        print(f"[plot_sacc] No data SACC at {fname_data}; proceeding with sims-only if available.")

    # Load simulation SACCs if any
    if Nsims > 0:
        for id_sim, sim_path in enumerate(sim_files):
            if verbose:
                print(f"# sim {id_sim}: {sim_path}")
            s = sacc.Sacc.load_fits(sim_path)
            for ms1, ms2 in ps_names:
                for fp in field_pairs:
                    f1, f2 = fp
                    ell, cl, cov = s.get_ell_cl(f"cl_{types[f1]}{types[f2]}",
                                                ms1, ms2, return_cov=True)
                    mask = ell <= lmax
                    x, y, err = ell[mask], cl[mask], np.sqrt(cov.diagonal())[mask]
                    plot_sims[ms1, ms2, fp]["x"] = x
                    plot_sims[ms1, ms2, fp]["y"].append(y)
                    plot_sims[ms1, ms2, fp]["err"] = err
                    plot_sims[ms1, ms2, fp]["title"] = f"{ms1} x {ms2} - {fp}"
                    plot_sims[ms1, ms2, fp]["ylabel"] = fp

    # Make plots
    for ms1, ms2 in ps_names:
        for fp in field_pairs:
            plot_name = f"plot_cells_{ms1}_{ms2}_{fp}.png"
            title = f"{ms1} x {ms2} - {fp}"
            ylabel = fp

            if Nsims > 0 and plot_sims[ms1, ms2, fp]["y"]:
                y_sim = np.mean(plot_sims[ms1, ms2, fp]["y"], axis=0)
                err_sim = plot_sims[ms1, ms2, fp]["err"] / np.sqrt(Nsims)
                x_sim = plot_sims[ms1, ms2, fp]["x"]
                plot_spectrum(
                    x_sim, y_sim, err_sim,
                    cb_data=plot_data[ms1, ms2, fp]["y"],
                    cb_data_err=plot_data[ms1, ms2, fp]["err"],
                    title=title, ylabel=ylabel, xlim=(2, lmax),
                    add_theory=True,
                    lth=plot_sims[ms1, ms2, fp]["x_th"],
                    clth=plot_sims[ms1, ms2, fp]["y_th"],
                    cbth=plot_sims[ms1, ms2, fp]["th_binned"],
                    save_file=f"{plot_dir}/{plot_name}",
                )
            else:
                # data-only mode (analytic covariance typical)
                x = plot_data[ms1, ms2, fp]["x"]
                y = plot_data[ms1, ms2, fp]["y"]
                err = plot_data[ms1, ms2, fp]["err"]
                if x is None or y is None or err is None:
                    # nothing to plot for this pair
                    continue
                plot_spectrum(
                    x, y, err,
                    cb_data=None, cb_data_err=None,
                    title=title, ylabel=ylabel, xlim=(2, lmax),
                    add_theory=True,
                    lth=plot_sims[ms1, ms2, fp]["x_th"],
                    clth=plot_sims[ms1, ms2, fp]["y_th"],
                    cbth=plot_sims[ms1, ms2, fp]["th_binned"],
                    save_file=f"{plot_dir}/{plot_name}",
                )

    print(f"[plot_sacc] Plots written to: {plot_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SACC spectra plotter')
    parser.add_argument("--globals", type=str, required=True,
                        help="Path to the global configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
