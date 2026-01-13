from soopercool import BBmeta, ps_utils, utils
import argparse
from itertools import product
import sacc
import numpy as np
import healpy as hp
import os


def main(args):
    """
    This script will read the spectra and covariance
    stored in the `sacc` files and plot the power
    spectra.
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    out_dir = meta.output_directory
    sacc_dir = f"{out_dir}/saccs"
    plot_dir = f"{out_dir}/plots/sacc_spectra"
    BBmeta.make_dir(plot_dir)
    lmax = meta.lmax

    beams = {
        ms: meta.read_beam(ms)[1][:lmax+1]
        for ms in meta.map_sets_list
    }

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]
    ps_names = meta.get_ps_names_list(type="all", coadd=True)

    types = {"T": "0", "E": "e", "B": "b"}

    Nsims = meta.covariance["cov_num_sims"]

    # Transfer function
    idx_bad_tf = {}
    bpw_mats = {}

    transfer_dir = meta.transfer_settings["transfer_directory"]
    for ftag1, ftag2 in meta.get_independent_filtering_pairs():
        tf_dir = f"{transfer_dir}/transfer_function_{ftag1}_x_{ftag2}.npz"
        if not os.path.isfile(tf_dir):
            tf_dir = f"{transfer_dir}/transfer_function_{ftag2}_x_{ftag1}.npz"
        idx = utils.multipole_min_from_tf(
            tf_dir,
            field_pairs=field_pairs,
            snr_cut=3
        )
        idx_bad_tf[ftag1, ftag2] = idx

    bpws = meta.get_inverse_couplings(return_bpwf=True)[1]
    for ms1, ms2 in ps_names:
        ftag1, ftag2 = (meta.filtering_tag_from_map_set(ms)
                        for ms in (ms1, ms2))
        bpw_mats[ms1, ms2] = bpws[ftag1, ftag2]

    plot_sims = {
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

    fiducial_cmb = meta.covariance["fiducial_cmb"]
    fiducial_dust = meta.covariance["fiducial_dust"]
    fiducial_synch = meta.covariance["fiducial_synch"]

    for ms1, ms2 in ps_names:
        nu1 = meta.freq_tag_from_map_set(ms1)
        nu2 = meta.freq_tag_from_map_set(ms2)

        clth = {}

        for i, fp in enumerate(["TT", "EE", "BB", "TE"]):
            clth[fp] = hp.read_cl(fiducial_cmb)[i, :lmax+1]
            if fiducial_dust is not None:
                if not os.path.isfile(fiducial_dust.format(nu1=nu1, nu2=nu2)):
                    nu1, nu2 = nu2, nu1
                clth[fp] += hp.read_cl(
                    fiducial_dust.format(nu1=nu1, nu2=nu2)
                )[i, :lmax+1]
            if fiducial_synch is not None:
                if not os.path.isfile(fiducial_synch.format(nu1=nu1, nu2=nu2)):
                    nu1, nu2 = nu2, nu1
                clth[fp] += hp.read_cl(
                    fiducial_synch.format(nu1=nu1, nu2=nu2)
                )[i, :lmax+1]
            clth[fp] *= beams[ms1]*beams[ms2]
        clth["EB"] = np.zeros(lmax+1)
        clth["TB"] = np.zeros(lmax+1)
        clth["l"] = np.arange(lmax+1)

        cl_th = {}
        for fp in field_pairs:
            if fp in clth:
                cl_th[fp] = clth[fp]
            else:
                # reverse the order of the two fields
                cl_th[fp] = clth[fp[::-1]]

        clth_binned = ps_utils.bin_theory_cls(cl_th, bpw_mats[(ms1, ms2)])

        ftag1 = meta.filtering_tag_from_map_set(ms1)
        ftag2 = meta.filtering_tag_from_map_set(ms2)

        for fp in field_pairs:
            x_th, y_th = clth["l"], cl_th[fp]
            th_binned = clth_binned[fp]

            plot_sims[ms1, ms2, fp]["x_th"] = x_th
            plot_sims[ms1, ms2, fp]["y_th"] = y_th
            plot_sims[ms1, ms2, fp]["th_binned"] = th_binned

    # Load data
    plot_data = {
        (ms1, ms2, fp): {
            "y": None,
            "err": None,
        } for ms1, ms2 in ps_names
        for fp in field_pairs
    }
    fname_data = f"{sacc_dir}/cl_and_mc_cov_sacc.fits"

    if not os.path.isfile(fname_data):
        print("No data sacc to print. Skipping...")
    else:
        s = sacc.Sacc.load_fits(fname_data)

        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                f1, f2 = fp

                ell, cl, cov = s.get_ell_cl(
                    f"cl_{types[f1]}{types[f2]}",
                    ms1,
                    ms2,
                    return_cov=True)
                mask = ell <= meta.lmax
                x, y, err = (ell, cl, np.sqrt(cov.diagonal()))
                x, y, err = (x[mask], y[mask], err[mask])

                plot_data[ms1, ms2, fp]["x"] = x
                plot_data[ms1, ms2, fp]["y"] = y
                plot_data[ms1, ms2, fp]["err"] = err

    # Load simulations
    for id_sim in range(Nsims):
        if verbose:
            print(f"# {id_sim+1} | {ms1} x {ms2}")
        if id_sim == 0:
            sacc_file = f"{sacc_dir}/cl_and_mc_cov_sacc_{id_sim:04d}.fits"
        else:
            sacc_file = f"{sacc_dir}/cl_sacc_{id_sim:04d}.fits"
        s = sacc.Sacc.load_fits(sacc_file)

        for ms1, ms2 in ps_names:
            ftag1 = meta.filtering_tag_from_map_set(ms1)
            ftag2 = meta.filtering_tag_from_map_set(ms2)

            for fp in field_pairs:
                f1, f2 = fp

                if id_sim == 0:
                    ell, cl, cov = s.get_ell_cl(
                        f"cl_{types[f1]}{types[f2]}",
                        ms1,
                        ms2,
                        return_cov=True
                    )
                else:
                    ell, cl = s.get_ell_cl(
                        f"cl_{types[f1]}{types[f2]}",
                        ms1,
                        ms2,
                        return_cov=False
                    )
                mask = ell <= meta.lmax
                x, y, err = ell, cl, np.sqrt(cov.diagonal())
                x, y, err = x[mask], y[mask], err[mask]

                plot_sims[ms1, ms2, fp]["x"] = x
                plot_sims[ms1, ms2, fp]["y"].append(y)
                plot_sims[ms1, ms2, fp]["err"] = err
                plot_sims[ms1, ms2, fp]["title"] = f"{ms1} x {ms2} - {fp}"
                plot_sims[ms1, ms2, fp]["ylabel"] = fp

    for ms1, ms2 in ps_names:
        for fp in field_pairs:

            plot_sims[ms1, ms2, fp]["y"] = np.mean(
                plot_sims[ms1, ms2, fp]["y"], axis=0
            )
            plot_sims[ms1, ms2, fp]["err"] /= np.sqrt(Nsims)
            plot_name = f"plot_cells_{ms1}_{ms2}_{fp}.pdf"

            ps_utils.plot_spectrum(
                plot_sims[ms1, ms2, fp]["x"],
                plot_sims[ms1, ms2, fp]["y"],
                plot_sims[ms1, ms2, fp]["err"],
                cb_data=plot_data[ms1, ms2, fp]["y"],
                cb_data_err=plot_data[ms1, ms2, fp]["err"],
                title=plot_sims[ms1, ms2, fp]["title"],
                ylabel=plot_sims[ms1, ms2, fp]["ylabel"],
                xlim=(2, meta.lmax),
                add_theory=True,
                lth=plot_sims[ms1, ms2, fp]["x_th"],
                clth=plot_sims[ms1, ms2, fp]["y_th"],
                cbth=plot_sims[ms1, ms2, fp]["th_binned"],
                save_file=f"{plot_dir}/{plot_name}"
            )
    print(f"  PLOTS: {plot_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sacc plotter')
    parser.add_argument("--globals", type=str,
                        help="Path to the global configuration file")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
