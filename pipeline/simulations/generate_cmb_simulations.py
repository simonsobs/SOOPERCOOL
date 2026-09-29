import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import soopercool.utils as ut
from soopercool import BBmeta
from soopercool import map_utils as mu
from soopercool import sim_utils as su


def main(args):
    """
    Generate CMB noiseless simulations of the map geometry provided by the
    soopercool yaml.

    Requires masks["analysis_mask"] from SOOPERCOOL yaml as a geometry
    template. Applies a 30-arcmin Gaussian beam.

    Important command line arguments are:
    - "--out_dir": output directory for sims. Defaults to
                   f"{out_dir}/sims_tf_val"
    - "--n_sims": number of simulations to be generated
    - "--sim_id_start": first sim ID (defaults to 0)

    Saves simulations under
    "{out_dir}/cmb_{res_label}_fwhm30_sim{id_sim:04d}_{CAR}.fits".
    Optionally plots decoupled sim CLs to "{out_dir}/sim_pcls.pdf".
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    pix_type = meta.pix_type
    lmax = meta.lmax
    nside = None if pix_type == "car" else meta.nside

    do_plots = not args.no_plots
    if args.n_sims is None:
        n_sims = meta.transfer_settings["tf_val_num_sims"]
    else:
        n_sims = args.n_sims
    id_start = args.sim_id_start

    if args.out_dir is None:
        tf_val_dir = f"{meta.output_directory}/sims_tf_val"
        BBmeta.make_dir(tf_val_dir)
    else:
        if not os.path.isdir(args.out_dir):
            raise ValueError(f"Directory does not exist: {args.out_dir}")
        tf_val_dir = args.out_dir

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)
    shape, wcs = meta.get_geometry()
    template = mu.template_from_map(mask, ncomp=3, pix_type=meta.pix_type)

    cosmo = {
       "cosmomc_theta": 0.0104085,
       "As": 2.1e-9,
       "ombh2": 0.02237,
       "omch2": 0.1200,
       "ns": 0.9649,
       "Alens": 1.0,
       "tau": 0.0544,
       "r": 0.0,
    }

    # NOTE: The hardcoded choice is a 30-arcminute Gaussian beam. We anticipate
    # this choice to not affect the conclusions on the TF validation w.r.t.
    # wider beams, but we should check explicitly when adding LF channels.
    lth, clth = ut.get_theory_cls(cosmo_params=cosmo, lmax=lmax, fwhm_amin=30.)
    np.savez(f"{tf_val_dir}/cl_cmb_fwhm30.fits", l=lth, **clth)

    if do_plots:
        ls = np.arange(2, lmax+1)
        b = nmt.NmtBin.from_lmax_linear(lmax, 1)
        f = nmt.NmtField(np.ones(shape), None, wcs=wcs, spin=2, lmax=lmax)
        wsp = nmt.NmtWorkspace.from_fields(f, f, b)
        cls = []

    for id_sim in range(id_start, id_start+n_sims):
        np.random.seed(id_sim)
        almsTEB = hp.synalm([clth["TT"], clth["TE"], clth["EE"], clth["BB"]],
                            lmax=lmax)
        if verbose:
            print(f"  # {id_sim} | fwhm30")

        sim = su.get_map_from_alms(almsTEB, template=template)
        res_label = f"nside{nside}"
        if pix_type == "car":
            res_arcmin = np.min(np.abs(wcs.wcs.cdelt))*60.
            res_label = f"{int(res_arcmin):2d}arcmin"
        fn_sim = f"{tf_val_dir}/cmb_{res_label}_fwhm30_sim{id_sim:04d}_{pix_type.upper()}.fits"  # noqa: E501
        mu.write_map(fn_sim, sim, pix_type=pix_type)
        if do_plots:
            mask_ones = np.ones(shape)
            f = nmt.NmtField(mask_ones, sim[1:], spin=2, wcs=wcs, lmax=lmax)
            cls += [wsp.decouple_cell(nmt.compute_coupled_cell(f, f))]
    if do_plots:
        _, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                     height_ratios=[3, 1])
        cls = np.array(cls)
        cols = ["b", "r", "darkorange", "teal"]
        for ip, pp in enumerate(["EE", "EB", "BE", "BB"]):
            ax1.errorbar(ls, np.mean(cls[:, ip], axis=0),
                         np.std(cls[:, ip], axis=0), c=cols[ip], label=pp)
            ax1.plot(ls, clth[pp][2:], ls="--", c=cols[ip], alpha=0.5)
            res = np.mean(cls[:, ip], axis=0)-clth[pp][2:]
            res /= (np.mean(cls[:, ip], axis=0)/np.sqrt(n_sims))
            ax2.plot(ls, res, ls="--", c=cols[ip])
            ax2.axhline(0, color="k")
            ax2.set_ylim(-5, 5)
        ax1.legend()
        ax1.set_ylabel(r"$C_\ell$")
        ax2.set_xlabel(r"$\ell$")
        ax1.set_yscale("log")
        plt.savefig(f"{tf_val_dir}/sim_pcls.pdf", bbox_inches="tight")
        if verbose:
            print(f"Plot saved to {tf_val_dir}/sim_pcls.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--globals", help="Path to the global parameter file."
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory. Defaults to "
        "{meta.output_directory}/sims_tf_val",
        default=None
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        help="Number of simulations. "
             "Defaults to meta.transfer_settings['tf_val_num_sims'].",
        default=None
    )
    parser.add_argument(
        "--sim_id_start",
        type=int,
        default=0,
        help="Simulation ID to start with"
    )
    parser.add_argument("--no-plots", action="store_true",
                        help="Pass to generate plots")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
