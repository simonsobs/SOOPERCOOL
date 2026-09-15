import healpy as hp
import numpy as np
import pymaster as nmt
import os
from soopercool import BBmeta
from soopercool import map_utils as mu
from soopercool import sim_utils as su
import soopercool.utils as ut
import matplotlib.pyplot as plt


def main(args):
    """
    Generate CMB noiseless simulations of the map geometry provided by the
    soopercool yaml.

    Requires masks["analysis_mask"] from SOOPERCOOL yaml as a geometry
    template. Loops over all beams in transfer_settings['tf_val_beams_list'];
    if the list in empty, applies 30-arcmin Gaussian beam and low-pass at
    ell=650.

    Important command line arguments are:
    - "--out_dir": output directory for sims
    - "--n_sims": number of simulations to be generated
    - "--sim_id_start": first sim ID (defaults to 0)

    Saves simulations under
    "{out_dir}/cmb_{res_label}_{beamlab}_sim{id_sim:04d}_{CAR}.fits".
    Optionally plots decoupled sim CLs to "{out_dir}/sim_pcls.pdf".
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    pix_type = meta.pix_type
    lmax = meta.lmax
    nside = None if pix_type == "car" else meta.nside

    do_plots = not args.no_plots
    n_sims = args.n_sims
    id_start = args.sim_id_start

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        raise ValueError(f"Directory does not exist: {out_dir}")
    tf_settings = meta.transfer_settings
    beams = {None: None}

    if tf_settings["tf_est_beams_list"]:
        beams = {}
        for beam_label in tf_settings["tf_est_beams_list"]:
            _, bl = meta.read_beam(beam_label, lmax=lmax)
            beams[beam_label] = bl

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)
    shape, wcs = (mask.shape, mask.wcs)
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

    _, clth = ut.get_theory_cls(cosmo_params=cosmo, lmax=lmax)

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
        for beam_label, bl in beams.items():
            if verbose:
                print(f"  # {id_sim} | {beam_label}")

            if beam_label is not None:
                almsTEB_post = su.beam_alms(almsTEB.copy(), bl)
            else:
                almsTEB_post = almsTEB
            sim = su.get_map_from_alms(almsTEB_post, template=template)
            beamlab = beam_label
            if beam_label is None:
                beamlab = "nobeam"
            res_label = f"nside{nside}"
            if pix_type == "car":
                res_arcmin = np.min(np.abs(wcs.wcs.cdelt))*60.
                res_label = f"{int(res_arcmin):2d}arcmin"
            fn_sim = f"{out_dir}/cmb_{res_label}_{beamlab}_sim{id_sim:04d}_{pix_type.upper()}.fits"  # noqa: E501
            mu.write_map(fn_sim, sim, pix_type=pix_type)
            if do_plots:
                mask_ones = np.ones(shape)
                f = nmt.NmtField(mask_ones, sim[-2:], spin=2, wcs=wcs,
                                 lmax=lmax)
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
        plt.savefig(f"{out_dir}/sim_pcls.pdf", bbox_inches="tight")
        if verbose:
            print(f"Plot saved to {out_dir}/sim_pcls.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--globals", help="Path to the global parameter file."
    )
    parser.add_argument(
        "--out_dir", help="Output directory"
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        help="Number of simulations"
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
