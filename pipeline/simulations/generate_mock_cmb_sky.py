import argparse
from soopercool import BBmeta
import healpy as hp
from soopercool import utils
import numpy as np


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory

    sims_dir = f"{out_dir}/cmb_sims"
    BBmeta.make_dir(sims_dir)

    lmax_sim = 3*meta.nside - 1

    # Create the CMB fiducial cl
    lth, clth = utils.get_theory_cls(
        meta.covariance["cosmo"],
        lmax=lmax_sim  # ensure cl accuracy up to lmax
    )
    np.savez(f"{sims_dir}/cl_theory.npz",
             l=lth, **clth)

    beams = {
        ms: meta.read_beam(ms)[1]
        for ms in meta.map_sets_list
    }

    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    for id_sim in range(meta.covariance["cov_num_sims"]):
        alms = hp.synalm([clth[fp] for fp in hp_ordering])
        for ms in meta.map_sets_list:

            alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]

            map = hp.alm2map(alms_beamed, nside=meta.nside)

            import matplotlib.pyplot as plt
            hp.mollview(map[1], title=f"{ms} - {id_sim}")
            plt.show()

            hp.write_map(f"{sims_dir}/cmb_{ms}_{id_sim:04d}.fits", map,
                         overwrite=True, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    args = parser.parse_args()
    main(args)
