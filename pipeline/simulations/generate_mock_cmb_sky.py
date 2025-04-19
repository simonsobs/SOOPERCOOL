import argparse
from soopercool import BBmeta
from soopercool import utils
from soopercool import map_utils as mu
import soopercool.utils as su
import numpy as np
from soopercool import sim_utils


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    # verbose = args.verbose
    out_dir = meta.output_directory

    sims_dir = f"{out_dir}/cmb_sims"
    plot_dir = f"{out_dir}/plots/cmb_sims"
    BBmeta.make_dir(sims_dir)
    BBmeta.make_dir(plot_dir)

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)
    lmax = mu.lmax_from_map(mask, pix_type=meta.pix_type)
    lmax_sim = lmax + 500

    # Create the CMB fiducial cl
    lth, clth = utils.get_theory_cls(
        meta.covariance["cosmo"],
        lmax=lmax_sim  # ensure cl accuracy up to lmax
    )
    np.savez(f"{sims_dir}/cl_theory.npz",
             l=lth, **clth)

    beams = {
        ms: su.read_beam_from_file(
            f"{meta.beam_dir_from_map_set(ms)}/"
            f"{meta.beam_file_from_map_set(ms)}",
            lmax=lmax_sim
        )[1]
        for ms in meta.map_sets_list
    }

    template = mu.template_from_map(mask, ncomp=3, pix_type=meta.pix_type)
    for id_sim in range(meta.covariance["cov_num_sims"]):

        meta.timer.start("cmb_sim")

        alms = sim_utils.get_alms_from_cls(
            ps_dict=clth,
            lmax=lmax_sim,
            fields="TEB",
            components=None
        )
        for ms in meta.map_sets_list:

            alms_beamed = sim_utils.beam_alms(
                alms,
                beams[ms]
            )

            map = sim_utils.get_map_from_alms(
                alms_beamed,
                template=template
            )

            mu.write_map(
                f"{sims_dir}/cmb_{ms}_{id_sim:04d}.fits",
                map,
                pix_type=meta.pix_type)

        meta.timer.stop(
            "cmb_sim",
            text_to_output=f"Simulated CMB maps for sim {id_sim:04d}"
        )

    # Plot functions, not sure if we want to add it
    # from pixell import enplot
    # for i, f in enumerate("TQU"):
    #     plot = enplot.plot(maps["sat_f093"][i], color="planck", ticks=10)
    #     enplot.write(f"{plot_dir}/cmb_sat_f093_{f}", plot)

    # hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    # for id_sim in range(meta.covariance["cov_num_sims"]):
    #     alms = hp.synalm([clth[fp] for fp in hp_ordering])
    #     for ms in meta.map_sets_list:
    #         if verbose:
    #             print(f"# {id_sim+1} | {ms}")

    #         alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]

    #         map = hp.alm2map(alms_beamed, nside=meta.nside)
    #         mu.write_map(f"{sims_dir}/cmb_{ms}_{id_sim:04d}.fits", map,
    #                      dtype=np.float32)
    #         hp.write_cl(f"{sims_dir}/cl_{ms}_{id_sim:04d}.fits",
    #                     hp.anafast(map), overwrite=True, dtype=np.float32)

    # # Plotting
    # cls = {ms: [] for ms in meta.map_sets_list}
    # for ms in meta.map_sets_list:
    #     for id_sim in range(meta.covariance["cov_num_sims"]):
    #         cls[ms].append(
    #             hp.read_cl(f"{sims_dir}/cl_{ms}_{id_sim:04d}.fits")
    #         )
    # cl_mean = {ms: np.mean(np.array(cls[ms]), axis=0)
    #            for ms in meta.map_sets_list}
    # cl_std = {ms: np.std(np.array(cls[ms]), axis=0)
    #           for ms in meta.map_sets_list}

    # ll = np.arange(lmax_sim + 1)
    # cl2dl = ll*(ll + 1)/2./np.pi
    # for ms in meta.map_sets_list:
    #     for ip, fp in enumerate(["TT", "EE", "BB", "TE"]):
    #         plt.title(f"{ms} | {fp}")
    #         plt.errorbar(ll, cl2dl*cl_mean[ms][ip], cl2dl*cl_std[ms][ip],
    #                      label="data", color="navy", lw=0.5, zorder=-32)
    #         plt.plot(ll, cl2dl*clth[fp], label=" theory",
    #                  color="k", ls="--")
    #         plt.plot(ll, cl2dl*clth[fp]*beams[ms]**2, label=" beamed theory",
    #                  color="darkorange", ls="--")
    #         plt.ylabel(r"$D_\ell$")
    #         plt.xlabel(r"$\ell$")
    #         plt.yscale("log")
    #         plt.legend()
    #         plt.savefig(f"{plot_dir}/cl_{ms}_{fp}.png")
    #         plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
