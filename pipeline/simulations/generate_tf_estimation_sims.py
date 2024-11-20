import argparse
from soopercool import BBmeta, utils
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from soopercool import sim_utils

import numpy as np


def main(args):
    """ """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    do_plots = not args.no_plots

    out_dir = meta.output_directory
    sim_dir = f"{out_dir}/tf_est_sims"
    BBmeta.make_dir(sim_dir)

    plot_dir = f"{out_dir}/plots/tf_est_sims"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    mask = mu.read_map(meta.masks["analysis_mask"], pix_type=meta.pix_type)
    lmax = mu.lmax_from_map(mask, pix_type=meta.pix_type)
    lmax_sim = lmax + 500
    lth = np.arange(lmax_sim + 1)

    tf_settings = meta.transfer_settings
    cl_power_law_tf_est = utils.power_law_cl(
        lth, **tf_settings["power_law_pars_tf_est"]
    )
    np.savez(f"{sim_dir}/cl_power_law_tf_est.npz", ell=lth, **cl_power_law_tf_est)

    Nsims = tf_settings["tf_est_num_sims"]

    if tf_settings["beams_list"] is not None:
        beams = {}
        for beam_label in tf_settings["beams_list"]:
            _, bl = meta.read_beam(beam_label, lmax=lmax_sim)
            beams[beam_label] = bl
    else:
        beams = {None: None}

    template = mu.template_from_map(mask, ncomp=3, pix_type=meta.pix_type)
    # from pspy import so_map
    # template = so_map.full_sky_car_template(ncomp=3, res=10).data
    mpi.init(True)

    for id_sim in mpi.taskrange(Nsims - 1):
        almsTEB = sim_utils.get_alms_from_cls(
            ps_dict=cl_power_law_tf_est, lmax=lmax_sim, fields="TEB", components=None
        )
        print(np.sum(almsTEB[0] - almsTEB[1]))
        print(almsTEB[0])
        print(almsTEB[1])
        print(almsTEB[2])

        for beam_label, bl in beams.items():
            if verbose:
                print(f"# {id_sim} | {beam_label}")

            if not tf_settings["do_not_beam_est_sims"]:
                suffix = f"_{beam_label}"
                almsTEB_post = sim_utils.beam_alms(almsTEB.copy(), bl)
            else:
                suffix = ""
                almsTEB_post = almsTEB

            sims = {
                f"pure{f}": sim_utils.get_map_from_alms(
                    almsTEB_post * select[:, None], template=template
                )
                for f, select in zip("TEB", np.eye(3))
            }

            for f in "TEB":
                fname = f"pure{f}_power_law_tf_est_{id_sim:04d}{suffix}.fits"
                mu.write_map(
                    f"{sim_dir}/{fname}", sims[f"pure{f}"], pix_type=meta.pix_type
                )

                from pixell import enplot

                for i, mode in zip([1, 2], "QU"):
                    plot = enplot.plot(sims[f"pure{f}"][i], ticks=10, color="planck")
                    enplot.write(
                        f"{plot_dir}/pure{f}_power_law_tf_est_{id_sim:04d}{suffix}_{mode}",  # noqa
                        plot,
                    )

    # if do_plots:
    #     ps_hp_order = ["TT", "EE", "BB", "TE", "EB", "TB"]
    #     for beam_label in beams:
    #         suffix = "" if beam_label is None else f"_{beam_label}"
    #         for f in "TEB":

    #             cls_dict = {fp: [] for fp in hp_ordering}

    #             for id_sim in range(Nsims):
    #                 fname = f"pure{f}_power_law_tf_est_{id_sim:04d}{suffix}"
    #                 alms = hp.map2alm(
    #                     mu.read_map(
    #                         f"{sim_dir}/{fname}.fits",
    #                         field=[0, 1, 2],
    #                         pix_type=meta.pix_type,
    #                         convert_K_to_muK=True),
    #                 )
    #                 cls = hp.alm2cl(alms)

    #                 for i, fp in enumerate(ps_hp_order):
    #                     cls_dict[fp] += [cls[i]]

    #             for fp in ps_hp_order:
    #                 cls_dict[fp] = np.mean(cls_dict[fp], axis=0)

    #             plt.figure(figsize=(10, 8))
    #             plt.xlabel(r"$\ell$", fontsize=15)
    #             plt.ylabel(r"$C_\ell$", fontsize=15)
    #             for fp in ps_hp_order:
    #                 plt.plot(cls_dict[fp], label=fp, lw=1.7)
    #             plt.yscale("symlog", linthresh=1e-6)
    #             plt.xlim(0, lmax_sim)
    #             plt.legend()
    #             plt.title(f"Power law pure{f} simulation")
    #             plt.savefig(f"{plot_dir}/power_law_pure{f}{suffix}.png",
    #                         bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pureT/E/B simulations \
                     for transfer function estimation"
    )
    parser.add_argument(
        "--globals", type=str, help="Path to the yaml with global parameters"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Pass to generate plots"
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
