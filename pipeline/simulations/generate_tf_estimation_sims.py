import argparse
from soopercool import BBmeta, utils
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from soopercool import sim_utils
from pixell import enplot
import numpy as np


def main(args):
    """
    This script generates transfer function estimation simulations as
    used in Hervias et al. 2025.

    These are Gaussian simulations with a power-law power spectrum that come in
    three flavors (pureT, E, B) which have 100% correlated alms for each
    realization.

    Most simulation parameters are read from the yaml file, section
    'transfer_settings'. Required arguments are:
    - tf_est_num_sims: the number of realizations to be generated
    - power_law_pars_tf_est: the power-law parameters for the simulations
    - unfiltered_map_dir: the directory where to save simulations for each
      filtering_tag (as defined under 'map_sets')
    - unfiltered_map_template: the string naming template to save simulations
      under for each filtering_tag (as defined under 'map_sets'). Accepts
      fstring placeholders {sim_id} (an integer) and {pure_type} (will loop
      over 'pureT', 'pureE', 'pureB')
    
    Additional parameters required are:
    - masks['analysis_mask']: serves as geometry template for the simulations
    - general_pars['pix_type']: pixelization (car or hp)
    - general_pars['car_template']: (if using car) CAR geometry template
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    do_plots = not args.no_plots

    tf_settings = meta.transfer_settings
    sim_dirs = {None: list(tf_settings["unfiltered_map_dir"].values())[0]}
    sim_templates = {None: list(tf_settings["unfiltered_map_template"].values())[0]}
    beams = {None: None}

    lmax = meta.lmax
    lmax_sim = lmax + 500
    lth = np.arange(lmax_sim + 1)
    Nsims = tf_settings["tf_est_num_sims"]

    if tf_settings["tf_est_beams_list"]:
        sim_dirs = {
            beam: tf_settings.unfiltered_map_dir[beam]
            for beam in tf_settings["beam_list"]
        }
        sim_templates = {
            beam: tf_settings.unfiltered_map_template[beam]
            for beam in tf_settings["beam_list"]
        }
        beams = {}
        for beam_label in tf_settings["tf_est_beams_list"]:
            _, bl = meta.read_beam(beam_label, lmax=lmax_sim)
            beams[beam_label] = bl

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)

    cl_power_law_tf_est = utils.power_law_cl(
        lth, **tf_settings["power_law_pars_tf_est"]
    )
    for sim_dir in sim_dirs.values():
        BBmeta.make_dir(sim_dir)
        np.savez(f"{sim_dir}/cl_power_law_tf_est.npz",
                 ell=lth, **cl_power_law_tf_est)
        
    template = mu.template_from_map(mask, ncomp=3, pix_type=meta.pix_type)
    mpi.init(True)

    for id_sim in mpi.taskrange(Nsims - 1):
        almsTEB = sim_utils.get_alms_from_cls(
            ps_dict=cl_power_law_tf_est,
            lmax=lmax_sim,
            fields="TEB",
            components=None
        )

        for beam_label, bl in beams.items():
            if verbose:
                print(f"  # {id_sim} | {beam_label}")

            if beam_label is not None:
                almsTEB_post = sim_utils.beam_alms(
                    almsTEB.copy(),
                    bl
                )
            else:
                almsTEB_post = almsTEB

            sims = {
                f"pure{f}": sim_utils.get_map_from_alms(
                    almsTEB_post * select[:, None],
                    template=template
                ) for f, select in zip("TEB", np.eye(3))
            }

            for f in "TEB":
                fdir = sim_dirs[beam_label]
                fname = sim_templates[beam_label].format(
                    pure_type=f"pure{f}",
                    id_sim=id_sim
                )
                mu.write_map(
                    f"{fdir}/{fname}",
                    sims[f"pure{f}"],
                    pix_type=meta.pix_type
                )
                if do_plots:
                    for i, mode in zip([0, 1, 2], "TQU"):
                        plot = enplot.plot(sims[f"pure{f}"][i],
                                        ticks=10,
                                        color="planck")
                        plot_fn = f"{'.'.join(fname.split(".")[:-1])}_{mode}"
                        enplot.write(f"{fdir}/{plot_fn}", plot)
                        if verbose:
                            print(f"  PLOT {fdir}/{plot_fn}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pureT/E/B simulations \
                     for transfer function estimation")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--no-plots", action="store_true",
                        help="Pass to generate plots")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
