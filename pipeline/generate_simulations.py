import argparse
from soopercool import BBmeta
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    do_plots = not args.no_plots

    out_dir = meta.output_directory
    masks_dir = f"{out_dir}/masks"

    sims_dir = f"{out_dir}/cov_sims"
    BBmeta.make_dir(sims_dir)

    noise_map_dir = meta.covariance["noise_sims_dir"]
    signal_map_dir = meta.covariance["signal_sims_dir"]

    noise_map_template = meta.covariance["noise_sims_template"]
    signal_map_template = meta.covariance["signal_sims_template"]

    binary = mu.read_map(f"{masks_dir}/binary_mask.fits")

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):

        base_dir = f"{sims_dir}/{id_sim:04d}"
        BBmeta.make_dir(base_dir)
        for ms in meta.map_sets_list:
            if verbose:
                print(f"# {id_sim} | {ms}")

            fname = signal_map_template.format(id_sim=id_sim, map_set=ms)
            cmb = mu.read_map(f"{signal_map_dir}/{fname}", field=[0, 1, 2])
            for id_bundle in range(meta.n_bundles_from_map_set(ms)):

                fname = noise_map_template.format(id_sim=id_sim, map_set=ms,
                                                  id_bundle=id_bundle)
                noise = mu.read_map(f"{noise_map_dir}/{fname}", field=[0, 1, 2])

                split_map = cmb + noise

                map_name = f"cov_sims_{ms}_bundle{id_bundle}.fits"
                mu.write_map(f"{base_dir}/{map_name}", split_map*binary,
                             dtype=np.float32)

                if do_plots:
                    import healpy as hp

                    for i, f in enumerate("TQU"):
                        hp.mollview(split_map[i]*binary,
                                    cmap="RdYlBu_r",
                                    title=f"{ms} - {id_sim} - {f}",
                                    min=-300 if f == "T" else -100,
                                    max=300 if f == "T" else 100)
                        plt.savefig(f"{base_dir}/cov_sims_{ms}_bundle{id_bundle}_{f}.png") # noqa
                        plt.clf()
                        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no-plots", action="store_true", help="Do not plot the maps.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
