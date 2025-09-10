import argparse
from soopercool import BBmeta
from soopercool import coupling_utils as cu
from soopercool import utils
import numpy as np


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_tf_est"
    do_plot = not args.no_plots if args.no_plots is not None else False

    tf_settings = meta.transfer_settings
    tf_dir = tf_settings["transfer_directory"]
    BBmeta.make_dir(tf_dir)

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()

    filtering_pairs = meta.get_independent_filtering_pairs()

    pcls_mat_dict = cu.read_pcls_matrices(
        cells_dir, filtering_pairs,
        tf_settings["tf_est_num_sims"],
        tf_settings["sim_id_start"]
    )

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = cu.average_pcls_matrices(
        pcls_mat_dict,
        filtering_pairs,
        filtered=True
    )
    pcls_mat_unfiltered_mean = cu.average_pcls_matrices(
        pcls_mat_dict,
        filtering_pairs,
        filtered=False
    )

    # Compute and save the transfer functions
    trans = cu.get_transfer_dict(
        pcls_mat_filtered_mean,
        pcls_mat_unfiltered_mean,
        pcls_mat_dict,
        filtering_pairs
    )

    full_tf = {}
    for ftag1, ftag2 in filtering_pairs:
        print(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz")
        tf = trans[ftag1, ftag2]
        np.savez(
            f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz",
            **tf
        )
        full_tf[ftag1, ftag2] = tf["full_tf"]

    if do_plot:
        import os
        plot_dir = "/".join(os.path.split(tf_dir)[:-1]) + "/plots/transfer_functions"  # noqa
        BBmeta.make_dir(plot_dir)

        for ftag1, ftag2 in filtering_pairs:
            tf_dict = np.load(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz")  # noqa

            utils.plot_transfer_function(
                lb, tf_dict, meta.lmin, meta.lmax,
                ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"],
                file_name=f"{plot_dir}/transfer_{ftag1}_x_{ftag2}.pdf"
            )
        print(plot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not plot transfer function.")
    args = parser.parse_args()
    main(args)
