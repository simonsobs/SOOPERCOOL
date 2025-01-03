import argparse
from soopercool import BBmeta
from soopercool import coupling_utils as cu
import numpy as np


def main(args):
    """ """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    tf_dir = f"{out_dir}/transfer_functions"
    BBmeta.make_dir(tf_dir)

    cells_dir = f"{out_dir}/cells_tf_est"

    tf_settings = meta.transfer_settings

    filtering_pairs = meta.get_independent_filtering_pairs()

    pcls_mat_dict = cu.read_pcls_matrices(
        cells_dir, filtering_pairs, tf_settings["tf_est_num_sims"]
    )

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = cu.average_pcls_matrices(
        pcls_mat_dict, filtering_pairs, filtered=True
    )
    pcls_mat_unfiltered_mean = cu.average_pcls_matrices(
        pcls_mat_dict, filtering_pairs, filtered=False
    )

    # Compute and save the transfer functions
    trans = cu.get_transfer_dict(
        pcls_mat_filtered_mean, pcls_mat_unfiltered_mean, pcls_mat_dict, filtering_pairs
    )

    full_tf = {}
    for ftag1, ftag2 in filtering_pairs:
        tf = trans[ftag1, ftag2]
        np.savez(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz", **tf)
        full_tf[ftag1, ftag2] = tf["full_tf"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    args = parser.parse_args()
    main(args)
