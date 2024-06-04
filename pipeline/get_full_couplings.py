import argparse
from soopercool import BBmeta
import numpy as np
import pymaster as nmt
from soopercool import coupling_utils as cu


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    couplings_dir = f"{out_dir}/couplings"

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    tf_settings = meta.transfer_settings

    ps_names = meta.get_ps_names_list("all", coadd=True)
    filtering_pairs = meta.get_independent_filtering_pairs()

    tf_dir = tf_settings["transfer_directory"]

    tf_dict = {}
    for ftag1, ftag2 in filtering_pairs:
        if ftag1 is None and ftag2 is None:
            tf = np.zeros((9, 9, n_bins))
            tf[np.arange(9), np.arange(9), :] = 1.
            tf_dict[ftag1, ftag2] = tf
        else:
            tf = np.load(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz")
            tf_dict[ftag1, ftag2] = tf["full_tf"]

    mcms_dict = cu.load_mcms(couplings_dir,
                             ps_names=ps_names, full_mcm=True)

    # First for the data (i.e. including beams, tf, and mcm)
    ps_names_and_ftags = {
        (ms1, ms2): (meta.filtering_tag_from_map_set(ms1),
                     meta.filtering_tag_from_map_set(ms2))
        for ms1, ms2 in ps_names
    }

    couplings = cu.get_couplings_dict(
        mcms_dict, nmt_bins,
        transfer_dict=tf_dict,
        ps_names_and_ftags=ps_names_and_ftags
    )

    for ms1, ms2 in ps_names:
        np.savez(
            f"{couplings_dir}/couplings_{ms1}_{ms2}.npz",
            **couplings[ms1, ms2]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--unity", action="store_true")
    args = parser.parse_args()
    main(args)
