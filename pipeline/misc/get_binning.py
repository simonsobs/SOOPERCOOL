from soopercool import BBmeta
import numpy as np
from soopercool.utils import create_binning
import argparse


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    binning_dir = f"{out_dir}/binning"
    BBmeta.make_dir(binning_dir)

    bin_low, bin_high, bin_center = create_binning(meta.nside,
                                                   args.deltal)
    print(bin_low, bin_high, bin_center)
    file_name = f"binning_nside{meta.nside}_deltal{args.deltal}.npz"

    bin_low2, bin_high2, bin_center2 = create_binning(meta.nside,
                                                      args.deltal,
                                                      end_first_bin=30)
    print(bin_low2, bin_high2, bin_center2)

    np.savez(
        f"{binning_dir}/{file_name}",
        bin_low=bin_low2,
        bin_high=bin_high2,
        bin_center=bin_center2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--deltal", type=int,
                        help="Delta ell for the binning.")
    args = parser.parse_args()
    main(args)
