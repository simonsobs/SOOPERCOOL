from soopercool import BBmeta
import numpy as np
from soopercool.utils import create_binning
import argparse
from soopercool import map_utils as mu


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    binning_dir = f"{out_dir}/binning"
    BBmeta.make_dir(binning_dir)

    mask_file = meta.masks["analysis_mask"]

    if mask_file is None:
        if meta.pix_type == "car":
            if meta.car_template is None:
                raise ValueError(
                    "CAR files: parameter file must contain either "
                    "analysis_mask or car_template."
                )
            else:
                lmax = mu.lmax_from_map(meta.car_template,
                                        pix_type=meta.pix_type)
    else:
        mask = mu.read_map(meta.masks["analysis_mask"],
                           pix_type=meta.pix_type,
                           car_template=meta.car_template)
        lmax = mu.lmax_from_map(mask, pix_type=meta.pix_type)
    print(lmax)

    bin_low, bin_high, bin_center = create_binning(lmax,
                                                   args.deltal)
    print(bin_low, bin_high, bin_center)
    file_name = f"binning_{meta.pix_type}_lmax{lmax}_deltal{args.deltal}"

    bin_low2, bin_high2, bin_center2 = create_binning(lmax,
                                                      args.deltal,
                                                      end_first_bin=30)
    print(bin_low2, bin_high2, bin_center2)

    np.savez(
        f"{binning_dir}/{file_name}.npz",
        bin_low=bin_low,
        bin_high=bin_high,
        bin_center=bin_center
    )
    np.savez(
        f"{binning_dir}/{file_name}_large_first_bin.npz",
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
