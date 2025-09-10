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
    if args.binning_dir is not None:
        binning_dir = args.binning_dir
    BBmeta.make_dir(binning_dir)

    from pixell import enmap
    geometry = enmap.read_map_geometry(meta.car_template)
    lmax = mu._lmax_from_car_geometry(geometry)

    # nmt_bins.lmax+1 cannot extend beyond lmax-1 because otherwise
    # coupled_cells would have len == lmax < nmt_bins.lmax+1
    bin_low, bin_high, bin_center = create_binning(lmax-1,
                                                   args.deltal)
<<<<<<< HEAD
=======
    #if bin_low[0] < 2:
    #    bin_low[0] = 2
        
    print(bin_low, bin_high, bin_center)
>>>>>>> origin/sa_dev
    file_name = f"binning_{meta.pix_type}_lmax{lmax}_deltal{args.deltal}"

    bin_low2, bin_high2, bin_center2 = create_binning(lmax-1,
                                                      args.deltal,
<<<<<<< HEAD
                                                      end_first_bin=30)
=======
                                                      end_first_bin=10)
    
    print()
    print(bin_low2, bin_high2, bin_center2)
>>>>>>> origin/sa_dev

    np.savez(
        f"{binning_dir}/{file_name}.npz",
        bin_low=bin_low,
        bin_high=bin_high,
        bin_center=bin_center
    )
    print(f"  SAVED: {binning_dir}/{file_name}.npz")
    np.savez(
        f"{binning_dir}/{file_name}_large_first_bin.npz",
        bin_low=bin_low2,
        bin_high=bin_high2,
        bin_center=bin_center2
    )
    print(f"  SAVED: {binning_dir}/{file_name}_large_first_bin.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--deltal", type=int,
                        help="Delta ell for the binning.")
    parser.add_argument("--binning_dir", default=None,
                        help="Directory to save binning at.")
    args = parser.parse_args()
    main(args)
