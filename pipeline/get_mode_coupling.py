from soopercool import map_utils as mu
from soopercool import BBmeta
from pixell import enmap
import pymaster as nmt
import numpy as np
import argparse


def main(args):
    """
    Compute and save the mode coupling matrix
    from a given analysis_mask specified
    in the paramfile.
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots

    out_dir = meta.output_directory

    mcm_dir = f"{out_dir}/couplings"
    BBmeta.make_dir(mcm_dir)
    plot_dir = f"{out_dir}/plots/couplings"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    nspec = 7

    nmt_bins = meta.read_nmt_binning()

    mask_file = meta.masks["analysis_mask"]
    if mask_file is not None:
        mask = mu.read_map(mask_file,
                           pix_type=meta.pix_type,
                           car_template=meta.car_template)
        lmax = mu.lmax_from_map(mask_file, pix_type=meta.pix_type)
    else:
        raise FileNotFoundError("The analysis mask must be specified.")

    if meta.lmax > lmax:
        raise ValueError(
            f"Specified lmax {meta.lmax} is larger than "
            f"the maximum lmax from map resolution {lmax}"
        )
    nl = meta.lmax + 1

    binner = np.array([
        nmt_bins.bin_cell(np.array([cl]))[0]
        for cl in np.eye(nl)
    ]).T

    wcs = None
    if hasattr(mask, 'wcs'):
        # This is a patch. Reproject mask onto template geometry.
        tshape, twcs = enmap.read_map_geometry(meta.car_template)
        shape, wcs = enmap.overlap(mask.shape, mask.wcs, tshape, twcs)
        flat_template = enmap.zeros(shape, wcs)
        mask = enmap.insert(flat_template.copy(), mask)

    field_spin0 = nmt.NmtField(
        mask,
        None,
        wcs=wcs,
        spin=0,
        lmax=meta.lmax
    )
    field_spin2 = nmt.NmtField(
        mask,
        None,
        wcs=wcs,
        spin=2,
        lmax=meta.lmax,
        purify_b=meta.pure_B
    )

    # Alright, compute and reshape coupling matrix.
    print("Computing MCM...")
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(field_spin0, field_spin2, nmt_bins, is_teb=True)

    mcm = np.transpose(
        w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
        axes=[1, 0, 3, 2]
    )

    # Save files
    print(f"Saving MCM to {mcm_dir}/mcm.npz...")
    np.savez(
        f"{mcm_dir}/mcm.npz",
        binner=binner,
        spin0xspin0=mcm[0, :, 0, :].reshape([1, nl, 1, nl]),
        spin0xspin2=mcm[1:3, :, 1:3, :],
        spin2xspin2=mcm[3:, :, 3:, :]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute mask mode coupling matrices"
    )
    parser.add_argument("--globals", help="Path to the paramfile")
    parser.add_argument("--verbose", help="Verbose mode",
                        action="store_true")
    parser.add_argument("--no-plots", help="Plot the results",
                        action="store_true")

    args = parser.parse_args()

    main(args)
