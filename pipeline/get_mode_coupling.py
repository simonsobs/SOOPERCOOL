import argparse
from soopercool import BBmeta
from soopercool import map_utils as mu
import pymaster as nmt
import numpy as np
import soopercool.utils as su
import os

from pixell import enmap


def main(args):
    """
    ...
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    # verbose = args.verbose

    out_dir = meta.output_directory

    mcm_dir = f"{out_dir}/couplings"
    BBmeta.make_dir(mcm_dir)
    plot_dir = f"{out_dir}/plots/couplings"
    if do_plots:
        BBmeta.make_dir(plot_dir)

    nspec = 7

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    mask_file = meta.masks["analysis_mask"]
    if mask_file is not None:
        mask = mu.read_map(mask_file,
                           pix_type=meta.pix_type,
                           car_template=meta.car_template)
        lmax = mu.lmax_from_map(mask_file, pix_type=meta.pix_type)
    else:
        raise FileNotFoundError("The analysis mask must be specified.")

    if nmt_bins.lmax < lmax:  # In case bins are lower than mask's lmax
        lmax = nmt_bins.lmax
    nl = lmax + 1

    binner = np.array([nmt_bins.bin_cell(np.array([cl]))[0]
                       for cl in np.eye(nl)]).T

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
        lmax=lmax
    )
    field_spin2 = nmt.NmtField(
        mask,
        None,
        wcs=wcs,
        spin=2,
        lmax=lmax,
        purify_b=meta.pure_B
    )

    # Alright, compute and reshape coupling matrix.
    print("Computing MCM")
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(field_spin0, field_spin2, nmt_bins, is_teb=True)

    mcm = np.transpose(w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
                       axes=[1, 0, 3, 2])
    mcm_binned = np.einsum('ij,kjlm->kilm', binner, mcm)

    # Load beams to correct the mode coupling matrix
    beams = {}
    for map_set in meta.map_sets_list:
        beam_dir = meta.beam_dir_from_map_set(map_set)
        beam_file = meta.beam_file_from_map_set(map_set)

        _, bl = su.read_beam_from_file(
            f"{beam_dir}/{beam_file}",
            lmax=lmax
        )
        beams[map_set] = bl

    # Beam-correct the mode coupling matrix
    beamed_mcm = {}
    for map_set1, map_set2 in meta.get_ps_names_list("all", coadd=True):
        beamed_mcm[map_set1, map_set2] = mcm * \
            np.outer(beams[map_set1],
                     beams[map_set2])[np.newaxis, :, np.newaxis, :]

    # Save files
    # Starting with the un-beamed non-purified MCM
    np.savez(
        f"{mcm_dir}/mcm.npz",
        binner=binner,
        spin0xspin0=mcm[0, :, 0, :].reshape([1, nl, 1, nl]),
        spin0xspin2=mcm[1:3, :, 1:3, :],
        spin2xspin2=mcm[3:, :, 3:, :],
        spin0xspin0_binned=mcm_binned[0, :, 0, :].reshape([1, n_bins, 1, nl]),
        spin0xspin2_binned=mcm_binned[1:3, :, 1:3, :],
        spin2xspin2_binned=mcm_binned[3:, :, 3:, :]
    )

    # Then the beamed MCM
    for map_set1, map_set2 in meta.get_ps_names_list("all", coadd=True):
        output_file = f"{mcm_dir}/mcm_{map_set1}_{map_set2}.npz"
        if not os.path.exists(output_file):
            m = beamed_mcm[map_set1, map_set2]
            mcm_binned = np.einsum('ij,kjlm->kilm', binner, m)
            np.savez(
                output_file,
                binner=binner,
                spin0xspin0=m[0, :, 0, :].reshape([1, nl, 1, nl]),
                spin0xspin2=m[1:3, :, 1:3, :],
                spin2xspin2=m[3:, :, 3:, :],
                spin0xspin0_binned=mcm_binned[0, :, 0, :].reshape([1, n_bins,
                                                                   1, nl]),
                spin0xspin2_binned=mcm_binned[1:3, :, 1:3, :],
                spin2xspin2_binned=mcm_binned[3:, :, 3:, :]
            )
        else:
            print(f"File {output_file} already exists. Skipping computation.")


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
