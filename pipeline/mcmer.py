import argparse
import numpy as np
import pymaster as nmt
from soopercool import BBmeta


def mcmer(args):
    """
    Compute the mode coupling matrix
    from the masks defined in the global
    parameter file.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)

    coupling_dir = meta.coupling_directory

    nl = 3*meta.nside
    nspec = 7

    # Read and degrade mask
    print("Reading mask")
    mask = meta.read_mask("analysis")

    # Create dummy NaMaster fields, spin-2 is optionally purified.
    field_spin0 = nmt.NmtField(mask, None, spin=0)
    field_spin2 = nmt.NmtField(mask, None, spin=2, purify_b=meta.pure_B)

    # Binning scheme is irrelevant for us, but NaMaster needs one.
    nmt_bins = meta.read_nmt_binning()

    # Create binner
    n_bins = nmt_bins.get_n_bands()
    binner = np.array([nmt_bins.bin_cell(np.array([cl]))[0]
                       for cl in np.eye(nl)]).T

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
        l, bl = meta.read_beam(map_set)
        beams[map_set] = bl[:nl]

    # Beam-correct the mode coupling matrix
    beamed_mcm = {}
    for map_set1, map_set2 in meta.get_ps_names_list("all", coadd=True):
        beamed_mcm[map_set1, map_set2] = mcm * \
            np.outer(beams[map_set1],
                     beams[map_set2])[np.newaxis, :, np.newaxis, :]
    # Save files
    # Starting with the un-beamed non-purified MCM
    fname = f"{coupling_dir}/mcm.npz"
    np.savez(
        fname,
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
        m = beamed_mcm[map_set1, map_set2]
        mcm_binned = np.einsum('ij,kjlm->kilm', binner, m)
        fname = f"{coupling_dir}/mcm_{map_set1}_{map_set2}.npz"
        np.savez(
            fname,
            binner=binner,
            spin0xspin0=m[0, :, 0, :].reshape([1, nl, 1, nl]),
            spin0xspin2=m[1:3, :, 1:3, :],
            spin2xspin2=m[3:, :, 3:, :],
            spin0xspin0_binned=mcm_binned[0, :, 0, :].reshape([1, n_bins,
                                                               1, nl]),
            spin0xspin2_binned=mcm_binned[1:3, :, 1:3, :],
            spin2xspin2_binned=mcm_binned[3:, :, 3:, :]
        )

    # Finally, for the different filtering tag pairs.
    # used for the transfer validation stage, in
    # case we beamed the estimation and validation simulations
    for ftag1, ftag2 in meta.get_independent_filtering_pairs():

        beam_label1 = meta.tags_settings[ftag1]["beam"]
        beam_label2 = meta.tags_settings[ftag2]["beam"]

        bl1 = beams[beam_label1] if beam_label1 is not None else np.ones(nl)
        bl2 = beams[beam_label2] if beam_label2 is not None else np.ones(nl)

        beamed_mcm = mcm * \
            np.outer(bl1, bl2)[np.newaxis, :, np.newaxis, :]

        mcm_binned = np.einsum('ij,kjlm->kilm', binner, beamed_mcm)

        fname = f"{coupling_dir}/mcm_{ftag1}_{ftag2}.npz"
        np.savez(
            fname,
            binner=binner,
            spin0xspin0=beamed_mcm[0, :, 0, :].reshape([1, nl, 1, nl]),
            spin0xspin2=beamed_mcm[1:3, :, 1:3, :],
            spin2xspin2=beamed_mcm[3:, :, 3:, :],
            spin0xspin0_binned=mcm_binned[0, :, 0, :].reshape([1, n_bins,
                                                               1, nl]),
            spin0xspin2_binned=mcm_binned[1:3, :, 1:3, :],
            spin2xspin2_binned=mcm_binned[3:, :, 3:, :]
        )

    if args.plots:
        import matplotlib.pyplot as plt

        print("Plotting")
        plt.figure()
        plt.imshow(mcm.reshape([nspec*nl, nspec*nl]))
        plt.colorbar()
        fname = f"{coupling_dir}/mcm.png"
        plt.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCM calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--plots", action='store_true',
                        help='Pass to generate a plot of the MCM.')
    args = parser.parse_args()

    mcmer(args)
