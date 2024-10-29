import os
import healpy as hp
import numpy as np
import argparse

# Yuji's code
import coordinator
import coadder


def main(args):
    """
    Read list of inverse noise-weighted atomic maps and list of per-pixel
    inverse noise covariance matrices, separate them into bundles, coadd
    them, adn write them to disk. Optionally apply a random sign flip to the
    atomics within each bundle before coadding, to obtain per-bundle noise
    realizations.

    Inputs:

        atomics_list: list of strings, containing the file paths of atomic
                      maps.

        invcov_list: list of strings, containing the file paths of per-pixel
                     inverse noise covariance matrices corresponding to each
                     atomic map.

        nbundles: integer, number of bundles to be formed from atomics.

        outdir: string, output directory.

        prefix_outmap: string, prefix to files written to outdir.
                       Default: None.

        do_signflip: boolean, whether to apply the sign flip procedure.
                     Default: False.

        maps_are_weighted: boolean, whether atomics are inverse noise-variance
                           weighted or not. Default: False.
    """
    os.makedirs(args.outdir, exist_ok=True)
    atomic_maps_list = np.load(args.atomics_list_fpath)
    natomics = len(atomic_maps_list["wmap"])

    # Load ivar weights and weighted maps
    # *****
    # FIXME: This whole part needs to be adapted and generalized to CAR or
    #        HEALPix.
    npix = len(atomic_maps_list[0][0, :])

    # invcov-weighted input maps
    weighted_maps = np.zeros(shape=(natomics, 3, npix), dtype=np.float32)
    # inverse noise covariance maps
    weights = np.zeros(shape=(natomics, 3, npix), dtype=np.float32)
    # hits maps
    hits_maps = np.zeros(shape=(natomics, npix), dtype=np.float32)

    # Random division into bundles (using Yuji's code)
    bundle_mask = coordinator.gen_masks_of_given_atomic_map_list_for_bundles(
        natomics, args.nbundles
    )

    # Loop over bundle-wise atomics
    for id_bundle in range(args.nbundles):
        print("Bundle #", id_bundle + 1)
        for i in range(natomics)[bundle_mask]:
            print(i, end=',')
            fname_wmap = atomic_maps_list["wmap"][i]

            # FIXME: Load correctly. At the moment this is a placeholder.
            weighted_maps[i] = coordinator.read_hdf5_map(fname_wmap)
            fname_weights = atomic_maps_list["weights"][i]
            weights[i] = coordinator.read_hdf5_map(fname_weights)[(0, 3, 5), :]

        # Coadd tthe ivar weights and hits maps
        coadd_weight_map = np.sum(weights, axis=0)
        mask = np.mean(coadd_weight_map[1:], axis=0) > 0.
        nside = hp.npix2nside(npix)
        coadd_hits_map = np.sum(hits_maps, axis=0)[mask]

        # Optionally apply signflip using Yuji's code
        if args.do_signflip:
            fname = 'sf_map.hdf5'

            sf = coadder.SignFlip()
            obs_weights = np.sum(weights[:, 1:, :]*0.5, axis=(1, 2))

            sf.gen_seq(obs_weights)
            signs = sf.seq * 2 - 1
        else:
            fname = 'coadd_map.hdf5'
            signs = np.ones(natomics)

        # Divide by weights to get back the unweighted map solution
        coadd_solved_map = np.divide(
            np.sum(signs[:, np.newaxis, np.newaxis]*weighted_maps, axis=0),
            coadd_weight_map, out=np.zeros_like(coadd_weight_map),
            where=mask
        )
    # *****

        # Save maps to disk
        fname = os.path.join(args.outdir, fname)
        dict_maps = {"weight_map": coadd_weight_map,
                     "solved_map": coadd_solved_map,
                     "hits_map": coadd_hits_map}

        # This is just a proxy to the obs ID, e.g.
        # "atomic_1709852088_ws2_f090_full"
        list_of_obsid = ["_".join(atm.split('/')[-1].split("_")[:-1])
                         for atm in atomic_maps_list["wmap"][bundle_mask]]

        # TODO: Adapt SOOPERCOOL to handle hdf5 maps as inputs.
        coordinator.write_hdf5_map(fname, nside, dict_maps, list_of_obsid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atomic_list_fpath",
                        help="Path to npz file with path to atomic maps.")
    parser.add_argument("--outdir",
                        help="Output directory for bundle maps list.")
    parser.add_argument("--nbundles",
                        help="Number of bundles to make from atomic maps.")
    parser.add_argument("--do_signflip", action="store_true",
                        help="Whether to make sign-flip noise realizations"
                        "from the atomic maps in each bundle.")

    args = parser.parse_args()
    main(args)
