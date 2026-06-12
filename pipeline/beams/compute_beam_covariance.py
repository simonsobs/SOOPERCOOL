import argparse
from soopercool import BBmeta
from soopercool import utils
import numpy as np


def main(args):
    """
    Computes beam covariance blocks for all pairs
    of spectra (ms1, ms2) x (ms3, ms4) and
    saves them to disk. Beam covariance blocks
    can be then linearly combined
    to correct mc/analytic covariances when compiling
    data in the final SACC archive.

    Command line arguments are
    --globals:
        Path to the SOOPERCOOL config file
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory

    cells_dir = f"{out_dir}/cells"

    beam_cov_dir = f"{out_dir}/beam_covariances"
    meta.make_dir(beam_cov_dir)

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()
    n_bins = len(lb)

    field_pairs = [f"{m1}{m2}" for m1 in "TEB" for m2 in "TEB"]
    cross_ps_names = meta.get_ps_names_list(type="all", coadd=True)

    beam_covs = {}
    for ms in meta.map_sets_list:
        beam_dir = meta.beam_dir_from_map_set(ms)
        beam_file = meta.beam_file_from_map_set(ms)

        _, bl, bl_cov = utils.read_beam_from_file(
            f"{beam_dir}/{beam_file}",
            lmax=nmt_bins.lmax,
            return_cov=True
        )
        # Normalize the covariance matrix
        bl_cov = bl_cov / np.outer(bl, bl)

        # Bin the covariance matrix
        # Note that this is not formally exact as
        # we would need first to multiply with a theory spectrum
        # and then bin. I'll test with this first and will update
        # later as fiducial model might also be relevant for other steps.
        binned_cov = np.zeros((n_bins, n_bins))
        for ii in range(n_bins):
            for jj in range(n_bins):
                idx_ii = nmt_bins.get_ell_list(ii)
                idx_jj = nmt_bins.get_ell_list(jj)
                binned_cov[ii, jj] = np.mean(bl_cov[np.ix_(idx_ii, idx_jj)])
        bl_cov = binned_cov
        beam_covs[ms] = bl_cov

    for i, (ms1, ms2) in enumerate(cross_ps_names):
        cl12 = np.load(
            f"{cells_dir}/decoupled_cross_pcls_{ms1}_x_{ms2}.npz"
        )
        for j, (ms3, ms4) in enumerate(cross_ps_names):
            if i > j:
                continue

            cl34 = np.load(
                f"{cells_dir}/decoupled_cross_pcls_{ms3}_x_{ms4}.npz"
            )

            # Compute the beam covariance block for the spectra pair
            # (ms1, ms2) x (ms3, ms4)
            beam_cov = ((ms1 == ms3) + (ms1 == ms4)) * beam_covs[ms1]
            beam_cov += ((ms2 == ms3) + (ms2 == ms4)) * beam_covs[ms2]

            full_beam_cov = np.zeros((
                n_bins * len(field_pairs),
                n_bins * len(field_pairs)
            ))
            for ii, fp1 in enumerate(field_pairs):
                for jj, fp2 in enumerate(field_pairs):
                    full_beam_cov[
                        ii*n_bins:(ii+1)*n_bins,
                        jj*n_bins:(jj+1)*n_bins
                    ] = beam_cov * np.outer(cl12[fp1], cl34[fp2])

            # Save the beam covariance blocks
            np.savez(
                f"{beam_cov_dir}/beam_cov_{ms1}_x_{ms2}_{ms3}_x_{ms4}.npz",
                cov=full_beam_cov
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    args = parser.parse_args()
    main(args)
