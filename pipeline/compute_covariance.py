import argparse
from soopercool import BBmeta
import numpy as np
from soopercool import cov_utils as cov
from soopercool import utils as su
from soopercool import mpi_utils as mpi


def main(args):
    """
    This script computes an analytical estimate of the covariance matrices
    given a parameter file.
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory

    # Define output directory
    cov_dir = f"{out_dir}/covariances"
    meta.make_dir(cov_dir)
    ps_dir = f"{out_dir}/cells"

    # Load binning
    nmt_bins = meta.read_nmt_binning()
    n_bins = nmt_bins.get_n_bands()

    # Define useful variables
    field_pairs = [f"{m1}{m2}" for m1 in "TEB" for m2 in "TEB"]
    cross_ps_names = meta.get_ps_names_list(type="all", coadd=True)

    # Load beams
    beams = {}
    for map_set in meta.map_sets_list:
        beam_dir = meta.beam_dir_from_map_set(map_set)
        beam_file = meta.beam_file_from_map_set(map_set)

        _, bl = su.read_beam_from_file(
            f"{beam_dir}/{beam_file}",
            lmax=nmt_bins.lmax
        )
        bb = nmt_bins.bin_cell(bl)
        beams[map_set] = bb

    # Load some signal theory power spectra
    signal = {}
    for ms1, ms2 in cross_ps_names:
        cl = np.load(
            f"{ps_dir}/weighted_cross_pcls_{ms1}_x_{ms2}.npz"
        )
        for fp in field_pairs:
            signal[ms1, ms2, fp] = cl[fp]
        if ms1 != ms2:
            for fp in field_pairs:
                if fp == fp[::-1]:
                    signal[ms2, ms1, fp] = signal[ms1, ms2, fp].copy()
                else:
                    signal[ms2, ms1, fp] = signal[ms1, ms2, fp[::-1]].copy()

    noise_interp = {}
    for ms1, ms2 in cross_ps_names:
        cl = np.load(
            f"{ps_dir}/weighted_noise_pcls_{ms1}_x_{ms2}.npz"
        )
        for fp in field_pairs:
            noise_interp[ms1, ms2, fp] = cl[fp]
        if ms1 != ms2:
            for fp in field_pairs:
                if fp == fp[::-1]:
                    noise_interp[ms2, ms1, fp] = (
                        noise_interp[ms1, ms2, fp].copy()
                    )
                else:
                    noise_interp[ms2, ms1, fp] = (
                        noise_interp[ms1, ms2, fp[::-1]].copy()
                    )

    # Lists all covmat elements to compute
    cov_names = []
    for i, (ms1, ms2) in enumerate(cross_ps_names):
        for j, (ms3, ms4) in enumerate(cross_ps_names):
            if i > j:
                continue
            cov_names.append((ms1, ms2, ms3, ms4))

    wsp = cov.load_workspace(cov_dir)
    cwsp = cov.load_covariance_workspace(cov_dir)

    # Load transfer functions
    tf_settings = meta.transfer_settings
    tf_dir = tf_settings["transfer_directory"]
    tf_dict = {}
    for ms1, ms2 in cross_ps_names:
        ftag1 = meta.filtering_tag_from_map_set(ms1)
        ftag2 = meta.filtering_tag_from_map_set(ms2)
        tf = np.load(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz")
        tf_dict[ms1, ms2] = tf

    # Load number of bundles
    n_bundles = {}
    for map_set in meta.map_sets:
        n_bundles[map_set] = meta.n_bundles_from_map_set(map_set)

    rank, size, comm = mpi.init(True)
    cov_names = comm.bcast(cov_names, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(cov_names))
    local_cov_names = [cov_names[i] for i in task_ids]

    for ms1, ms2, ms3, ms4 in local_cov_names:

        print(f"Computing covariance for {ms1} x {ms2}, {ms3} x {ms4}")
        # Compute each covariance blocks
        cov_dict = cov.compute_covariance_block(
            ms1, ms2, ms3, ms4,
            meta.exp_tag_from_map_set(ms1),
            meta.exp_tag_from_map_set(ms2),
            meta.exp_tag_from_map_set(ms3),
            meta.exp_tag_from_map_set(ms4),
            wsp, cwsp,
            signal, noise_interp,
            nmt_bins,
            n_bundles=n_bundles
        )
        beam12 = beams[ms1] * beams[ms2]
        beam34 = beams[ms3] * beams[ms4]

        # Normalize with transfer function
        # and beams. Should be done ell per ell
        # but leads to huge instabilities
        for fp1 in field_pairs:
            for fp2 in field_pairs:
                tf1 = tf_dict[ms1, ms2][f"{fp1}_to_{fp1}"]
                tf2 = tf_dict[ms3, ms4][f"{fp2}_to_{fp2}"]

                cov_dict[fp1, fp2] /= tf1[:, None] * tf2[None, :]
                cov_dict[fp1, fp2] /= beam12[:, None] * beam34[None, :]

        full_cov = np.zeros((n_bins*len(field_pairs), n_bins*len(field_pairs)))
        for i, fp1 in enumerate(field_pairs):
            for j, fp2 in enumerate(field_pairs):
                full_cov[
                    i*n_bins:(i+1)*n_bins,
                    j*n_bins:(j+1)*n_bins
                ] = cov_dict[fp1, fp2]

        # Save block covariance matrix
        np.savez(
            f"{cov_dir}/analytic_cov_{ms1}_x_{ms2}_{ms3}_x_{ms4}.npz",
            cov=full_cov,
            lb=nmt_bins.get_effective_ells()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analytic covariance matrix computation."
    )
    parser.add_argument("--globals", type=str, help="Path to the yaml file")

    args = parser.parse_args()

    main(args)
