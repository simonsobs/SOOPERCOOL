import argparse
from soopercool import BBmeta
import numpy as np
from soopercool import map_utils as mu
import pymaster as nmt
import os
from soopercool import mpi_utils as mpi
import sotodlib.preprocess.preprocess_util as pp_utils
from itertools import product
import time


def main(args):
    """
    This script computes an analytical estimate of the covariance matrices
    given a parameter file.
    """
    rank, size, comm = mpi.init(True)

    meta = BBmeta(args.globals)
    logger = pp_utils.init_logger("benchmark", verbosity=1)

    out_dir = meta.output_directory
    # Define output directory
    cov_dir = f"{out_dir}/covariances"
    meta.make_dir(cov_dir)

    # Load binning
    nmt_bins = meta.read_nmt_binning()

    # Load analysis mask
    mask = mu.read_map(
        meta.masks["analysis_mask"],
        pix_type=meta.pix_type,
        car_template=meta.car_template
    )
    wcs = mask.wcs if meta.pix_type == "car" else None

    # TODO: improve hits input
    # maybe with the path specified under
    # map_set in paramfile
    hits = mu.read_map(
        meta.masks["analysis_mask"].replace(
            "analysis_mask.fits",
            "normalized_hits.fits"
        ),
        pix_type=meta.pix_type,
        car_template=meta.car_template
    )

    t0 = time.time()
    fields = {}
    fields["spin0"] = nmt.NmtField(
        mask,
        maps=None,
        wcs=wcs,
        spin=0,
        lmax=meta.lmax
    )
    fields["spin2"] = nmt.NmtField(
        mask,
        maps=None,
        purify_b=meta.pure_B,
        wcs=wcs,
        spin=2,
        lmax=meta.lmax
    )

    sigma = hits.copy()
    sigma[hits > 0] = 1 / np.sqrt(hits[hits > 0])
    mask_noise = mask * sigma
    pix_type = "car" if hasattr(mask, "geometry") else "hp"
    rescale_noise = mu.sky_average(
        (sigma * mask) ** 2,
        pix_type=pix_type,
    ) / mu.sky_average(
        mask ** 2,
        pix_type=pix_type,
    )

    fields["spin0", "noise"] = nmt.NmtField(
        mask_noise / np.sqrt(rescale_noise),
        maps=None,
        wcs=wcs,
        spin=0,
        lmax=meta.lmax
    )
    fields["spin2", "noise"] = nmt.NmtField(
        mask_noise / np.sqrt(rescale_noise),
        maps=None,
        purify_b=meta.pure_B,
        wcs=wcs,
        spin=2,
        lmax=meta.lmax
    )
    logger.info(f"[{rank}] Time to set up nmt fields: {time.time() - t0:.2f}s")

    spin_pairs = list(product(["spin0", "spin2"], repeat=2))

    # Initialize MPI
    spin_combos = [
        (s0, s1, s2, s3)
        for (s0, s1) in spin_pairs
        for (s2, s3) in spin_pairs
    ]
    spin_combos = comm.bcast(spin_combos, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(spin_combos))
    local_spin_combos = [spin_combos[i] for i in task_ids]

    t0 = time.time()
    wsp = {}
    for s0, s1 in spin_pairs:
        fname_wsp = f"{cov_dir}/wsp_{s0}_{s1}"
        if not os.path.exists(fname_wsp):
            wsp[s0, s1] = nmt.NmtWorkspace(
                fields[s0],
                fields[s1],
                nmt_bins
            )
            if rank == 0:
                wsp[s0, s1].write_to(fname_wsp)
        else:
            wsp[s0, s1] = nmt.NmtWorkspace.from_file(fname_wsp)
    logger.info(
        f"[{rank}] Time to compute workspaces: {time.time() - t0:.2f}s"
    )

    for s0, s1, s2, s3 in local_spin_combos:

        # signal-signal
        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0],
                fields[s1],
                fields[s2],
                fields[s3]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(
            f"[{rank}] signal-signal cwsp computed in {time.time() - t0:.2f}s"
        )

        # signal-noise
        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_snsn_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0],
                fields[s1, "noise"],
                fields[s2],
                fields[s3, "noise"]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(
            f"[{rank}] snsn cwsp computed in {time.time() - t0:.2f}s"
        )

        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_snns_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0],
                fields[s1, "noise"],
                fields[s2, "noise"],
                fields[s3]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(f"[{rank}] snns cwsp computed in {time.time() - t0:.2f} s")

        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_nssn_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0, "noise"],
                fields[s1],
                fields[s2],
                fields[s3, "noise"]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(f"[{rank}] nssn cwsp computed in {time.time() - t0:.2f} s")

        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_nsns_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0, "noise"],
                fields[s1],
                fields[s2, "noise"],
                fields[s3]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(f"[{rank}] nsns cwsp computed in {time.time() - t0:.2f} s")

        # noise-noise
        t0 = time.time()
        fname_cwsp = f"{cov_dir}/cwsp_nn_{s0}_{s1}_{s2}_{s3}"
        if not os.path.exists(fname_cwsp):
            cwsp = nmt.NmtCovarianceWorkspace(
                fields[s0, "noise"],
                fields[s1, "noise"],
                fields[s2, "noise"],
                fields[s3, "noise"]
            )
            cwsp.write_to(fname_cwsp)
        logger.info(f"[{rank}] nn cwsp computed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and saves couplings for covariance matrices"
    )
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    args = parser.parse_args()

    main(args)
