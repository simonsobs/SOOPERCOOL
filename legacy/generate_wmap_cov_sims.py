import os
from itertools import combinations_with_replacement as cwr

import healpy as hp
import numpy as np
import pymaster as nmt

import soopercool.mpi_utils as mpi_utils

""" This script loads WMAP noise bundele sims, coadd them, and store them to
compute a noise-only covariance for consistency checks. """


def load_wmap_noise(nside, freq, id_sim, id_bundle):
    """
    Load WMAP noise maps from NERSC at nside 256.
    """
    assert nside == 256
    to_muk = 1.0e3
    str_bundle = str(id_bundle + 1)
    str_freq = str(int(freq)).zfill(3)
    bands_dict = {"023": "K1", "033": "Ka1"}
    fname_in = f"/global/cfs/cdirs/sobs/users/cranucci/wmap/nside256_coords_eq/noise/{id_sim:04d}"
    fname_in += f"/noise_maps_mK_band{bands_dict[str_freq]}_yr{str_bundle}.fits"
    return to_muk * hp.read_map(fname_in, field=range(3))


def coadd_bundles(bundles_list):
    """
    Coadd a list of map bundles into a single map.
    """
    coadd = np.zeros_like(bundles_list[0])
    for bundle in bundles_list:
        coadd += bundle

    return coadd


def compute_workspace(nmt_bin, nside, mask_dir):
    """
    Compute the NaMaster workspace to compute decoupled pseudo-C_ells.
    """
    w = nmt.NmtWorkspace()
    mask = hp.ud_grade(hp.read_map(mask_dir), nside_out=nside)
    f = nmt.NmtField(mask, None, spin=2, purify_b=True)
    w.compute_coupling_matrix(f, f, nmt_bin)

    return w, mask


def nmt_bin_from_edges(bin_edges, nside):
    """
    Computes a NaMaster NmtBin object given an input array of bin edges.
    """
    bin_edges = np.array(bin_edges)
    bin_edges = bin_edges[bin_edges < 3 * nside]
    bin_edges = np.concatenate((bin_edges, [3 * nside]))
    return nmt.NmtBin.from_edges(bin_edges[:-1], bin_edges[1:])


def get_decoupled_ps(map1, map2, mask, nmt_bin, wsp):
    """
    Compute decoupled pseudo C_ells from a map pair given a NaMaster workspace.
    """
    f1 = nmt.NmtField(mask, map1[1:], purify_b=True)
    f2 = nmt.NmtField(mask, map2[1:], purify_b=True)
    pcl = nmt.compute_coupled_cell(f1, f2)
    cl_dict = {
        f: wsp.decouple_cell(pcl)[f_idx]
        for f_idx, f in enumerate(["EE", "EB", "BE", "BB"])
    }
    cl_dict["l"] = nmt_bin.get_effective_ells()

    return cl_dict


def generate_sim(nside, id_sim, Nbundles, freqs_list, sims_dir):
    """
    Load, coadd, and store WMAP noise maps to disk.
    """
    for f in freqs_list:
        f_str = str(f).zfill(3)
        bundles_list = [
            load_wmap_noise(nside, f, id_sim, id_bundle)
            for id_bundle in range(Nbundles)
        ]
        coadd = coadd_bundles(bundles_list)
        dirname = f"{sims_dir}/{str(id_sim).zfill(4)}"
        os.makedirs(dirname, exist_ok=True)
        fname = f"{dirname}/wmap_f{f_str}_noise_coadded.fits"
        print(fname)
        hp.write_map(fname, coadd, overwrite=True)


def compute_cells(mapset_list, sims_dir, id_sim, sim_range, cl_dir, mask, nmt_bin, wsp):
    """
    Load simulated maps from disk and compute cross-frequency C_ells.
    """
    os.makedirs(cl_dir, exist_ok=True)
    cross_ps_names = cwr(mapset_list, 2)
    # id_range = [s for s in sim_range if s != id_sim]

    for ms1, ms2 in cross_ps_names:
        # id_sim2 = np.random.choice(id_range)
        id_sim2 = id_sim
        map1, map2 = (
            hp.read_map(
                f"{sims_dir}/{str(id_sim).zfill(4)}/{ms}_noise_coadded.fits",
                field=(0, 1, 2),
            )  # noqa
            for id_sim, ms in zip([id_sim, id_sim2], [ms1, ms2])
        )
        cl_dict = get_decoupled_ps(map1, map2, mask, nmt_bin, wsp)
        f = f"{cl_dir}/decoupled_cross_pcls_nobeam_{ms1}_{ms2}_{id_sim:04d}.npz"
        np.savez(f, **cl_dict)


def compute_covariance(Nsims, mapset_list, cl_dir, cov_dir):
    """ """
    field_pairs = [f"{m1}{m2}" for m1 in "EB" for m2 in "EB"]
    cross_ps_names = list(cwr(mapset_list, 2))
    # print(cross_ps_names)
    cov_names = []
    for i, (ms1, ms2) in enumerate(cross_ps_names):
        for j, (ms3, ms4) in enumerate(cross_ps_names):
            if i > j:
                continue
            cov_names.append((ms1, ms2, ms3, ms4))

    cl_dict = {}
    for ms1, ms2 in cross_ps_names:
        cl_list = []
        for iii in range(Nsims):
            cells_dict = np.load(
                f"{cl_dir}/decoupled_cross_pcls_nobeam_{ms1}_{ms2}_{iii:04d}.npz",  # noqa
            )
            # print(ms1, ms2, iii, len(cells_dict["EE"]))
            cl_vec = np.concatenate(
                [cells_dict[field_pair] for field_pair in field_pairs]
            )
            cl_list.append(cl_vec)
        cl_dict[ms1, ms2] = np.array(cl_list)

    n_bins = cl_dict[list(cross_ps_names)[i]].shape[-1] // len(field_pairs)

    full_cov_dict = {}

    for id_mapset in range(len(cov_names)):
        ms1, ms2, ms3, ms4 = cov_names[id_mapset]

        cl12 = cl_dict[ms1, ms2]
        cl34 = cl_dict[ms3, ms4]

        cl12_mean = np.mean(cl12, axis=0)
        cl34_mean = np.mean(cl34, axis=0)

        cov = np.mean(
            np.einsum("ij,ik->ijk", cl12 - cl12_mean, cl34 - cl34_mean), axis=0
        )
        full_cov_dict[ms1, ms2, ms3, ms4] = cov

        cov_dict = {}
        for i, field_pair_1 in enumerate(field_pairs):
            for j, field_pair_2 in enumerate(field_pairs):

                cov_block = cov[
                    i * n_bins : (i + 1) * n_bins, j * n_bins : (j + 1) * n_bins
                ]
                cov_dict[field_pair_1 + field_pair_2] = cov_block

        os.makedirs(cov_dir, exist_ok=True)
        fname = f"{cov_dir}/mc_cov_{ms1}_{ms2}_{ms3}_{ms4}.npz"
        print(fname)
        np.savez(fname, **cov_dict)


def main():
    Nsims = 100
    Nbundles = 9
    freqs_list = [23, 33]
    mapset_list = [f"wmap_f{str(f).zfill(3)}" for f in freqs_list]
    nside = 256
    mask_dir = (
        "/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_wmap/masks/analysis_mask.fits"
    )
    cl_dir = "/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_wmap_noise/cells_sims"
    cov_dir = "/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_wmap_noise/covariances"
    pre_dir = "/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_wmap_noise/pre_processing"
    sims_dir = "/pscratch/sd/k/kwolz/bbdev/SOOPERCOOL/outputs_wmap_noise/sims"
    bin_edges = [
        2,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
    ]

    os.makedirs(pre_dir, exist_ok=True)
    nmt_bin = nmt_bin_from_edges(bin_edges, nside)
    print(nmt_bin.get_effective_ells())

    wsp, mask = compute_workspace(nmt_bin, nside, mask_dir)

    # Initialize MPI
    use_mpi4py = True
    mpi_utils.init(use_mpi4py)

    print("-------------------------------------------------------------------")
    print("           Generating sims                                         ")
    print("-------------------------------------------------------------------")
    for id_sim in mpi_utils.taskrange(2 * Nsims - 1):
        generate_sim(nside, id_sim, Nbundles, freqs_list, sims_dir)

    print("-------------------------------------------------------------------")
    print("           Computing C_ells                                        ")
    print("-------------------------------------------------------------------")
    for id_sim in mpi_utils.taskrange(2 * Nsims - 1):
        compute_cells(
            mapset_list, sims_dir, id_sim, range(Nsims), cl_dir, mask, nmt_bin, wsp
        )

    print("-------------------------------------------------------------------")
    print("           Computing covariances                                   ")
    print("-------------------------------------------------------------------")
    compute_covariance(Nsims, mapset_list, cl_dir, cov_dir)


main()
