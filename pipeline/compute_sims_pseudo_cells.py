from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from soopercool import mpi_utils as mpi
from pathlib import Path
import os
import argparse
import numpy as np
import pymaster as nmt
from pixell import enmap
import healpy as hp
from itertools import product


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_sims"
    couplings_dir = f"{out_dir}/couplings"
    if Path(couplings_dir).is_symlink():
        couplings_dir = Path(couplings_dir).resolve(strict=False)
    nsims = meta.covariance["cov_num_sims"]

    BBmeta.make_dir(cells_dir)

    print("WARNING: Assuming signal sims are beam convolved.")
    use_alms, use_maps = (False, False)
    if "signal_alm_sims_dir" in meta.covariance:
        use_alms = meta.covariance["signal_alm_sims_dir"] is not None
    if not use_alms and "signal_map_sims_dir" in meta.covariance:
        use_maps = meta.covariance["signal_map_sims_dir"] is not None
    if use_alms:
        print("Using alms for signal covariance")
        signal_alm_dirs = meta.covariance["signal_alm_sims_dir"]
        signal_alm_templates = meta.covariance["signal_alm_sims_template"]
    if use_maps:
        print("Using maps for signal covariance")
        signal_map_dirs = meta.covariance["signal_map_sims_dir"]
        signal_map_templates = meta.covariance["signal_map_sims_template"]
    if not use_alms and not use_maps:
        print("Using noise sims only for covariance")

    binary_dir = meta.masks["analysis_mask"].replace("analysis", "binary")
    binary = mu.read_map(binary_dir,
                         pix_type=meta.pix_type,
                         car_template=meta.car_template)
    map_types = ["signal", "noise", "coadd"] 
    for typ in map_types:
        BBmeta.make_dir(f"{cells_dir}/{typ}")

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)

    lmax = mu.lmax_from_map(
        meta.masks["analysis_mask"],
        pix_type=meta.pix_type
    )
    if meta.lmax > lmax:
        raise ValueError(
            f"Specified lmax {meta.lmax} is larger than "
            f"the maximum lmax from map resolution {lmax}"
        )

    nmt_bins = meta.read_nmt_binning()
    n_bins = nmt_bins.get_n_bands()
    ps_pairs = meta.get_ps_names_list(type="all", coadd=False)

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        cname = f"{couplings_dir}/couplings_{ms1}_{ms2}.npz"
        if not os.path.isfile(cname):
            cname = f"{couplings_dir}/couplings_{ms2}_{ms1}.npz"

        inv_couplings_beamed[ms1, ms2] = np.load(cname)["inv_coupling"][:, :n_bins, :, :n_bins].reshape([n_bins*9, n_bins*9])  # noqa

    decoupled_pcls = {
        ps: []
        for ps in meta.get_ps_names_list(type="all", coadd=False)
    }
    cls_dict_sims = {pn: {} for pn in ps_pairs}

    rank, size, comm = mpi.init(True, logger=None)
    id_start = meta.covariance["cov_id_start"]
    nsims = meta.covariance["cov_num_sims"]

    # Initialize tasks for MPI sharing
    mpi_shared_list = [id_sim for id_sim in range(id_start, id_start+nsims)]

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list),
                                    logger=None)
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim in local_mpi_list:

        # Create namaster fields
        fields = {}
        for ms in meta.map_sets_list:
            ft = meta.freq_tag_from_map_set(ms)
            if use_alms:
                fname = signal_alm_templates[ms].format(id_sim=id_sim,
                                                        freq_tag=ft)
                alms = hp.read_alm(f"{signal_alm_dirs[ms]}/{fname}",
                                   hdu=(1, 2, 3))  # alms in muK_CMB
                signal = mu.alm2map(alms,
                                    pix_type=meta.pix_type,
                                    nside=meta.nside,
                                    car_map_template=meta.car_template)
            if use_maps:
                fname = signal_map_templates[ms].format(id_sim=id_sim,
                                                        freq_tag=ft)
                signal = mu.read_map(
                    f"{signal_map_dirs[ms]}/{fname}",
                    pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2],
                    car_template=meta.car_template,
                    convert_K_to_muK=False,  # signal maps are in muK_CMB
                )
            for id_bundle in range(meta.n_bundles_from_map_set(ms)):
                noise_map_dir = meta.covariance["noise_map_sims_dir"][ms]
                noise_map_template = meta.covariance["noise_map_sims_template"][ms]  # noqa
                fname = noise_map_template.format(
                    id_sim=id_sim, map_set=ms, id_bundle=id_bundle
                )
                noise_map = mu.read_map(
                    f"{noise_map_dir}/{fname}",
                    pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2],
                    convert_K_to_muK=True,  # noise maps are in K_CMB
                    car_template=meta.car_template
                )
                for typ in map_types:
                    if verbose:
                        print(f"Namaster field {typ} for sim {id_sim+1} | "
                              f"({ms}, bundle {id_bundle})")
                    if typ == "signal":
                        m = signal
                    elif typ == "noise":
                        m = noise_map
                    elif typ == "coadd":
                        m = noise_map.copy()
                        mu.add_map(signal, m, meta.pix_type)
                    mu.multiply_map(binary, m, meta.pix_type)

            wcs = None
            if meta.pix_type == "car":
                # This is a patch. Reproject mask and map onto template
                # geometry.
                tshape, twcs = enmap.read_map_geometry(meta.car_template)
                if twcs != m.wcs:
                    shape, wcs = enmap.overlap(m.shape, m.wcs, tshape, twcs)
                else:
                    shape, wcs = tshape, twcs
                if mask.wcs != wcs:
                    shape, wcs = enmap.overlap(mask.shape, mask.wcs,
                                               shape, wcs)
                if not (m.wcs == mask.wcs == twcs):
                    flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)
                    mask = enmap.insert(flat_template.copy()[0], mask)
                    m = enmap.insert(flat_template.copy(), m)
            field_spin0 = nmt.NmtField(mask, m[:1], wcs=wcs, lmax=meta.lmax)
            field_spin2 = nmt.NmtField(mask, m[1:], wcs=wcs, lmax=meta.lmax,
                                       purify_b=meta.pure_B)

        fields[typ, ms, id_bundle] = {
            "spin0": field_spin0,
            "spin2": field_spin2
        }

        for typ, (map_name1, map_name2) in product(map_types, ps_pairs):
            fname = f"{cells_dir}/{typ}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz"  # noqa
            map_set1, id_bundle1 = map_name1.split("__")
            map_set2, id_bundle2 = map_name2.split("__")

            if verbose:
                print(f"Coupled {typ} Cl for sim {id_sim+1} | "
                      f"({map_set1}, bundle {id_bundle1}) x "
                      f"({map_set2}, bundle {id_bundle2})")
            if os.path.isfile(fname) and args.no_overwrite:
                cls_dict = np.load(fname)
                for fp, cls in cls_dict.items():
                    if fp not in cls_dict_sims[(map_name1, map_name2)]:
                        cls_dict_sims[(map_name1, map_name2)][fp] = []
                    cls_dict_sims[(map_name1, map_name2)][fp] += [cls]
                continue

            pcls = pu.get_coupled_pseudo_cls(
                fields[typ, map_set1, int(id_bundle1)],
                fields[typ, map_set2, int(id_bundle2)],
                nmt_bins
            )

            decoupled_pcls = pu.decouple_pseudo_cls(
                    pcls, inv_couplings_beamed[map_set1, map_set2]
                    )
            for fp, cls in decoupled_pcls.items():
                if fp not in cls_dict_sims[(map_name1, map_name2)]:
                    cls_dict_sims[(map_name1, map_name2)][fp] = []
                cls_dict_sims[(map_name1, map_name2)][fp] += [np.array(cls).squeeze()]  # noqa

            np.savez(fname, **decoupled_pcls, lb=nmt_bins.get_effective_ells())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--globals",
        help="Path to the global parameter file."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode."
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite spectra if existing."
    )
    args = parser.parse_args()
    main(args)
