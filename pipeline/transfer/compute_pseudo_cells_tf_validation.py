import argparse
from soopercool import BBmeta
import pymaster as nmt
import numpy as np
from soopercool import ps_utils as pu
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from soopercool import coupling_utils as cu
import soopercool.utils as su
from pixell import enmap


def main(args):
    """
    Compute (TF*MCM)-decoupled power spectra from (un)filtered TF validation
    simulations stored under the yaml section transfer_settings['validation']

    This script must be run before 'validate_transfer_function.py'. Running it
    is NOT a prerequisite for 'validate_transfer_function_kspace.py'.
    """
    rank, size, comm = mpi.init(True)

    meta = BBmeta(args.globals)
    verbose = args.verbose
    out_dir = meta.output_directory
    couplings_dir = f"{out_dir}/couplings"

    cls_tf_val_dir = f"{out_dir}/cells_tf_val"
    BBmeta.make_dir(cls_tf_val_dir)

    kspace_dir = f"{out_dir}/kspace_filtered_sims"

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()
    n_bins = nmt_bins.get_n_bands()
    ps_pairs = meta.get_ps_names_list(type="all", coadd=True)

    if "validation" not in meta.transfer_settings:
        raise KeyError("Subsection 'transfer.validation' in config missing.")

    validation_dict = meta.transfer_settings["validation"]
    simdir_unfiltered = validation_dict["unfiltered_map_dir"]
    simdir_filtered = validation_dict["filtered_map_dir"]

    mask_file = meta.masks["analysis_mask"]
    if mask_file is None:
        raise ValueError("An analysis mask must be provided.")
    mask = mu.read_map(mask_file,
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

    tf_settings = meta.transfer_settings
    sim_id_start = 0 if "sim_id_start" not in tf_settings else tf_settings["sim_id_start"]  # noqa
    sim_ids = range(sim_id_start, tf_settings["tf_val_num_sims"]+sim_id_start)

    # Load MCMs, transfer functions and compute coupling matrices
    # This avoid saving all products to disk and save disk space.
    mcm = cu.read_mcm(
        f"{couplings_dir}/mcm.npz",
        full_mcm=True
    )
    inv_couplings = {}
    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):

        _, bl1 = su.read_beam_from_file(
            "/".join([
                meta.beam_dir_from_map_set(ms1),
                meta.beam_file_from_map_set(ms1)
            ]),
            lmax=meta.lmax
        )
        _, bl2 = su.read_beam_from_file(
            "/".join([
                meta.beam_dir_from_map_set(ms2),
                meta.beam_file_from_map_set(ms2)
            ]),
            lmax=meta.lmax
        )
        print("bl", ms1, ms2, bl1[:10])
        beam = np.outer(bl1, bl2)

        transfer = cu.load_transfer_function(
            meta.transfer_settings["transfer_directory"],
            ms1, ms2,
            meta.filtering_tag_from_map_set,
            meta.kspace_tag_from_map_set,
            nmt_bins
        )
        _, inv_couplings_fil = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=transfer,
            compute_Dl=meta.compute_Dl,
            beam=beam
        )
        _, inv_couplings_unf = cu.compute_couplings(
            mcm,
            nmt_bins,
            transfer=None,
            compute_Dl=meta.compute_Dl,
            beam=beam
        )
        inv_couplings_fil = inv_couplings_fil.reshape([n_bins*9, n_bins*9])
        inv_couplings_unf = inv_couplings_unf.reshape([n_bins*9, n_bins*9])
        inv_couplings["filtered", ms1, ms2] = inv_couplings_fil
        inv_couplings["unfiltered", ms1, ms2] = inv_couplings_unf

    filtering_tags = meta.get_filtering_tags()
    filtering_tag_pairs = meta.get_independent_filtering_pairs()

    if (None, None) in filtering_tags and len(filtering_tags) < 1:
        raise ValueError("There must be at least one filter \
                         applied to the data to be able to \
                         compute a transfer function for it")

    mpi_shared_list = [(id_sim, ftag1, ftag2)
                       for ftag1, ftag2 in filtering_tag_pairs
                       for id_sim in sim_ids]

    # Every rank must have the same list order
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)

    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, ftag1, ftag2 in local_mpi_list:
        if verbose:
            print(f" Doing id_sim {id_sim} | {ftag1} x {ftag2}")

        # Create namaster fields
        ftags_unique = list(dict.fromkeys([ftag1, ftag2]))
        fields = {
            ftag: {
                "filtered": {},
                "unfiltered": {}
            } for ftag in ftags_unique
        }

        for ftag in ftags_unique:
            preproc_ftag, kspace_tag = ftag
            if verbose:
                print(f" Field for ('{preproc_ftag}', '{kspace_tag}')")

            unfiltered_map_dir = simdir_unfiltered[preproc_ftag]
            unfiltered_map_tmpl = validation_dict["unfiltered_map_template"][preproc_ftag] # noqa
            unfiltered_map_file = unfiltered_map_tmpl.format(id_sim=id_sim)
            unfiltered_map_file = f"{unfiltered_map_dir}/{unfiltered_map_file}"

            if kspace_tag is None:
                filtered_map_dir = validation_dict["filtered_map_dir"][preproc_ftag] # noqa
            else:
                filtered_map_dir = kspace_dir

            filtered_map_dir = simdir_filtered[preproc_ftag]
            filtered_map_tmpl = validation_dict["filtered_map_template"][preproc_ftag] # noqa
            filtered_map_file = filtered_map_tmpl.format(id_sim=id_sim)
            filtered_map_file = f"{filtered_map_dir}/{filtered_map_file}"

            if kspace_tag is not None:
                filtered_map_file = filtered_map_file.replace(
                    ".fits", f"_kspace_{kspace_tag}.fits"
                )

            m_unf = mu.read_map(
                unfiltered_map_file,
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
            )
            m_f = mu.read_map(
                filtered_map_file,
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
            )
            for isfil_tag, m in zip(["filtered", "unfiltered"], [m_f, m_unf]):
                wcs = None
                if hasattr(m, 'wcs'):
                    # This is a patch. Reproject mask and map onto template
                    # geometry.
                    tshape, twcs = enmap.read_map_geometry(meta.car_template)
                    shape, wcs = enmap.overlap(m.shape, m.wcs, tshape, twcs)
                    shape, wcs = enmap.overlap(mask.shape, mask.wcs, shape, wcs)  # noqa
                    flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)
                    mask = enmap.insert(flat_template.copy()[0], mask)
                    m = enmap.insert(flat_template.copy(), m)

                    # Deal with possibly missing atomics => different footprint
                    mask_restrict = m.copy()
                    ax = 1 if meta.pix_type == "hp" else (1, 2)
                    mask_restrict = mask_restrict[np.any(mask_restrict,
                                                         axis=ax)]
                    mask_restrict = np.all(mask_restrict, axis=0).astype(float)
                    mask_restrict *= np.array(mask)
                    _, wcs = enmap.read_map_geometry(meta.car_template)
                else:
                    mask_restrict = mask

                field_spin0 = nmt.NmtField(
                    mask_restrict,
                    m[:1],
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax
                )
                field_spin2 = nmt.NmtField(
                    mask_restrict,
                    m[1:],
                    wcs=wcs,
                    lmax=meta.lmax,
                    lmax_mask=meta.lmax,
                    purify_b=meta.pure_B
                )
                fields[ftag][isfil_tag] = {
                    "spin0": field_spin0,
                    "spin2": field_spin2
                }

        # Computing power spectra
        for ms1, ms2 in ps_pairs:
            preproc_ftag1 = meta.filtering_tag_from_map_set(ms1)
            kspace_tag1 = meta.kspace_tag_from_map_set(ms1)
            preproc_ftag2 = meta.filtering_tag_from_map_set(ms2)
            kspace_tag2 = meta.kspace_tag_from_map_set(ms2)
            if ((preproc_ftag1, kspace_tag1) != ftag1 or (preproc_ftag2, kspace_tag2) != ftag2):  # noqa: E501
                continue

            if verbose:
                print(f" Power spectrum for {ftag1} x {ftag2}")

            pcls_filtered = pu.get_coupled_pseudo_cls(
                fields[ftag1]["filtered"],
                fields[ftag2]["filtered"],
                nmt_bins
            )
            pcls_unfiltered = pu.get_coupled_pseudo_cls(
                fields[ftag1]["unfiltered"],
                fields[ftag2]["unfiltered"],
                nmt_bins
            )
            decoupled_cls_filtered = pu.decouple_pseudo_cls(
                pcls_filtered, inv_couplings["filtered", ms1, ms2]
            )
            decoupled_cls_unfiltered = pu.decouple_pseudo_cls(
                pcls_unfiltered, inv_couplings["unfiltered", ms1, ms2]
            )

            out_f = f"{cls_tf_val_dir}/cls_tf_val_{ftag1[0]}_{ftag1[1]}_x_{ftag2[0]}_{ftag2[1]}_filtered_{id_sim:04d}.npz"  # noqa
            out_unf = f"{cls_tf_val_dir}/cls_tf_val_{ftag1[0]}_{ftag1[1]}_x_{ftag2[0]}_{ftag2[1]}_unfiltered_{id_sim:04d}.npz"  # noqa

            np.savez(out_f, **decoupled_cls_filtered, lb=lb)
            np.savez(out_unf, **decoupled_cls_unfiltered, lb=lb)

        comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
