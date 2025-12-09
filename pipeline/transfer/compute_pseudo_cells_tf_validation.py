import argparse
from soopercool import BBmeta
import pymaster as nmt
import numpy as np
from soopercool import ps_utils as pu
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from pixell import enmap


def main(args):
    """
    """
    rank, size, comm = mpi.init(True)

    meta = BBmeta(args.globals)
    verbose = args.verbose
    out_dir = meta.output_directory

    cls_tf_val_dir = f"{out_dir}/cells_tf_val"
    BBmeta.make_dir(cls_tf_val_dir)

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()
    ps_pairs = meta.get_ps_names_list(type="all", coadd=True)

    inv_couplings_filtered = meta.get_inverse_couplings()["filtered"]
    inv_couplings_unfiltered = meta.get_inverse_couplings()["unfiltered"]

    simdir_unfiltered = meta.transfer_settings["unfiltered_map_dir"]
    if "validation" not in simdir_unfiltered:
        raise KeyError("Unfiltered validation sims must be provided.")
    simdir_unfiltered = simdir_unfiltered["validation"]

    simdir_filtered = meta.transfer_settings["filtered_map_dir"]
    if "validation" not in simdir_filtered:
        raise KeyError("Filtered validation sims must be provided.")
    simdir_filtered = simdir_filtered["validation"]

    mask_file = meta.masks["analysis_mask"]
    if mask_file is None:
        raise ValueError("An analysis mask must be provided.")
    mask = mu.read_map(mask_file,
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)

    tf_settings = meta.transfer_settings
    sim_id_start = 0 if "sim_id_start" not in tf_settings else tf_settings["sim_id_start"]  # noqa
    sim_ids = range(sim_id_start, tf_settings["tf_val_num_sims"]+sim_id_start)

    mpi_shared_list = sim_ids
    # Every rank must have the same list order
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)

    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim in local_mpi_list:
        if verbose:
            print(f"id_sim {id_sim}")

        # Create namaster fields
        fields = {"filtered": {}, "unfiltered": {}}

        for map_set in meta.map_sets_list:
            if verbose:
                print(f" Field for {map_set}")
            unfiltered_map_dir = simdir_unfiltered[map_set]
            unfiltered_map_tmpl = tf_settings["unfiltered_map_template"]["validation"][map_set] # noqa
            unfiltered_map_file = unfiltered_map_tmpl.format(id_sim=id_sim)
            unfiltered_map_file = f"{unfiltered_map_dir}/{unfiltered_map_file}"

            filtered_map_dir = simdir_filtered[map_set]
            filtered_map_tmpl = tf_settings["filtered_map_template"]["validation"][map_set] # noqa
            filtered_map_file = filtered_map_tmpl.format(id_sim=id_sim)
            filtered_map_file = f"{filtered_map_dir}/{filtered_map_file}"

            m_unf = mu.read_map(
                unfiltered_map_file,
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
                convert_K_to_muK=True
            )
            m_f = mu.read_map(
                filtered_map_file,
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
                convert_K_to_muK=True
            )
            for f_tag, m in zip(["filtered", "unfiltered"], [m_f, m_unf]):
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

                field_spin0 = nmt.NmtField(mask_restrict, m[:1], wcs=wcs,
                                           lmax=meta.lmax)
                field_spin2 = nmt.NmtField(mask_restrict, m[1:], wcs=wcs,
                                           lmax=meta.lmax(),
                                           purify_b=meta.pure_B)
                fields[f_tag][map_set] = {
                    "spin0": field_spin0,
                    "spin2": field_spin2
                }

        # Computing power spectra
        for ms1, ms2 in ps_pairs:
            if verbose:
                print(f" Power spectrum for {ms1} x {ms2}")

            pcls_filtered = pu.get_coupled_pseudo_cls(
                fields["filtered"][ms1],
                fields["filtered"][ms2],
                nmt_bins
            )
            pcls_unfiltered = pu.get_coupled_pseudo_cls(
                fields["unfiltered"][ms1],
                fields["unfiltered"][ms2],
                nmt_bins
            )
            decoupled_cls_filtered = pu.decouple_pseudo_cls(
                pcls_filtered, inv_couplings_filtered[ms1, ms2]
            )
            decoupled_cls_unfiltered = pu.decouple_pseudo_cls(
                pcls_unfiltered, inv_couplings_unfiltered[ms1, ms2]
            )

            out_f = f"{cls_tf_val_dir}/cls_tf_val_{ms1}_x_{ms2}_filtered_{id_sim:04d}.npz"  # noqa
            out_unf = f"{cls_tf_val_dir}/cls_tf_val_{ms1}_x_{ms2}_unfiltered_{id_sim:04d}.npz"  # noqa

            np.savez(out_f, **decoupled_cls_filtered, lb=lb)
            np.savez(out_unf, **decoupled_cls_unfiltered, lb=lb)

        comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
