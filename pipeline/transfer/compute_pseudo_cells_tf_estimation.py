import argparse
import os
from soopercool import BBmeta
import pymaster as nmt
import numpy as np
from soopercool import ps_utils
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

    do_plot = not args.no_plots
    if do_plot:
        plot_dir = f"{out_dir}/plots/cells_tf_est"
        BBmeta.make_dir(plot_dir)

    pcls_tf_est_dir = f"{out_dir}/cells_tf_est"
    BBmeta.make_dir(pcls_tf_est_dir)

    nmt_bins = meta.read_nmt_binning()
    lb = nmt_bins.get_effective_ells()

    mask_file = meta.masks["analysis_mask"]
    if mask_file is not None:
        mask = mu.read_map(mask_file,
                           pix_type=meta.pix_type,
                           car_template=meta.car_template)
        purify_b = meta.pure_B
    else:
        print("WARNING: The analysis mask is not specified. "
              "Estimating TF sims power spectra with a binary mask "
              "constructed from filtered data. "
              "SWITCHING OFF purification.")
        purify_b = False

    lmax = mu.lmax_from_map(
        meta.masks["analysis_mask"],
        pix_type=meta.pix_type
    )
    if meta.lmax > lmax:
        raise ValueError(
            f"Specified lmax {meta.lmax} is larger than "
            f"the maximum lmax from map resolution {lmax}"
        )

    filtering_tags = meta.get_filtering_tags()
    filtering_tag_pairs = meta.get_independent_filtering_pairs()

    if None in filtering_tags and len(filtering_tags) < 1:
        raise ValueError("There must be at least one filter \
                          applied to the data to be able to \
                          compute a transfer function for it")

    tf_settings = meta.transfer_settings
    sim_ids = range(tf_settings["sim_id_start"],
                    tf_settings["sim_id_start"]+tf_settings["tf_est_num_sims"])

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
        ftags_unique = list(dict.fromkeys([ftag1, ftag2]))
        fields = {
            ftag: {
                "filtered": {},
                "unfiltered": {}
            } for ftag in ftags_unique
        }

        for pure_type in ["pureT", "pureE", "pureB"]:
            for ftag in ftags_unique:

                unfiltered_map_dir = tf_settings["unfiltered_map_dir"][ftag]
                unfiltered_map_tmpl = tf_settings["unfiltered_map_template"][ftag] # noqa

                unfiltered_map_file = unfiltered_map_tmpl.format(
                    id_sim=id_sim, pure_type=pure_type
                )
                unfiltered_map_file = f"{unfiltered_map_dir}/{unfiltered_map_file}" # noqa

                filtered_map_dir = tf_settings["filtered_map_dir"][ftag]
                filtered_map_tmpl = tf_settings["filtered_map_template"][ftag]
                filtered_map_file = filtered_map_tmpl.format(
                    id_sim=id_sim, pure_type=pure_type
                )
                filtered_map_file = f"{filtered_map_dir}/{filtered_map_file}"

                map = mu.read_map(
                    unfiltered_map_file,
                    pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2],
                    car_template=meta.car_template
                )
                map_filtered = mu.read_map(
                    filtered_map_file,
                    pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2],
                    car_template=meta.car_template
                )

                if mask_file is None:
                    # If analysis_mask is none, compute binary mask on the fly.
                    weights_file = filtered_map_file.replace(".fits",
                                                             "_weights.fits")
                    if os.path.isfile(weights_file):
                        mask = mu.binary_mask_from_map(
                            weights_file, pix_type=meta.pix_type
                        )
                    else:
                        raise FileNotFoundError("File does not exist: "
                                                f"{weights_file}")

                    mu.plot_map(
                        mask,
                        file_name=f"{out_dir}/binary_mask_{pure_type}",
                        lims=[-1, 1],
                        title=pure_type,
                        pix_type=meta.pix_type
                    )
                    mu.write_map(
                        f"{out_dir}/binary_mask_{pure_type}_sim{id_sim:04d}_{ftag}.fits",  # noqa
                        mask, pix_type=meta.pix_type
                    )
                    print(
                        f"Mask saved to {out_dir}/binary_mask_{pure_type}_sim{id_sim:04d}_{ftag}.png"  # noqa
                    )

                wcs = None
                if hasattr(map, 'wcs'):
                    # This is a patch.
                    # Reproject mask and maps onto template geometry.
                    tshape, twcs = enmap.read_map_geometry(meta.car_template)
                    shape, wcs = enmap.overlap(map.shape, map.wcs, tshape,
                                               twcs)
                    shape, wcs = enmap.overlap(map_filtered.shape,
                                               map_filtered.wcs, shape, wcs)
                    shape, wcs = enmap.overlap(mask.shape, mask.wcs, shape,
                                               wcs)
                    flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)

                    map = enmap.insert(flat_template.copy(), map)
                    map_filtered = enmap.insert(flat_template.copy(),
                                                map_filtered)
                    mask = enmap.insert(flat_template[0].copy(), mask)

                    # Deal with possibly missing atomics => different footprint
                    mask_restrict = map_filtered.copy()
                    mask_restrict = mask_restrict[
                        ["pureT", "pureE", "pureB"].index(pure_type)
                    ]
                    mask_restrict[mask_restrict != 0] = 1.
                    mask *= mask_restrict
                    _, wcs = enmap.read_map_geometry(meta.car_template)

                field = {
                    "spin0": nmt.NmtField(
                        mask,
                        map[:1],
                        wcs=wcs,
                        lmax=meta.lmax
                    ),
                    "spin2": nmt.NmtField(
                        mask,
                        map[1:],
                        purify_b=purify_b,
                        wcs=wcs,
                        lmax=meta.lmax
                    )
                }
                field_filtered = {
                    "spin0": nmt.NmtField(
                        mask,
                        map_filtered[:1],
                        wcs=wcs,
                        lmax=meta.lmax
                    ),
                    "spin2": nmt.NmtField(
                        mask,
                        map_filtered[1:],
                        purify_b=purify_b,
                        wcs=wcs,
                        lmax=meta.lmax
                    )
                }

                fields[ftag]["unfiltered"][pure_type] = field
                fields[ftag]["filtered"][pure_type] = field_filtered

        if ftag1 is None and ftag2 is None:
            continue

        pcls_mat_filtered, pcls_mat_filtered_unbinned = \
            ps_utils.get_pcls_mat_transfer(
                fields[ftag1]["filtered"],
                nmt_bins, fields2=fields[ftag2]["filtered"],
                return_unbinned=True
            )
        pcls_mat_unfiltered, pcls_mat_unfiltered_unbinned = \
            ps_utils.get_pcls_mat_transfer(
                fields[ftag1]["unfiltered"],
                nmt_bins, fields2=fields[ftag2]["unfiltered"],
                return_unbinned=True
            )

        out_f = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_filtered_{id_sim:04d}.npz"  # noqa
        out_unf = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_unfiltered_{id_sim:04d}.npz"  # noqa

        if do_plot:
            fplt = f"{plot_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_{id_sim:04d}.pdf"  # noqa
            ps_utils.plot_pcls_mat_transfer(
                pcls_mat_unfiltered, pcls_mat_filtered, lb, fplt,
                lmax=600
            )

        np.savez(out_f, pcls_mat=pcls_mat_filtered)
        np.savez(out_unf, pcls_mat=pcls_mat_unfiltered)

        out_f = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_filtered_unbinned_{id_sim:04d}.npz"  # noqa
        out_unf = f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_unfiltered_unbinned_{id_sim:04d}.npz"  # noqa
        np.savez(out_f, pcls_mat=pcls_mat_filtered_unbinned)
        np.savez(out_unf, pcls_mat=pcls_mat_unfiltered_unbinned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    main(args)
