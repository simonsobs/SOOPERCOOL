import argparse
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
    meta = BBmeta(args.globals)
    verbose = args.verbose
    out_dir = meta.output_directory

    pcls_tf_est_dir = f"{out_dir}/cells_tf_est"
    BBmeta.make_dir(pcls_tf_est_dir)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)

    mask_file = meta.masks["analysis_mask"]
    if mask_file is not None:
        mask = mu.read_map(mask_file, pix_type=meta.pix_type)
        purify_b = meta.pure_B
    else:
        print("WARNING: The analysis mask is not specified. "
              "Estimating TF sims power spectra with a binary mask "
              "constructed from filtered data. "
              "SWITCHING OFF purification.")
        purify_b = False

    filtering_tags = meta.get_filtering_tags()
    filtering_tag_pairs = meta.get_independent_filtering_pairs()

    if None in filtering_tags and len(filtering_tags) < 1:
        raise ValueError("There must be at least one filter \
                          applied to the data to be able to \
                          compute a transfer function for it")

    tf_settings = meta.transfer_settings

    mpi.init(True)

    for id_sim in mpi.taskrange(tf_settings["tf_est_num_sims"] - 1):

        fields = {
            ftag: {
                "filtered": {},
                "unfiltered": {}
            } for ftag in filtering_tags
        }

        for ftag in filtering_tags:
            if verbose:
                print(f"# {id_sim+1} | {ftag}")
            for pure_type in ["pureT", "pureE", "pureB"]:

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
                    unfiltered_map_file, pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2]
                )
                map_filtered = mu.read_map(
                    filtered_map_file, pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2]
                )
                if mask_file is None:
                    # If analysis_mask is none, compute binary mask on the fly.
                    mask = mu.binary_mask_from_map(map_filtered,
                                                   pix_type=meta.pix_type)
                    mu.plot_map(
                        mask,
                        file_name=f"{out_dir}/binary_mask_{pure_type}",
                        lims=[-1, 1],
                        title=pure_type,
                        pix_type=meta.pix_type
                    )
                    print(
                        f"Plot saved to {out_dir}/binary_mask_{pure_type}.png"
                    )

                wcs = None
                if hasattr(map, 'wcs'):
                    try:
                        wcs = map.wcs
                        nmt.NmtField(mask, None, wcs=wcs)
                    except ValueError:
                        res = 10. * np.pi/180/60
                        _, wcs = enmap.fullsky_geometry(
                            res=res, proj='car', variant='CC'
                        )

                field = {
                    "spin0": nmt.NmtField(mask, map[:1], wcs=wcs),
                    "spin2": nmt.NmtField(mask, map[1:],
                                          purify_b=purify_b, wcs=wcs)
                }

                field_filtered = {
                    "spin0": nmt.NmtField(mask, map_filtered[:1], wcs=wcs),
                    "spin2": nmt.NmtField(mask, map_filtered[1:],
                                          purify_b=purify_b, wcs=wcs)
                }

                fields[ftag]["unfiltered"][pure_type] = field
                fields[ftag]["filtered"][pure_type] = field_filtered

        for ftag1, ftag2 in filtering_tag_pairs:
            if ftag1 is None and ftag2 is None:
                continue

            pcls_mat_filtered = ps_utils.get_pcls_mat_transfer(
                fields[ftag1]["filtered"],
                nmt_bins, fields2=fields[ftag2]["filtered"]
            )
            pcls_mat_unfiltered = ps_utils.get_pcls_mat_transfer(
                fields[ftag1]["unfiltered"],
                nmt_bins, fields2=fields[ftag2]["unfiltered"]
            )

            np.savez(f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_filtered_{id_sim:04d}.npz", # noqa
                     pcls_mat=pcls_mat_filtered)
            np.savez(f"{pcls_tf_est_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_unfiltered_{id_sim:04d}.npz", # noqa
                     pcls_mat=pcls_mat_unfiltered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
