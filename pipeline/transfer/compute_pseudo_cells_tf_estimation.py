import argparse
from soopercool import BBmeta
import healpy as hp
import pymaster as nmt
import numpy as np
from soopercool import ps_utils


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    out_dir = meta.output_directory

    pcls_tf_est_dir = f"{out_dir}/cells_tf_est"
    BBmeta.make_dir(pcls_tf_est_dir)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)

    mask_file = meta.masks["analysis_mask"]
    mask = hp.read_map(mask_file)

    filtering_tags = meta.get_filtering_tags()
    filtering_tag_pairs = meta.get_independent_filtering_pairs()

    tf_settings = meta.transfer_settings

    for id_sim in range(tf_settings["tf_est_num_sims"]):

        fields = {
            ftag: {
                "filtered": {},
                "unfiltered": {}
            } for ftag in filtering_tags
        }

        for ftag in filtering_tags:
            for pure_type in ["pureT", "pureE", "pureB"]:

                filtered_map_dir = tf_settings["filtered_map_dir"][ftag]
                unfiltered_map_dir = tf_settings["unfiltered_map_dir"][ftag]

                filtered_map_tmpl = tf_settings["filtered_map_template"][ftag]
                unfiltered_map_tmpl = tf_settings["unfiltered_map_template"][ftag] # noqa

                filtered_map_file = filtered_map_tmpl.format(
                    id_sim=id_sim, pure_type=pure_type
                )
                unfiltered_map_file = unfiltered_map_tmpl.format(
                    id_sim=id_sim, pure_type=pure_type
                )

                filtered_map_file = f"{filtered_map_dir}/{filtered_map_file}"
                unfiltered_map_file = f"{unfiltered_map_dir}/{unfiltered_map_file}" # noqa

                map = hp.read_map(filtered_map_file,
                                  field=[0, 1, 2])
                map_filtered = hp.read_map(unfiltered_map_file,
                                           field=[0, 1, 2])

                field = {
                    "spin0": nmt.NmtField(mask, map[:1]),
                    "spin2": nmt.NmtField(mask, map[1:],
                                          purify_b=meta.pure_B)
                }

                field_filtered = {
                    "spin0": nmt.NmtField(mask, map_filtered[:1]),
                    "spin2": nmt.NmtField(mask, map_filtered[1:],
                                          purify_b=meta.pure_B)
                }

                fields[ftag]["unfiltered"][pure_type] = field
                fields[ftag]["filtered"][pure_type] = field_filtered

        for ftag1, ftag2 in filtering_tag_pairs:
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
    args = parser.parse_args()
    main(args)
