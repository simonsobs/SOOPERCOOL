from soopercool import fft_utils as sfft
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from soopercool import BBmeta
import argparse
import os


def main(args):
    """
    Apply kspace filter to a set of TF estimation and (optionally) TF
    validation simulations.
    
    Loops over all map sets, then multiplies by the binarized analysis mask,
    then applies the kspace filter as indicated in the yaml file under
    transfer_settings['kspace_pars'].
    User can choose to apply the kspace filtering on the (otherwise)
    unfiltered maps to make kspace-only filtered sims, as per the
    '--apply_on_unfiltered' parser.
    Saves output maps to either path indicated under '--out_dir', or else the
    default path from the yaml, "{output_directory}/kspace_filtered_sims".
    """

    rank, size, comm = mpi.init(True)

    meta = BBmeta(args.globals)

    if meta.pix_type == "hp":
        raise NotImplementedError("Only CAR supported for now.")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = f"{meta.output_directory}/kspace_filtered_sims"
    meta.make_dir(out_dir)

    print("apply on unfiltered?", args.apply_on_unfiltered)

    mask = mu.read_map(
        meta.masks["analysis_mask"],
        pix_type=meta.pix_type,
        car_template=meta.car_template
    )

    mask_binary = mu.binarize_mask(mask)
    mask_kspace = mu.binarize_mask(mask)

    # NEW: Accept external kspace mask that is larger than the analysis mask.
    # Crop-smooth its edges and apodize it to make sure the kspace filter
    # does not introduce any biasing artifacts.
    crop_size = 2
    smooth_scale = 2
    apod_radius_deg = 2
    apod_type = "C1"

    if args.kspace_mask is not None:
        mask_kspace = mu.read_map(args.kspace_mask, pix_type=meta.pix_type,
                                  fields_hp=[0])
        mask_kspace = mu.crop_borders(mask_kspace, crop_size, smooth_scale,
                                      pix_type=meta.pix_type)
        mask_kspace = mu.apodize_mask(mask_kspace, apod_radius_deg, apod_type,
                                      pix_type=meta.pix_type)

    tf_settings = meta.transfer_settings
    id_start, n_sims_est = (
        tf_settings["sim_id_start"],
        tf_settings["tf_est_num_sims"],
    )
    do_tf_val = False
    if "validation" in tf_settings:
        if tf_settings["validation"] is not None:
            n_sims_val = tf_settings["tf_val_num_sims"]
            do_tf_val = True

    pure_types = [f"pure{f}" for f in "TEB"]

    files_list = []
    for map_set in meta.map_sets:
        print(f" Processing map set: {map_set}")

        kspace_tag = meta.kspace_tag_from_map_set(map_set)
        ftag = meta.filtering_tag_from_map_set(map_set)
        if kspace_tag is not None:
            print(f"   Applying k-space filter: {kspace_tag}")
            kspace_pars = tf_settings["kspace_pars"][kspace_tag]
            print(f"   k-space filter parameters: {kspace_pars}")

        else:
            print(
                f"   No k-space filter to be applied to map set {map_set}. "
                "Skipping."
            )

        print("  Filtering TF estimation sims")
        f_prefix = {True: "un", False: ""}[args.apply_on_unfiltered]
        map_dir = tf_settings[f"{f_prefix}filtered_map_dir"][ftag]
        for id_sim in range(id_start, id_start + n_sims_est):
            for pure_type in pure_types:
                fname = tf_settings[f"{f_prefix}filtered_map_template"][ftag].format(  # noqa: E501
                    pure_type=pure_type, id_sim=id_sim
                )
                path = f"{map_dir}/{fname}"
                files_list.append((path, kspace_pars, kspace_tag))
        if not do_tf_val:
            continue

        print("  Filtering TF validation sims")
        map_dir = tf_settings["validation"][f"{f_prefix}filtered_map_dir"][ftag]  # noqa: E501
        for id_sim in range(id_start, id_start + n_sims_val):
            for pure_type in pure_types:
                fname = tf_settings["validation"][f"{f_prefix}filtered_map_template"][ftag].format(  # noqa: E501
                    id_sim=id_sim
                )
                path = f"{map_dir}/{fname}"
                files_list.append((path, kspace_pars, kspace_tag))

    # Every rank must have the same list order
    mpi_shared_list = comm.bcast(files_list, root=0)

    task_ids = mpi.distribute_tasks(size, rank, len(files_list))
    local_files_list = [mpi_shared_list[i] for i in task_ids]

    for map_fname, kspace_pars, kspace_tag in local_files_list:
        m = mu.read_map(
            map_fname,
            pix_type=meta.pix_type,
            car_template=meta.car_template,
            fields_hp=[0, 1, 2],
        )
        m *= mask_kspace

        # TODO: need to add a step before to mask noisy edges of the map
        # with bright pixels which makes the filtering more stable
        # Maybe using the binary + galactic mask is enough for this!
        m_filtered = sfft.kspace_filter(
            m,
            pix_type=meta.pix_type,
            **kspace_pars
        )
        fname_out = f"{out_dir}/{os.path.split(map_fname)[-1]}"
        fname_out = fname_out.replace(".fits", f"_kspace_{kspace_tag}.fits")
        mu.write_map(
            fname_out,
            m_filtered * mask_binary,
            pix_type=meta.pix_type,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--globals", help="Path to the soopercool parameter file"
    )
    parser.add_argument(
        "--out_dir",
        help="(Optional) path to output directory for k-filtered sims."
             "Defaults to '{meta.output_directory}/kspace_filtered_sims'.",
        default=None
    )
    parser.add_argument(
        "--apply_on_unfiltered",
        action="store_true",
        help="(Optional) whether to apply kspace filtering on unfiltered "
             "or filtered maps. The latter is used for filter-and-bin maps, "
             "the former may be used for kspace-only TF sims.",
    )
    parser.add_argument(
        "--kspace_mask",
        help="(Optional) path to external mask that will be edge-cropped," \
             " smoothed, apodized and then used to multiply the map by "
             "before kspace filtering.",
        default=None
    )
    args = parser.parse_args()

    main(args)
