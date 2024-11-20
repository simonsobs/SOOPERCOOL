import argparse
import re
from soopercool import BBmeta
from soopercool import map_utils as mu
from soopercool import utils as su
import numpy as np


def main(args):
    """ """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory

    masks_dir = f"{out_dir}/masks"
    plot_dir = f"{out_dir}/plots/masks"
    BBmeta.make_dir(masks_dir)
    if do_plots:
        BBmeta.make_dir(plot_dir)

    masks_settings = meta.masks

    # If a global hits file is indicated in the paramter file, use it.
    if masks_settings["global_hits_file_overwrite"] is not None:
        sum_hits = mu.read_map(
            masks_settings["global_hits_file_overwrite"], pix_type=meta.pix_type
        )
        # Create binary
        binary = sum_hits.copy()
        binary[:] = 1
        binary[sum_hits == 0] = 0
    else:
        # Loop over the (map_set, id_bundles)
        # pairs to define a common binary mask
        hit_maps = []
        for map_set in meta.map_sets_list:
            n_bundles = meta.n_bundles_from_map_set(map_set)
            for id_bundle in range(n_bundles):

                map_dir = meta.map_dir_from_map_set(map_set)
                map_template = meta.map_template_from_map_set(map_set)
                map_file = map_template.replace("{id_bundle}", str(id_bundle))
                type_options = [
                    f for f in re.findall(r"\{.*?\}", map_template) if "|" in f
                ]
                if not type_options:
                    raise ValueError(
                        "The map directory must contain both maps "
                        "and hits files, indicated by a "
                        "corresponding suffix."
                    )
                else:
                    # Select the hitmap
                    option = type_options[0].replace("{", "")
                    option = option.replace("}", "").split("|")[1]

                type_options = [
                    f for f in re.findall(r"\{.*?\}", map_template) if "|" in f
                ]
                if not type_options:
                    raise ValueError(
                        "The map directory must contain both maps "
                        "and hits files, indicated by a "
                        "corresponding suffix."
                    )
                else:
                    # Select the hitmap
                    option = type_options[0].replace("{", "")
                    option = option.replace("}", "").split("|")[1]

                    map_file = map_file.replace(type_options[0], option)

                print(f"Reading hitmap for {map_set} - bundle {id_bundle}")
                if verbose:
                    print(f"    file_name: {map_dir}/{map_file}")

                hits = mu.read_map(f"{map_dir}/{map_file}", pix_type=meta.pix_type)
                hit_maps.append(hits)

        # Create binary and normalized hitmap
        binary = hit_maps[0].copy()
        sum_hits = hit_maps[0].copy()
        binary[:] = 1
        sum_hits[:] = 0
        for hit_map in hit_maps:
            binary[hit_map == 0] = 0
            sum_hits += hit_map
        sum_hits[binary == 0] = 0

    # Normalize and smooth hitmaps
    sum_hits = mu.smooth_map(sum_hits, fwhm_deg=1, pix_type=meta.pix_type)
    sum_hits /= np.amax(sum_hits)

    # Save products
    mu.write_map(
        f"{masks_dir}/binary_mask.fits", binary, dtype=np.int32, pix_type=meta.pix_type
    )
    mu.write_map(
        f"{masks_dir}/normalized_hits.fits",
        sum_hits,
        dtype=np.float32,
        pix_type=meta.pix_type,
    )

    if do_plots:
        print("np.shape(binary)", np.shape(binary))
        mu.plot_map(
            binary,
            title="Binary mask",
            file_name=f"{plot_dir}/binary_mask",
            pix_type=meta.pix_type,
            lims=[-binary.max(), binary.max()],
        )

        mu.plot_map(
            sum_hits,
            title="Normalized hits",
            file_name=f"{plot_dir}/normalized_hits",
            pix_type=meta.pix_type,
            lims=[-1, 1],
        )

    analysis_mask = binary.copy()

    if masks_settings["galactic_mask"] is not None:
        print("Reading galactic mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['galactic_mask']}")
        gal_mask = mu.read_map(
            masks_settings["galactic_mask"],
            pix_type=meta.pix_type,
            geometry=analysis_mask.geometry,
        )
        if do_plots:
            mu.plot_map(
                gal_mask,
                title="Galactic mask",
                file_name=f"{plot_dir}/galactic_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1],
            )
        analysis_mask *= gal_mask

    if masks_settings["external_mask"] is not None:
        print("Reading external mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['external_mask']}")
        ext_mask = mu.read_map(
            masks_settings["external_mask"],
            pix_type=meta.pix_type,
            geometry=analysis_mask.geometry,
        )
        if do_plots:
            mu.plot_map(
                ext_mask,
                title="External mask",
                file_name=f"{plot_dir}/external_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1],
            )
        analysis_mask *= ext_mask

    analysis_mask = mu.apodize_mask(
        analysis_mask,
        apod_radius_deg=masks_settings["apod_radius"],
        apod_type=masks_settings["apod_type"],
        pix_type=meta.pix_type,
    )

    if masks_settings["point_source_mask"] is not None:
        print("Reading point source mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['point_source_mask']}")
        ps_mask = mu.read_map(
            masks_settings["point_source_mask"],
            pix_type=meta.pix_type,
            geometry=analysis_mask.geometry,
        )
        ps_mask = mu.apodize_mask(
            ps_mask,
            apod_radius_deg=masks_settings["apod_radius_point_source"],
            apod_type=masks_settings["apod_type"],
            pix_type=meta.pix_type,
        )
        if do_plots:
            mu.plot_map(
                ps_mask,
                title="Point source mask",
                file_name=f"{plot_dir}/point_source_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1],
            )

        analysis_mask *= ps_mask

    # Weight with hitmap
    analysis_mask *= sum_hits
    mu.write_map(
        f"{masks_dir}/analysis_mask.fits", analysis_mask, pix_type=meta.pix_type
    )

    if do_plots:
        mu.plot_map(
            analysis_mask,
            title="Analysis mask",
            file_name=f"{plot_dir}/analysis_mask",
            pix_type=meta.pix_type,
            lims=[-1, 1],
        )

    # Compute and plot spin derivatives
    first, second = su.get_spin_derivatives(analysis_mask)

    if do_plots:
        mu.plot_map(
            first,
            title="First spin derivative",
            file_name=f"{plot_dir}/first_spin_derivative",
        )
        mu.plot_map(
            second,
            title="Second spin derivative",
            file_name=f"{plot_dir}/second_spin_derivative",
        )

    if args.verbose:
        print("---------------------------------------------------------")
        print("Using custom mask. " "Its spin derivatives have global min and max of:")
        print(
            "first:     ",
            np.amin(first),
            np.amax(first),
            "\nsecond:    ",
            np.amin(second),
            np.amax(second),
        )
        print("---------------------------------------------------------")

    print("\nSUMMARY")
    print("-------")
    print(f"Wrote analysis mask to {masks_dir}/analysis_mask.fits")
    print("Parameters")
    print(f"    Galactic mask: {masks_settings['galactic_mask']}")
    print(f"    External mask: {masks_settings['external_mask']}")
    print(f"    Point source mask: {masks_settings['point_source_mask']}")
    print(f"    Apodization type: {masks_settings['apod_type']}")
    print(f"    Apodization radius: {masks_settings['apod_radius']}")
    print(
        f"    Apodization radius point source: {masks_settings['apod_radius_point_source']}"
    )  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get analysis mask")
    parser.add_argument("--globals", help="Path to the paramfile")
    parser.add_argument("--verbose", help="Verbose mode", action="store_true")
    parser.add_argument("--no-plots", help="Plot the results", action="store_true")

    args = parser.parse_args()

    main(args)
