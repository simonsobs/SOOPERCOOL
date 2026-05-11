import argparse
import re
from soopercool import BBmeta
from soopercool import map_utils as mu
import numpy as np


def main(args):
    """
    Generates an analysis mask, a binary mask, and a hits map from a given
    set of input bundles and map sets.

    General steps:
    -------------
    1. Compute a global polarization map weighting from either hits or
       polarization weights, summed over all bundles and map sets or.
       Alternatively, use a global external hits map.

    2. Compute a binary mask from the sum of per-bundle and map set hits.
        a) (optional) mutiplying by a box mask

    3. Normalize overall hits

    4. Compute analysis mask by:
        a) (optionally) multiplying the binary mask by Galactic mask
        b) (optionally) multiplying by another external mask
        b) cropping the mask borders
        c) smoothing and apodizing

    5. (Optional) account for point sources by:
        a) apodizing the point source mask
        b) multiplying the analysis mask by the point source mask.

    6. Multiply analysis mask by overall hits map.
    """
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

    ##########
    # HITS MAP
    ##########
    # If a global hits file is indicated in the paramter file, use it.
    if masks_settings["global_hits"] is not None:
        # NOTE: When using a global hits file, we weight the binary mask with
        # the global hits map, but the binary mask will still be defined as
        # the union of all map_set footprints.
        sum_hits = mu.read_map(
            masks_settings["global_hits"],
            pix_type=meta.pix_type,
            car_template=meta.car_template
        )
    else:
        # NOTE: When generating the hits map on the fly, we loop over the
        # (map_set, id_bundles) pairs to define a common hits map as the
        # sum of per-(map_set, bundle) hits (or weights).
        hit_maps = []
        for map_set in meta.map_sets_list:
            n_bundles = meta.n_bundles_from_map_set(map_set)
            for id_bundle in range(n_bundles):
                map_dir = meta.map_dir_from_map_set(map_set)
                map_template = meta.map_template_from_map_set(map_set)
                map_file = map_template.replace(
                    "{id_bundle}",
                    str(id_bundle)
                )
                type_options = [
                    f for f in re.findall(r"\{.*?\}", map_template)
                    if "|" in f
                ]
                if not type_options:
                    raise ValueError(
                        "The map directory must contain maps, "
                        "hits, and weights files, indicated by a "
                        "corresponding suffix."
                    )
                else:
                    # Select the hits (or weights) map
                    option = type_options[0].replace("{", "")
                    option = option.replace("}", "").split("|")[1]

                    map_file = map_file.replace(
                        type_options[0],
                        option
                    )
                    if masks_settings["use_weights"]:
                        # Use weights instead of hits
                        map_file = map_file.replace(
                            "hits",
                            "weights"
                        )

                print(f"Reading map for {map_set} - bundle {id_bundle}")
                if verbose:
                    print(f"    file_name: {map_dir}/{map_file}")

                # Raise an error if file does not exists
                try:
                    hits = mu.read_map(
                        f"{map_dir}/{map_file}",
                        pix_type=meta.pix_type,
                        car_template=meta.car_template
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"File {map_dir}/{map_file} not found. "
                        "Please check the map directory and template "
                        "in the parameter file or provide a global hits file."
                    )
                hit_maps.append(hits)

        if masks_settings["use_weights"]:
            # NOTE: When using weights, create a common weighting by summing
            # the per-(map_set, bundle) weights, taking the square root of
            # weightQ**2 + weightU**2 and discarding all pixels with values
            # below its median value.
            sum_hits = hit_maps[0][1].copy() * 0.
            for h in hit_maps:
                sum_hits += np.sqrt(h[1]**2 + h[2]**2)
            threshold = np.median(sum_hits[sum_hits > 0])
            sum_hits[sum_hits < threshold] = 0
        else:
            sum_hits = hit_maps[0].copy() * 0.
            for h in hit_maps:
                sum_hits += h

    #############
    # BINARY MASK
    #############
    # NOTE: The binary mask is
    # generated from the sum of the per-(map_set, bundle) hits.
    binary = sum_hits.copy()
    binary[:] = 1

    for map_set in meta.map_sets_list:
        n_bundles = meta.n_bundles_from_map_set(map_set)
        for id_bundle in range(n_bundles):
            map_dir = meta.map_dir_from_map_set(map_set)
            map_template = meta.map_template_from_map_set(map_set)
            map_file = map_template.replace(
                "{id_bundle}",
                str(id_bundle)
            )
            type_options = [
                f for f in re.findall(r"\{.*?\}", map_template)
                if "|" in f
            ]
            if not type_options:
                raise ValueError(
                    "The map directory must contain both maps "
                    "and hits files, indicated by a "
                    "corresponding suffix."
                )
            else:
                # Select the map
                option = type_options[0].replace("{", "")
                option = option.replace("}", "").split("|")[0]

                map_file = map_file.replace(
                    type_options[0],
                    option
                )

                print(f"Reading map for {map_set} - bundle {id_bundle}")
                if verbose:
                    print(f"    file_name: {map_dir}/{map_file}")

                # Raise an error if file does not exists
                try:
                    map = mu.read_map(
                        f"{map_dir}/{map_file}",
                        pix_type=meta.pix_type,
                        car_template=meta.car_template
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"File {map_dir}/{map_file} not found. "
                        "Please check the map directory and template "
                        "in the parameter file."
                    )
            # Select only the pixels that are observed
            # in this map_set and bundle
            binary[map[1] == 0] = 0

    if masks_settings["box_mask"] is not None:
        # Apply a rectangular box mask to the binary mask and the hits map.
        binary = mu.apply_box_mask(binary, masks_settings["box_mask"])
        sum_hits = mu.apply_box_mask(sum_hits, masks_settings["box_mask"])

    ###############
    # NORMALIZATION
    ###############
    # Normalize hitmaps to be bound by [0, 1].
    if masks_settings["use_weights"] and not masks_settings["global_hits"]:
        sum_hits = (sum_hits - np.amin(sum_hits))
        sum_hits /= (np.amax(sum_hits) - np.amin(sum_hits))
    else:
        sum_hits /= np.amax(sum_hits)

    # Calculate and print the fraction of the sky covered by the mask
    fsky_hits = mu.get_fsky_from_hits(sum_hits)
    fsky_bin = mu.get_fsky_from_hits(binary)
    print(f"Fraction of the sky covered: {fsky_hits:.3e} (hits), "
          f"{fsky_bin:.3e} (binary)")

    # Save products
    mu.write_map(
        f"{masks_dir}/binary_mask.fits",
        binary,
        dtype=np.int32,
        pix_type=meta.pix_type
    )
    mu.write_map(
        f"{masks_dir}/normalized_hits.fits",
        sum_hits,
        dtype=np.float32,
        pix_type=meta.pix_type
    )

    if do_plots:
        mu.plot_map(
            binary,
            title="Binary mask",
            file_name=f"{plot_dir}/binary_mask",
            pix_type=meta.pix_type,
            lims=[-binary.max(), binary.max()]
        )
        mu.plot_map(
            sum_hits,
            title="Normalized hits",
            file_name=f"{plot_dir}/normalized_hits",
            pix_type=meta.pix_type,
            lims=[-1, 1]
        )

    ###############
    # ANALYSIS MASK
    ###############
    analysis_mask = binary.copy()

    if masks_settings["galactic_mask"] is not None:
        print("Reading galactic mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['galactic_mask']}")
        gal_mask = mu.read_map(masks_settings["galactic_mask"],
                               pix_type=meta.pix_type,
                               geometry=analysis_mask.geometry)
        gal_mask[gal_mask != 0] = 1.
        if do_plots:
            mu.plot_map(
                gal_mask,
                title="Galactic mask",
                file_name=f"{plot_dir}/galactic_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1]
            )
        analysis_mask *= gal_mask
        if do_plots:
            mu.plot_map(
                analysis_mask,
                title="Analysis mask after galactic mask",
                file_name=f"{plot_dir}/binary_galactic_mask",
                pix_type=meta.pix_type,
                lims=[-analysis_mask.max(), analysis_mask.max()]
            )

    if masks_settings["external_mask"] is not None:
        print("Reading external mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['external_mask']}")
        ext_mask = mu.read_map(masks_settings["external_mask"],
                               pix_type=meta.pix_type,
                               geometry=analysis_mask.geometry)
        if do_plots:
            mu.plot_map(
                ext_mask,
                title="External mask",
                file_name=f"{plot_dir}/external_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1]
            )
        analysis_mask *= ext_mask
        if do_plots:
            mu.plot_map(
                analysis_mask,
                title="Analysis mask after external mask",
                file_name=f"{plot_dir}/binary_external_mask",
                pix_type=meta.pix_type,
                lims=[-analysis_mask.max(), analysis_mask.max()]
            )

    # Crop binary mask to make it more "continuous"
    analysis_mask = mu.crop_borders(
        analysis_mask,
        crop_size=2.0,
        smooth_scale=0.7,
        pix_type=meta.pix_type
    )
    if do_plots:
        mu.plot_map(
            analysis_mask,
            title="Analysis mask after cropping borders",
            file_name=f"{plot_dir}/binary_galactic_cropped_mask",
            pix_type=meta.pix_type,
            lims=[-analysis_mask.max(), analysis_mask.max()]
        )

    # Smooth and apodize analysis mask
    analysis_mask = mu.smooth_map(
        analysis_mask,
        fwhm_deg=masks_settings['smoothing_radius'],
        pix_type=meta.pix_type
    )
    analysis_mask = mu.apodize_mask(
        analysis_mask,
        apod_radius_deg=masks_settings["apod_radius"],
        apod_type=masks_settings["apod_type"],
        pix_type=meta.pix_type
    )
    if do_plots:
        mu.plot_map(
            analysis_mask,
            title="Analysis mask after apodization",
            file_name=f"{plot_dir}/binary_galactic_cropped_apodized_mask",
            pix_type=meta.pix_type,
            lims=[-analysis_mask.max(), analysis_mask.max()]
        )

    ###################
    # POINT SOURCE MASK
    ###################
    if masks_settings["point_source_mask"] is not None:
        print("Reading point source mask ...")
        if verbose:
            print(f"    file_name: {masks_settings['point_source_mask']}")
        ps_mask = mu.read_map(masks_settings["point_source_mask"],
                              pix_type=meta.pix_type,
                              geometry=analysis_mask.geometry)
        ps_mask = mu.apodize_mask(
            ps_mask,
            apod_radius_deg=masks_settings["apod_radius_point_source"],
            apod_type=masks_settings["apod_type"],
            pix_type=meta.pix_type
        )
        if do_plots:
            mu.plot_map(
                ps_mask,
                title="Point source mask",
                file_name=f"{plot_dir}/point_source_mask",
                pix_type=meta.pix_type,
                lims=[-1, 1]
            )

        analysis_mask *= ps_mask
        if do_plots:
            mu.plot_map(
                analysis_mask,
                title="Analysis mask after point source mask",
                file_name=f"{plot_dir}/binary_galactic_cropped_apodized_ps_mask", # noqa
                pix_type=meta.pix_type,
                lims=[-analysis_mask.max(), analysis_mask.max()]
            )

    ############
    # FINAL MASK
    ############
    # Weight with hits map
    analysis_mask *= sum_hits
    mu.write_map(
        f"{masks_dir}/analysis_mask.fits",
        analysis_mask,
        pix_type=meta.pix_type
    )

    if do_plots:
        mu.plot_map(
            analysis_mask,
            title="Analysis mask",
            file_name=f"{plot_dir}/analysis_mask",
            pix_type=meta.pix_type,
            lims=[-1, 1]
        )

    ###################
    # QUICK DIAGNOSTICS
    ###################
    # Compute and plot spin derivatives
    first, second = mu.get_spin_derivatives(analysis_mask)

    if do_plots:
        mu.plot_map(
            first,
            title="First spin derivative",
            pix_type=meta.pix_type,
            file_name=f"{plot_dir}/first_spin_derivative"
        )
        mu.plot_map(
            second,
            title="Second spin derivative",
            pix_type=meta.pix_type,
            file_name=f"{plot_dir}/second_spin_derivative"
        )

    if args.verbose:
        print("---------------------------------------------------------")
        print("Using custom mask. "
              "Its spin derivatives have global min and max of:")
        print(f"first:     {np.amin(first):.2e}, {np.amax(first):.2e}",
              f"\nsecond:    {np.amin(second):.2e}, {np.amax(second):.2e}")
        print("---------------------------------------------------------")

    print("\nSUMMARY")
    print("-------")
    print(f"Wrote analysis mask to {masks_dir}/analysis_mask.fits")
    if do_plots:
        print("Plot directory:", plot_dir)
    print("Parameters")
    print(f"    Galactic mask: {masks_settings['galactic_mask']}")
    print(f"    External mask: {masks_settings['external_mask']}")
    print(f"    Point source mask: {masks_settings['point_source_mask']}")
    print(f"    Apodization type: {masks_settings['apod_type']}")
    print(f"    Apodization radius: {masks_settings['apod_radius']}")
    print(f"    Apodization radius point source: {masks_settings['apod_radius_point_source']}")  # noqa: E501


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get analysis mask")
    parser.add_argument("--globals", help="Path to the paramfile")
    parser.add_argument("--verbose", help="Verbose mode",
                        action="store_true")
    parser.add_argument("--no-plots", help="Plot the results",
                        action="store_true")
    args = parser.parse_args()

    main(args)
