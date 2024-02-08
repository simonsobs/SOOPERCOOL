import argparse
import healpy as hp
from soopercool.utils import (random_src_mask, get_apodized_mask_from_nhits,
                              get_binary_mask_from_nhits,
                              get_spin_derivatives)
from soopercool import BBmeta
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import urllib.request


def mask_handler(args):
    """
    """
    meta = BBmeta(args.globals)
    mask_dir = meta.mask_directory

    timeout_seconds = 300  # Set the timeout [sec] for the socket

    if args.plots:
        cmap = cm.YlOrRd
        cmap.set_under("w")
        plot_dir = meta.plot_dir_from_output_dir(meta.mask_directory_rel)

    os.makedirs(mask_dir, exist_ok=True)

    # Get nhits map
    # Download nominal hits map from URL with a timeout and save temporarily.
    meta.timer.start("nhits")
    urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
    url = f"{urlpref}norm_nHits_SA_35FOV_ns512.fits"

    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        with open("temp.fits", 'w+b') as f:
            f.write(response.read())
    nhits_nominal = hp.ud_grade(hp.read_map("temp.fits"), meta.nside, power=-2)
    os.remove("temp.fits")

    # If we don't use a custom nhits map, work with the nominal nhits map.
    if not meta.use_input_nhits:
        print("Using nominal apodized mask for analysis")
        nhits = nhits_nominal
    else:
        if not os.path.exists(meta.masks["input_nhits_path"]):
            print("Could not find input nhits map.")
            if meta.filtering_type == "toast":
                print("Get nhits map from provided TOAST schedule.")
                meta.get_nhits_map_from_toast_schedule()
            else:
                raise FileNotFoundError(
                    "Cannot find nhits file {}".format(
                        meta.masks["input_nhits_path"]))
        print("Using custom apodized mask for analysis")
        nhits = meta.read_hitmap()
    meta.save_hitmap(nhits)

    meta.timer.stop("nhits", "Get hits map", args.verbose)

    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(nhits, cmap=cmap, cbar=False)
        hp.graticule()
        nhits_save_path = os.path.join(plot_dir,
                                       meta.masks["nhits_map"])
        plt.savefig(nhits_save_path.replace('.fits', '.png'))

    # Generate binary survey mask from the hits map
    meta.timer.start("binary")
    binary_mask = get_binary_mask_from_nhits(nhits, meta.nside,
                                             zero_threshold=1e-3)
    meta.save_mask("binary", binary_mask, overwrite=True)
    meta.timer.stop("binary", "Computing binary mask", args.verbose)

    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(binary_mask, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(os.path.join(plot_dir,
                                 meta.masks["binary_mask"]).replace('.fits',
                                                                    '.png'))

    # Make nominal apodized mask from the nominal hits map
    meta.timer.start("apodized")
    nominal_mask = get_apodized_mask_from_nhits(
        nhits_nominal, meta.nside,
        galactic_mask=None,
        point_source_mask=None,
        zero_threshold=1e-3, apod_radius=10.0,
        apod_radius_point_source=4.0,
        apod_type="C1"
    )
    first_nom, second_nom = get_spin_derivatives(nominal_mask)
    meta.timer.stop("apodized", "Computing nominal apodized mask",
                    args.verbose)

    if not meta.use_input_nhits:
        final_mask = nominal_mask
        first = first_nom
        second = second_nom

    else:
        # Assemble custom analysis mask from hits map and point source mask
        # stored at disk, and Planck Galactic masks downloaded in-place.

        # Download Galactic mask
        if "galactic" in meta.masks["include_in_mask"]:
            print("Download and save planck galactic masks ...")
            mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
            if not os.path.exists(mask_p15_file):
                urlpref = "http://pla.esac.esa.int/pla/aio/"
                urlpref = f"{urlpref}product-action?MAP.MAP_ID="
                url = f"{urlpref}HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
                with urllib.request.urlopen(url, timeout=timeout_seconds):
                    urllib.request.urlretrieve(url, filename=mask_p15_file)

            # Save different galactic masks
            gal_keys = ["GAL020", "GAL040", "GAL060", "GAL070",
                        "GAL080", "GAL090", "GAL097", "GAL099"]
            for id_key, gal_key in enumerate(gal_keys):
                meta.timer.start(f"gal{id_key}")
                fname = f"{mask_dir}/{meta.masks['galactic_mask_root']}_{gal_key.lower()}.fits"  # noqa
                gal_mask_p15 = hp.read_map(mask_p15_file, field=id_key)
                if not os.path.exists(fname):
                    # Rotate in equatorial coordinates
                    r = hp.Rotator(coord=['G', 'C'])
                    gal_mask_p15 = r.rotate_map_pixel(gal_mask_p15)
                    gal_mask_p15 = hp.ud_grade(gal_mask_p15, meta.nside)
                    gal_mask_p15 = np.where(gal_mask_p15 > 0.5, 1, 0)
                    hp.write_map(
                        fname,
                        gal_mask_p15,
                        overwrite=True,
                        dtype=np.int32
                    )
                meta.timer.stop(f"gal{id_key}",
                                f"Galactic mask {gal_key} projection",
                                args.verbose)

                if args.plots:
                    plt.figure(figsize=(16, 9))
                    hp.mollview(gal_mask_p15, cmap=cmap, cbar=False)
                    hp.graticule()
                    plt.savefig(
                        os.path.join(plot_dir,
                                     fname.split("/")[-1].replace('.fits',
                                                                  '.png'))
                    )

        # Get point source mask
        if "point_source" in meta.masks["include_in_mask"]:
            meta.timer.start("ps_mask")
            ps_fname = meta.point_source_mask_name
            # Load from disk if file exists
            if os.path.isfile(ps_fname):
                ps_mask = binary_mask * hp.ud_grade(hp.read_map(ps_fname),
                                                    meta.nside)
                meta.timer.stop("ps_mask", "Load point source mask from disk",
                                args.verbose)
            # Otherwise, generate random point source mask
            else:
                nsrcs = meta.mock_nsrcs
                mask_radius_arcmin = meta.mock_srcs_hole_radius
                ps_mask = random_src_mask(binary_mask, nsrcs,
                                          mask_radius_arcmin)
                meta.save_mask("point_source", ps_mask, overwrite=True)
                meta.timer.stop("ps_mask", "Generate mock point source mask",
                                args.verbose)

            if args.plots:
                plt.figure(figsize=(16, 9))
                hp.mollview(ps_mask, cmap=cmap, cbar=False)
                hp.graticule()
                plt.savefig(
                    os.path.join(plot_dir,
                                 ps_fname.split("/")[-1].replace('.fits',
                                                                 '.png'))
                )

        # Add the masks
        meta.timer.start("final_mask")
        galactic_mask = None
        point_source_mask = None
        if "galactic" in meta.masks["include_in_mask"]:
            galactic_mask = meta.read_mask("galactic")
        if "point_source" in meta.masks["include_in_mask"]:
            point_source_mask = meta.read_mask("point_source")

        # Combine, apodize, and hits-weight the masks
        final_mask = get_apodized_mask_from_nhits(
            nhits, meta.nside,
            galactic_mask=galactic_mask,
            point_source_mask=point_source_mask,
            zero_threshold=1e-3, apod_radius=meta.masks["apod_radius"],
            apod_radius_point_source=meta.masks["apod_radius_point_source"],
            apod_type=meta.masks["apod_type"]
        )

        # Make sure first two spin derivatives are bounded below twice the
        # respective global maximum values of the nominal analysis mask.
        # If not, issue warning.
        first, second = get_spin_derivatives(final_mask)

        if args.verbose:
            print("---------------------------------------------------------")
            print("Using custom mask. "
                  "Its spin derivatives have global min and max of:")
            print("first:     ", np.amin(first), np.amax(first),
                  "\nsecond:    ", np.amin(second), np.amax(second))
            print("\nFor comparison, the nominal mask has:")
            print("first_nom: ", np.amin(first_nom), np.amax(first_nom),
                  "\nsecond_nom:", np.amin(second_nom), np.amax(second_nom))
            print("---------------------------------------------------------")

        first_is_bounded = (np.amax(first) < 2*np.amax(first_nom)
                            and np.amin(first) > 2*np.amin(first_nom))
        second_is_bounded = (np.amax(second) < 2*np.amax(second_nom)
                             and np.amin(second) > 2*np.amin(second_nom))

        if not (first_is_bounded and second_is_bounded):
            print("WARNING: Your analysis mask may not be smooth enough, "
                  "so B-mode purification could induce biases.")
        meta.timer.stop("final_mask", "Assembling final analysis mask",
                        args.verbose)

    # Save analysis mask
    meta.save_mask("analysis", final_mask, overwrite=True)

    if args.plots:
        # Plot analysis mask
        plt.figure(figsize=(16, 9))
        hp.mollview(final_mask, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(
            os.path.join(plot_dir,
                         meta.masks["analysis_mask"]).replace('.fits',
                                                              '.png')
        )
        plt.clf()

        # Plot first spin derivative of analysis mask
        plt.figure(figsize=(16, 9))
        hp.mollview(first, title="First spin derivative", cmap=cmap,
                    cbar=True)
        hp.graticule()
        plt.savefig(
            os.path.join(plot_dir,
                         meta.masks["analysis_mask"]).replace('.fits',
                                                              '_first.png')
        )
        plt.clf()

        # Plot second spin derivative of analysis mask
        plt.figure(figsize=(16, 9))
        hp.mollview(second, title="Second spin derivative", cmap=cmap,
                    cbar=True)
        hp.graticule()
        plt.savefig(
            os.path.join(plot_dir,
                         meta.masks["analysis_mask"]).replace('.fits',
                                                              '_second.png')
        )
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mask_handler(args)
