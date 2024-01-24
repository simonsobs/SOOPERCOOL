import argparse
import healpy as hp
from soopercool.utils import (random_src_mask, get_apodized_mask_from_nhits,
                              get_binary_mask_from_nhits)
from soopercool import BBmeta
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import urllib.request
import pathlib


def mask_handler(args):
    """
    """
    meta = BBmeta(args.globals)
    mask_dir = meta.mask_directory

    timeout_seconds = 300  # Set the timeout [sec] for the socket

    if args.plots:
        cmap = cm.YlOrRd
        cmap.set_under("w")

    os.makedirs(mask_dir, exist_ok=True)

    print("Download and save hits map ...")
    nhits_file = meta.hitmap_file
    # Create hitmap_file directory if not exist
    pathlib.Path(nhits_file).parent.mkdir(parents=True, exist_ok=True)
    # Download hits map
    urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
    url = f"{urlpref}norm_nHits_SA_35FOV_ns512.fits"
    # Open the URL with a timeout
    with urllib.request.urlopen(url, timeout=timeout_seconds):
        urllib.request.urlretrieve(url, filename=nhits_file)
        nhits = hp.ud_grade(hp.read_map(nhits_file), meta.nside, power=-2)
        hp.write_map(meta.hitmap_file, nhits, overwrite=True)
    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(nhits, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(meta.binary_mask_name.replace('.fits', '_nhits.png'))

    # Generate binary survey mask from the hits map
    meta.timer.start("Computing binary mask")
    nhits = hp.read_map(nhits_file)
    binary_mask = get_binary_mask_from_nhits(nhits, meta.nside,
                                             zero_threshold=1e-3)
    meta.save_mask("binary", binary_mask, overwrite=True)
    meta.timer.stop("Computing binary mask", args.verbose)

    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(binary_mask, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(meta.binary_mask_name.replace('.fits', '.png'))

    if not args.self_assemble:
        # Download SAT apodized mask used in the SO BB
        # pipeline paper (https://arxiv.org/abs/2302.04276)
        # and use as analysis mask
        print("Download and save SAT apodized mask ...")
        sat_apo_file = meta._get_analysis_mask_name()
        urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
        url = f"{urlpref}apodized_mask_bbpipe_paper.fits"
        with urllib.request.urlopen(url, timeout=timeout_seconds):
            urllib.request.urlretrieve(url, filename=sat_apo_file)
            sat_apo_mask = hp.read_map(sat_apo_file, field=0)
            sat_apo_mask = hp.ud_grade(sat_apo_mask, meta.nside)
            meta.save_mask("analysis", sat_apo_mask, overwrite=True)
    else:
        # Assemble custom analysis mask from hits map, Galactic mask and
        # point source mask

        # Download galactic mask
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
                    plt.savefig(fname.replace("fits", "png"))

        # Generate mock point source mask
        if "point_source" in meta.masks["include_in_mask"]:
            meta.timer.start("ps_mask")
            nsrcs = meta.mock_nsrcs
            mask_radius_arcmin = meta.mock_srcs_hole_radius
            ps_mask = random_src_mask(binary_mask, nsrcs,
                                      mask_radius_arcmin)
            meta.save_mask("point_source", ps_mask, overwrite=True)
            meta.timer.stop("ps_mask",
                            "Generate mock point source mask", args.verbose)

            if args.plots:
                plt.figure(figsize=(16, 9))
                hp.mollview(ps_mask, cmap=cmap, cbar=False)
                hp.graticule()
                plt.savefig(meta.point_source_mask_name.replace(".fits",
                                                                ".png"))

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
            apod_type="C1"
        )

        meta.save_mask("analysis", final_mask, overwrite=True)
        meta.timer.stop("final_mask", "Compute and save final analysis mask",
                        args.verbose)

    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(meta.read_mask("analysis"), cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(meta.analysis_mask_name.replace(".fits", ".png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self_assemble", action="store_true")
    args = parser.parse_args()

    mask_handler(args)
