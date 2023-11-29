import argparse
import healpy as hp
from soopercool.utils import random_src_mask
from soopercool import BBmeta
import pymaster as nmt
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import urllib.request

cmap = cm.YlOrRd
cmap.set_under("w")


def mask_handler(args):
    """
    """
    meta = BBmeta(args.globals)
    mask_dir = meta.mask_directory

    timeout_seconds = 300  # Set the timeout [sec] for the socket

    os.makedirs(mask_dir, exist_ok=True)

    # Download SAT hits map
    print("Download and save SAT hits map ...")
    sat_nhits_file = meta.hitmap_file
    if not os.path.exists(sat_nhits_file):
        urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
        url = f"{urlpref}norm_nHits_SA_35FOV_ns512.fits"
        # Open the URL with a timeout
        with urllib.request.urlopen(url, timeout=timeout_seconds):
            # Retrieve the file and save it locally
            urllib.request.urlretrieve(url, filename=sat_nhits_file)

    # Download SAT apodized mask used in the SO BB
    # pipeline paper (https://arxiv.org/abs/2302.04276)
    print("Download and save SAT apodized mask ...")
    sat_apo_file = meta._get_analysis_mask_name()
    if not os.path.exists(sat_apo_file):
        urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
        url = f"{urlpref}apodized_mask_bbpipe_paper.fits"
        with urllib.request.urlopen(url, timeout=timeout_seconds):
            urllib.request.urlretrieve(url, filename=sat_apo_file)

    # Download galactic mask
    if "galactic" in meta.masks["include_in_mask"]:
        print("Download and save planck galactic masks ...")
        mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
        if not os.path.exists(mask_p15_file):
            urlpref = "http://pla.esac.esa.int/"
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

    # Get binary mask
    binary_mask_file = meta._get_binary_mask_name()
    if not os.path.exists(binary_mask_file):
        sat_nhits = hp.read_map(sat_nhits_file)
        binary_mask = (sat_nhits > 0).astype(float)
        binary_mask_ud = hp.ud_grade(binary_mask, meta.nside)
        hp.write_map(binary_mask_file, binary_mask_ud, dtype=np.int32)
    binary_mask = hp.read_map(binary_mask_file)

    if "point_source" in meta.masks["include_in_mask"]:
        # Generate a mock source mask
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
            plt.savefig(meta.point_source_mask_name.replace(".fits", ".png"))

    # Add the masks
    meta.timer.start("final_mask")
    final_mask = binary_mask.copy()
    if "galactic" in meta.masks["include_in_mask"]:
        mask = meta.read_mask("galactic")
        final_mask *= mask
    if "point_source" in meta.masks["include_in_mask"]:
        mask = meta.read_mask("point_source")
        final_mask *= mask

    final_mask = nmt.mask_apodization(final_mask, meta.masks["apod_radius"],
                                      apotype=meta.masks["apod_type"])
    meta.save_mask("analysis", final_mask, overwrite=True)
    meta.timer.stop("final_mask", "Compute and save final analysis mask",
                    args.verbose)

    if args.plots:
        plt.figure(figsize=(16, 9))
        hp.mollview(final_mask, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(meta.analysis_mask_name.replace(".fits", ".png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mask_handler(args)
