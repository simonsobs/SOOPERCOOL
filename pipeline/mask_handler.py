import argparse
import healpy as hp
from bbmaster.utils import *
import pymaster as nmt
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import urllib.request

import yaml

cmap = cm.YlOrRd
cmap.set_under("w")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    o = parser.parse_args()

    #man = PipelineManager(o.globals)
    with open(o.globals, "r") as f:
        global_dict = yaml.safe_load(f)

    mask_dir = global_dict["masks"]["mask_directory"]

    os.makedirs(mask_dir, exist_ok=True)

    # Save binary mask
    print("Load and save binary mask ...")
    binary_mask = hp.read_map("../data/mask_binary.fits.gz", hdu=1)
    binary_mask = hp.ud_grade(binary_mask, global_dict["nside"])
    binary_mask = np.where(binary_mask > 0.5, 1, 0)
    hp.write_map(
        f"{mask_dir}/{global_dict['masks']['binary_mask']}",
        binary_mask,
        overwrite=True,
        dtype=np.int32
    )

    plt.figure(figsize=(16,9))
    hp.mollview(binary_mask, cmap=cmap, cbar=False)
    hp.graticule()
    plt.savefig(f"{mask_dir}/{global_dict['masks']['binary_mask'].replace('.fits', '.png')}")

    # Download galactic mask
    print("Download and save planck galactic masks")
    mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
    if not os.path.exists(mask_p15_file):
        urllib.request.urlretrieve(
            "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
            filename=mask_p15_file
        )

    # Save different galactic masks
    gal_keys = ["GAL020", "GAL040", "GAL060", "GAL070", "GAL080", "GAL090", "GAL097", "GAL099"]
    for id_key, gal_key in enumerate(gal_keys):
        print(f"Save galactic mask {gal_key}")
        gal_mask_p15 = hp.read_map(mask_p15_file, field=id_key)
        # Rotate in equatorial coordinates
        r = hp.Rotator(coord=['G','C'])
        gal_mask_p15 = r.rotate_map_pixel(gal_mask_p15)

        gal_mask_p15 = hp.ud_grade(gal_mask_p15, global_dict["nside"])
        gal_mask_p15 = np.where(gal_mask_p15 > 0.5, 1, 0)
        fname = f"{mask_dir}/{global_dict['masks']['galactic_mask_root']}_{gal_key.lower()}.fits"
        hp.write_map(
            fname, 
            gal_mask_p15, 
            overwrite=True,
            dtype=np.int32
        )

        plt.figure(figsize=(16,9))
        hp.mollview(gal_mask_p15, cmap=cmap, cbar=False)
        hp.graticule()
        plt.savefig(fname.replace("fits", "png"))
    
    # Generate a mock source mask
    print("Generate mock point source mask ...")
    nsrcs = global_dict["sim_pars"]["mock_nsrcs"]
    mask_radius_arcmin = global_dict["sim_pars"]["mock_srcs_hole_radius"]
    ps_mask = random_src_mask(binary_mask, nsrcs, mask_radius_arcmin)
    fname = f"{mask_dir}/{global_dict['masks']['point_source_mask']}"
    hp.write_map(fname, ps_mask, overwrite=True, dtype=np.int32)

    plt.figure(figsize=(16,9))
    hp.mollview(ps_mask, cmap=cmap, cbar=False)
    hp.graticule()
    plt.savefig(fname.replace(".fits", ".png"))

    # Add the masks
    print("Create the final mask ...")
    final_mask = binary_mask.copy()
    if "galactic" in global_dict["masks"]["include_in_mask"]:
        fname = f"{mask_dir}/{global_dict['masks']['galactic_mask_root']}_{global_dict['masks']['gal_mask_mode']}.fits"
        mask = hp.read_map(fname)
        final_mask *= mask
    if "point_source" in global_dict["masks"]["include_in_mask"]:
        fname = f"{mask_dir}/{global_dict['masks']['point_source_mask']}"
        mask = hp.read_map(fname)
        final_mask *= mask
    
    final_mask = nmt.mask_apodization(final_mask, global_dict["masks"]["apod_radius"], apotype=global_dict["masks"]["apod_type"])
    fname = f"{mask_dir}/{global_dict['masks']['analysis_mask']}"
    hp.write_map(fname, final_mask, overwrite=True, dtype=np.float32)

    plt.figure(figsize=(16,9))
    hp.mollview(final_mask, cmap=cmap, cbar=False)
    hp.graticule()
    plt.savefig(fname.replace(".fits", ".png"))
    