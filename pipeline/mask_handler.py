import argparse
from soopercool import BBmeta
import os
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import urllib.request

# cmap = cm.YlOrRd
# cmap.set_under("w")


def mask_handler(args):
    """
    """
    meta = BBmeta(args.globals)
    mask_dir = meta.mask_directory

    os.makedirs(mask_dir, exist_ok=True)

    # # Download galactic mask
    # print("Download and save planck galactic masks ...")
    # mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
    # if not os.path.exists(mask_p15_file):
    #     urllib.request.urlretrieve(
    #         "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits",  # noqa
    #         filename=mask_p15_file
    #     )

#     # Save different galactic masks
#     meta.timer.start("proj_gal")
#     gal_keys = ["GAL020", "GAL040", "GAL060", "GAL070",
#                 "GAL080", "GAL090", "GAL097", "GAL099"]
#     for id_key, gal_key in enumerate(gal_keys):
#         meta.timer.start(f"gal{id_key}")
#         gal_mask_p15 = hp.read_map(mask_p15_file, field=id_key)
#         # Rotate in equatorial coordinates
#         r = hp.Rotator(coord=['G', 'C'])
#         gal_mask_p15 = r.rotate_map_pixel(gal_mask_p15)

#     # Download galactic mask
#     print("Download and save planck galactic masks ...")
#     mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
#     if not os.path.exists(mask_p15_file):
#         urllib.request.urlretrieve(
#             "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits",  # noqa
#             filename=mask_p15_file
#         )

#     # Save different galactic masks
#     meta.timer.start("proj_gal")
#     gal_keys = ["GAL020", "GAL040", "GAL060", "GAL070",
#                 "GAL080", "GAL090", "GAL097", "GAL099"]
#     for id_key, gal_key in enumerate(gal_keys):
#         meta.timer.start(f"gal{id_key}")
#         gal_mask_p15 = hp.read_map(mask_p15_file, field=id_key)
#         # Rotate in equatorial coordinates
#         r = hp.Rotator(coord=['G', 'C'])
#         gal_mask_p15 = r.rotate_map_pixel(gal_mask_p15)

#         gal_mask_p15 = hp.ud_grade(gal_mask_p15, meta.nside)
#         gal_mask_p15 = np.where(gal_mask_p15 > 0.5, 1, 0)
#         fname = f"{mask_dir}/{meta.masks['galactic_mask_root']}_{gal_key.lower()}.fits"  # noqa
#         hp.write_map(
#             fname,
#             gal_mask_p15,
#             overwrite=True,
#             dtype=np.int32
#         )
#         meta.timer.stop(f"gal{id_key}",
#                         f"Galactic mask {gal_key} projection",
#                         args.verbose)

#         if args.plots:
#             plt.figure(figsize=(16, 9))
#             hp.mollview(gal_mask_p15, cmap=cmap, cbar=False)
#             hp.graticule()
#             plt.savefig(fname.replace("fits", "png"))
#     meta.timer.stop("proj_gal", "Projecting Planck galactic masks",
#                     args.verbose)

#     binary_mask = meta.read_mask("binary")

#     # Generate a mock source mask
#     meta.timer.start("ps_mask")
#     nsrcs = meta.mock_nsrcs
#     mask_radius_arcmin = meta.mock_srcs_hole_radius
#     ps_mask = random_src_mask(binary_mask, nsrcs,
#                               mask_radius_arcmin)
#     meta.save_mask("point_source", ps_mask, overwrite=True)
#     meta.timer.stop("ps_mask", "Generate mock point source mask", args.verbose) # noqa

#     if args.plots:
#         plt.figure(figsize=(16, 9))
#         hp.mollview(ps_mask, cmap=cmap, cbar=False)
#         hp.graticule()
#         plt.savefig(meta.point_source_mask_name.replace(".fits", ".png"))

#     # Add the masks
#     meta.timer.start("final_mask")
#     final_mask = binary_mask.copy()
#     if "galactic" in meta.masks["include_in_mask"]:
#         mask = meta.read_mask("galactic")
#         final_mask *= mask
#     if "point_source" in meta.masks["include_in_mask"]:
#         mask = meta.read_mask("point_source")
#         final_mask *= mask

#     final_mask = nmt.mask_apodization(final_mask, meta.masks["apod_radius"],
#                                       apotype=meta.masks["apod_type"])
#     meta.save_mask("analysis", final_mask, overwrite=True)
#     meta.timer.stop("final_mask", "Compute and save final analysis mask",
#                     args.verbose)

#     if args.plots:
#         plt.figure(figsize=(16, 9))
#         hp.mollview(final_mask, cmap=cmap, cbar=False)
#         hp.graticule()
#         plt.savefig(meta.analysis_mask_name.replace(".fits", ".png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mask_handler(args)
