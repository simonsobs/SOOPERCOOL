import argparse
from soopercool import BBmeta
from soopercool import map_utils as mu

from pixell import enmap, enplot
import numpy as np
import glob
import random


def get_submap_center_radec(imap):
    y, x = imap.shape[-2], imap.shape[-1]
    dec, ra = np.rad2deg(enmap.pix2sky(imap.shape, imap.wcs, [y//2, x//2]))
    return ra, dec


def get_submap_corners_radec(imap):
    y, x = imap.shape[-2], imap.shape[-1]
    corners = [[0, 0], [0, x], [y, 0], [y, x]]
    radec_corners = []
    for corner in corners:
        dec, ra = np.rad2deg(enmap.pix2sky(imap.shape, imap.wcs, corner))
        radec_corners.append([ra, dec])
    return np.array(radec_corners)


def build_map_list(map_dir, freq_tag, type="wmap", wafer="all"):
    """
    Build list of atomic maps from a given directory

    Parameters
    ----------
    map_dir : str
        The directory where the maps are stored
    freq_tag : str
        The frequency tag of the maps (e.g. "f090", "f150")
    type : str
        The type of the maps (e.g. "wmap", "weights")
    wafer : str
        The wafer number of the maps, "all" for all wafers
        either "0", ...
    """
    if wafer == "all":
        wafer = "[0-9]"
    regex = f"{map_dir}/**/*ws{wafer}_{freq_tag}*{type}.fits"
    out = glob.glob(regex, recursive=True)
    out.sort()
    return out


def group_list(list, n_groups, seed=1234):
    """
    """
    list = np.asarray(list)
    ids_groups = np.arange(n_groups)
    n = len(list) // n_groups
    ids_list = np.zeros(len(list), dtype=int)
    for id_group in ids_groups:
        if id_group == ids_groups[-1]:
            ids_list[id_group*n:] = id_group
        else:
            ids_list[id_group*n:(id_group+1)*n] = id_group
    random.seed(seed)
    random.shuffle(ids_list)
    return {
        id_group: list[ids_list == id_group] for id_group in ids_groups
    }, ids_list


def coadd_maps(map_list, res=10):
    """
    """
    template_geom = enmap.band_geometry((np.deg2rad(-75), np.deg2rad(25)),
                                        res=np.deg2rad(10/60))
    car = enmap.zeros((3, *template_geom[0]), template_geom[1])
    wcar = enmap.zeros((3, *template_geom[0]), template_geom[1])
    hits = enmap.zeros(*template_geom)

    shape, wcs = car.geometry
    for f in map_list:
        m = mu.read_map(f, pix_type='car')
        car = enmap.insert(car, m, op=np.ndarray.__iadd__)
        w = mu.read_map(f.replace("wmap", "weights"), pix_type='car')
        w = np.moveaxis(w.diagonal(), -1, 0)
        wcar = enmap.insert(wcar, w, op=np.ndarray.__iadd__)
        h = mu.read_map(f.replace("wmap", "hits"), pix_type='car')[0]
        hits = enmap.insert(hits, h, op=np.ndarray.__iadd__)
    wcar[wcar == 0] = np.inf
    return car / wcar, wcar, hits


def get_CAR_template(ncomp, res, dec_cut=None):
    """
    """
    if dec_cut is None:
        shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60))
    else:
        shape, wcs = enmap.band_geometry(np.deg2rad(dec_cut),
                                         res=np.deg2rad(res/60))
    shape = (ncomp, *shape)
    return enmap.zeros(shape, wcs)


def build_mask_from_boxes(template, box_edges):
    """
    """
    mask = template.copy()
    for ra0, ra1, dec0, dec1 in box_edges:
        box = np.array([[dec0, ra0], [dec1, ra1]])
        box = np.deg2rad(box)
        sub = enmap.submap(template, box=box)
        sub[:] = 1.
        mask = enmap.insert(mask, sub, op=np.ndarray.__iadd__)
    mask[mask != 0] = 1
    return mask


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    maps_dir = f"{out_dir}/bundled_maps"
    BBmeta.make_dir(maps_dir)

    plot_dir = f"{out_dir}/plots/bundled_maps"
    BBmeta.make_dir(plot_dir)

    freq_tags = args.ftags.split(",")
    n_bundles = args.n_bundles

    fname_list = {}
    for ftag in freq_tags:
        fname_list[ftag] = build_map_list(
            args.map_dir, ftag, type="wmap",
            wafer="all")
        print(f"Found maps for {ftag}: {len(fname_list[ftag])}")

    # Per-wafer
    for id_waf in range(7):
        for ftag in freq_tags:
            fname_list[ftag, id_waf] = build_map_list(args.map_dir, ftag,
                                                      type="wmap",
                                                      wafer=str(id_waf))
            print(f"Found maps for {ftag} in wafer {id_waf}: \
                  {len(fname_list[ftag, id_waf])}")

    template_CAR = get_CAR_template(ncomp=1, res=10, dec_cut=[-75, 25])

    # Build a mask
    box_edges = [
        [-108, -180, -30, 9],
        [180, 144, -30, 9]
    ]
    restricted_region = build_mask_from_boxes(template_CAR, box_edges)

    # Filter map_lists
    keep = {(ftag, id_waf): [] for ftag in freq_tags for id_waf in range(7)}
    for ftag in freq_tags:
        keep[ftag] = []
        for id_waf in range(7):
            for fname in fname_list[ftag, id_waf]:
                m = mu.read_map(fname, pix_type='car')

                binary = enmap.insert(template_CAR.copy(), m[0])
                binary[binary != 0] = 1
                overlap = np.sum(binary * restricted_region) / np.sum(binary)

                if overlap >= 0.5:
                    # print(f"Found map !")
                    keep[ftag, id_waf].append(fname)
                    keep[ftag].append(fname)

    # Check that the number of maps is the same
    for ftag in freq_tags:
        for ftag2 in freq_tags:
            if ftag == ftag2:
                continue
            keep[ftag] = [
                f for f in keep[ftag]
                if f.replace(ftag, ftag2) in keep[ftag2]
            ]

        for id_waf in range(7):
            keep[ftag, id_waf] = [
                f for f in keep[ftag, id_waf]
                if f in keep[ftag]]

    for ftag in freq_tags:
        print(F"Found {len(keep[ftag])} maps for {ftag} (after RA/DEC cuts)")
        for id_waf in range(7):
            print(f"Found {len(keep[ftag, id_waf])} maps for {ftag} \
                   in wafer {id_waf} (after RA/DEC cuts)")

    # Group the maps
    to_coadd, ids = {}, {}
    for ftag in freq_tags:
        for id_waf in range(7):
            sublist, ids_list = group_list(keep[ftag, id_waf], n_bundles)
            to_coadd[ftag, id_waf] = sublist
            ids[ftag, id_waf] = ids_list

    # Full coadd list
    to_coadd_full = {}
    for ftag in freq_tags:
        for id_bundle in range(n_bundles):
            to_coadd_full[ftag, id_bundle] = np.concatenate(
                [to_coadd[ftag, id_waf][id_bundle]
                 for id_waf in range(7)])

    coadded_maps = {}
    coadded_weights = {}
    coadded_hits = {}
    unit_conv = 1e6 * 13.2  # pW to uK
    for ftag in freq_tags:
        for id_bundle in range(n_bundles):
            meta.timer.start("coadding")
            for id_waf in range(7):
                signal, weight, hits = coadd_maps(to_coadd[ftag, id_waf][id_bundle])  # noqa
                coadded_maps[ftag, id_waf, id_bundle] = signal * unit_conv
                coadded_weights[ftag, id_waf, id_bundle] = weight
                coadded_hits[ftag, id_waf, id_bundle] = hits

            signal, weight, hits = coadd_maps(to_coadd_full[ftag, id_bundle])
            coadded_maps[ftag, id_bundle] = signal * unit_conv
            coadded_weights[ftag, id_bundle] = weight
            coadded_hits[ftag, id_bundle] = hits

            meta.timer.stop("coadding", f"Coadded {ftag} bundle {id_bundle}")

    for ftag in freq_tags:
        for id_bundle in range(n_bundles):
            plots = enplot.get_plots(
                coadded_maps[ftag, id_bundle],
                range=[100000, 100, 100], colorbar=1, ticks=5
            )
            for field, plot in zip("TQU", plots):
                enplot.write(
                    f"{plot_dir}/coadd_{ftag}_bundle{id_bundle}_allwaf_{field}.png",  # noqa
                    plot)
            # Plot hits
            plot_hit = enplot.get_plots(coadded_hits[ftag, id_bundle],
                                        range=[100000], colorbar=1, ticks=5)[0]
            enplot.write(
                f"{plot_dir}/coadd_{ftag}_bundle{id_bundle}_allwaf_hits.png",
                plot_hit
            )
            file_name = f"{maps_dir}/TQU_CAR_coadd_{ftag}_bundle{id_bundle}_allwaf.fits"  # noqa
            mu.write_map(file_name, coadded_maps[ftag, id_bundle],
                         pix_type='car')
            mu.write_map(file_name.replace("TQU", "weights"),
                         coadded_weights[ftag, id_bundle], pix_type='car')
            mu.write_map(file_name.replace("TQU", "hits"),
                         coadded_hits[ftag, id_bundle], pix_type='car')

            with open(f"{maps_dir}/atomic_map_list_{ftag}_bundle{id_bundle}.txt", "w") as f:  # noqa
                for fname in to_coadd_full[ftag, id_bundle]:
                    f.write(f"{fname.replace(args.map_dir+'/', '')}\n")

            for id_waf in range(7):
                plots = enplot.get_plots(
                    coadded_maps[ftag, id_waf, id_bundle],
                    range=[100000, 100, 100], colorbar=1, ticks=5
                )
                for field, plot in zip("TQU", plots):
                    enplot.write(f"{plot_dir}/coadd_{ftag}_bundle{id_bundle}_waf{id_waf}_{field}.png", plot)  # noqa
                # Plot hits
                plot_hit = enplot.get_plots(
                    coadded_hits[ftag, id_waf, id_bundle],
                    range=[100000], colorbar=1, ticks=5)[0]
                enplot.write(f"{plot_dir}/coadd_{ftag}_bundle{id_bundle}_waf{id_waf}_hits.png", plot_hit)  # noqa
                file_name = f"{maps_dir}/TQU_CAR_coadd_{ftag}_bundle{id_bundle}_waf{id_waf}.fits"  # noqa
                mu.write_map(file_name, coadded_maps[ftag, id_bundle],
                             pix_type='car')
                mu.write_map(
                    file_name.replace("TQU", "weights"),
                    coadded_weights[ftag, id_waf, id_bundle],
                    pix_type='car')
                mu.write_map(file_name.replace("TQU", "hits"),
                             coadded_hits[ftag, id_waf, id_bundle],
                             pix_type='car')

                with open(f"{maps_dir}/atomic_map_list_{ftag}_waf{id_waf}_bundle{id_bundle}.txt", "w") as f:  # noqa
                    for fname in to_coadd[ftag, id_waf][id_bundle]:
                        f.write(f"{fname.replace(args.map_dir+'/', '')}\n")

    # HEALPIX reprojection
    from pixell import reproject
    import healpy as hp
    for ftag in freq_tags:
        for id_bundle in range(n_bundles):
            TQU_hp = reproject.map2healpix(coadded_maps[ftag, id_bundle],
                                           nside=256)
            hits_hp = reproject.map2healpix(
                coadded_hits[ftag, id_bundle], nside=256, extensive=True,
                method="spline"
            )
            mu.write_map(f"{maps_dir}/coadd_{ftag}_bundle{id_bundle}_map.fits",
                         TQU_hp, dtype=np.float32, pix_type='hp')
            mu.write_map(
                f"{maps_dir}/coadd_{ftag}_bundle{id_bundle}_hits.fits",
                hits_hp, dtype=np.float32, pix_type='hp')

            for id_waf in range(7):
                TQU_hp = reproject.map2healpix(
                    coadded_maps[ftag, id_waf, id_bundle], nside=256
                )
                hits_hp = reproject.map2healpix(
                    coadded_hits[ftag, id_waf, id_bundle], nside=256,
                    extensive=True, method="spline"
                )
                mu.write_map(f"{maps_dir}/coadd_{ftag}_wafer{id_waf}_bundle{id_bundle}_map.fits",  # noqa
                             TQU_hp, dtype=np.float32, pix_type='car')
                hp.write_map(f"{maps_dir}/coadd_{ftag}_wafer{id_waf}_bundle{id_bundle}_hits.fits",  # noqa
                             hits_hp, dtype=np.float32, pix_type='car')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--n_bundles", type=int, help="Number of bundles.")
    parser.add_argument("--ftags",
                        help="Frequency tags separated by commas ','")
    parser.add_argument("--map_dir", help="Path to the atomic root dir.")
    parser.add_argument("--seed", type=int,
                        help="Seed for the random number generator.")
    args = parser.parse_args()
    main(args)
