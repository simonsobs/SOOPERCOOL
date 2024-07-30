import numpy as np
import healpy as hp
from pixell import enmap, enplot
import matplotlib.pyplot as plt
import pymaster as nmt


def _check_pix_type(pix_type):
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")


def ud_grade(map_in, nside_out, power=None, pix_type='hp'):
    if pix_type != 'hp':
        raise ValueError("Can't U/D-grade non-HEALPix maps")
    return hp.ud_grade(map_in, nside_out=nside_out, power=power)


def read_map(map_file,
             pix_type='hp',
             fields_hp=None,
             convert_K_to_muK=False,
             geometry=None):
    """
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        kwargs = {"field": fields_hp} if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        m = enmap.read_map(map_file, geometry=geometry)

    return conv*m


def write_map(map_file, map, dtype=None, pix_type='hp',
              convert_muK_to_K=False):
    """
    """
    if convert_muK_to_K:
        map *= 1.e-6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        hp.write_map(map_file, map, overwrite=True, dtype=dtype)
    else:
        enmap.write_map(map_file, map)


def smooth_map(map, fwhm_deg, pix_type="hp"):
    """
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        return hp.smoothing(map, fwhm=np.deg2rad(fwhm_deg))
    else:
        sigma_deg = fwhm_deg / np.sqrt(8 * np.log(2))
        return enmap.smooth_gauss(map, np.deg2rad(sigma_deg))


def _plot_map_hp(map, lims=None, file_name=None):
    """
    """
    ncomp = map.shape[0] if len(map.shape) == 2 else 1
    cmap = "YlOrRd" if ncomp == 1 else "RdYlBu_r"
    if lims is None:
        range_args = [{} for i in range(ncomp)]

    if ncomp == 1 and lims is not None:
        range_args = [{
            "min": lims[0],
            "max": lims[1]
        }]
    if ncomp == 3 and lims is not None:
        range_args = [
            {
                "min": lims[i][0],
                "max": lims[i][1]
            } for i in lims(3)
        ]
    for i in range(ncomp):
        if ncomp != 1:
            f = "TQU"[i]
        hp.mollview(map[i], cmap=cmap, **range_args[i], cbar=True)
        if file_name:
            if ncomp == 1:
                plt.savefig(f"{file_name}.png", bbox_inches="tight")
            else:
                plt.savefig(f"{file_name}_{f}.png", bbox_inches="tight")
        else:
            plt.show()


def _plot_map_car(map, lims=None, file_name=None):
    """
    """
    ncomp = map.shape[0] if len(map.shape) == 3 else 1

    if lims is None:
        range_args = {}

    if ncomp == 1 and lims is not None:
        range_args = {
            "min": lims[0],
            "max": lims[1]
        }
    if ncomp == 3 and lims is not None:
        range_args = {
            "min": [lims[i][0] for i in range(ncomp)],
            "max": [lims[i][1] for i in range(ncomp)]
        }

    plot = enplot.plot(
         map,
         colorbar=True,
         ticks=10,
         **range_args
    )
    for i in range(ncomp):
        suffix = ""
        if ncomp != 1:
            suffix = f"_{'TQU'[i]}"

        if file_name:
            enplot.write(
                f"{file_name}{suffix}.png",
                plot[i]
            )
        else:
            enplot.show(plot[i])


def plot_map(map, file_name=None, lims=None,
             pix_type="hp"):
    """
    """
    _check_pix_type(pix_type)

    if pix_type == "hp":
        _plot_map_hp(map, lims, file_name)
    else:
        _plot_map_car(map, lims, file_name)


def apodize_mask(mask, apod_radius_deg, apod_type, pix_type="hp"):
    """
    CAR apodization code inspired from pspy.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        mask_apo = nmt.mask_apodization(
            mask,
            apod_radius_deg,
            apod_type
        )
    else:
        distance = enmap.distance_transform(mask)
        distance = np.rad2deg(distance)

        mask_apo = mask.copy()
        idx = np.where(distance > apod_radius_deg)

        if apod_type == "C1":
            mask_apo = 0.5 - 0.5 * np.cos(-np.pi * distance / apod_radius_deg)
        elif apod_type == "C2":
            mask_apo = (
                distance / apod_radius_deg -
                np.sin(2 * np.pi * distance / apod_radius_deg) / (2 * np.pi)
            )
        else:
            raise ValueError(f"Unknown apodization type {apod_type}")
        mask_apo[idx] = 1

    return mask_apo
