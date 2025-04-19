import numpy as np
import healpy as hp
from pixell import enmap, enplot, curvedsky
import matplotlib.pyplot as plt
from pixell import uharm
import pymaster as nmt


def _check_pix_type(pix_type):
    """
    Error handling for pixellization types.

    Parameters
    ----------
    pix_type : str
        Pixellization type.
    """
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")


def add_map(imap, omap, pix_type):
    """
    Add a single map imap to an existing omap. omap is modified in place.
    """
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        omap += imap
    elif pix_type == 'car':
        enmap.extract(imap, omap.shape, omap.wcs, omap=omap,
                      op=np.ndarray.__iadd__)


def multiply_map(imap, omap, pix_type):
    """
    Multiply a single map imap with an existing omap.
    omap is modified in-place.
    """
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        omap += imap
    elif pix_type == 'car':
        enmap.extract(imap, omap.shape, omap.wcs, omap=omap,
                      op=np.ndarray.__imul__)


def ud_grade(map_in, nside_out, power=None, pix_type='hp'):
    """
    Utility function to upgrade or downgrade a map.
    Only support the healpix pixellization type.

    Parameters
    ----------
    map_in : np.ndarray
        Input map.
    nside_out : int
        Output nside.
    power : float, optional
        Set to -2 to keep the sum invariant (for hits)
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    map_out : np.ndarray
        Output map.
    """
    if pix_type != 'hp':
        raise ValueError("Can't U/D-grade non-HEALPix maps")
    return hp.ud_grade(map_in, nside_out=nside_out, power=power)


def map2alm(map, pix_type="hp"):
    """
    """
    _check_pix_type(pix_type)

    if isinstance(map, str):
        if pix_type == "car":
            _, wcs = enmap.read_map_geometry(map)
            res = np.deg2rad(np.min(np.abs(wcs.wcs.cdelt)))
            lmax = uharm.res2lmax(res)
            return lmax
        else:
            map = read_map(map)

    if pix_type == "hp":
        return hp.map2alm(map)
    else:
        return curvedsky.map2alm(map)


def alm2map(alm, pix_type="hp", nside=None, car_map_template=None):
    """
    """
    _check_pix_type(pix_type)

    if pix_type == "hp":
        assert nside is not None, "nside is required"
        return hp.alm2map(alm, nside=nside)
    else:
        if isinstance(car_map_template, str):
            shape, wcs = enmap.read_map_geometry(car_map_template)
        else:
            shape, wcs = car_map_template.geometry
        map = enmap.zeros((3,) + shape, wcs)
        curvedsky.alm2map(alm, map)
        return map


def lmax_from_map(map, pix_type="hp"):
    """
    Determine the maximum multipole from a map and its
    pixellization type.

    Parameters
    ----------
    map : str or np.ndarray or enmap.ndmap
        Input filename or map.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    int
        Maximum multipole.
    """
    _check_pix_type(pix_type)

    if isinstance(map, str):
        if pix_type == "car":
            _, wcs = enmap.read_map_geometry(map)
            res = np.deg2rad(np.min(np.abs(wcs.wcs.cdelt)))
            lmax = uharm.res2lmax(res)
            return lmax
        else:
            map = read_map(map)
    if pix_type == "hp":
        nside = hp.npix2nside(map.shape[-1])
        return 3 * nside - 1
    else:
        _, wcs = map.geometry
        res = np.deg2rad(np.min(np.abs(wcs.wcs.cdelt)))
        lmax = uharm.res2lmax(res)
        return lmax


def _get_pix_type(map_file):
    """
    Determine the pixellization type from a map file.
    Since `read_fits_header` does not handle compression,
    assign the HEALPIX type to compressed files.

    Parameters
    ----------
    map_file : str
        Map file name.

    Returns
    -------
    str
        Pixellization type.
    """
    if "fits.gz" in map_file:
        return "hp"

    header = enmap.read_fits_header(map_file)
    if "WCSAXES" in header:
        return "car"
    else:
        return "hp"


def read_map(map_file,
             pix_type='hp',
             fields_hp=None,
             convert_K_to_muK=False,
             geometry=None,
             car_template=None):
    """
    Read a map from a file, regardless of the pixellization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    pix_type : str, optional
        Pixellization type.
    fields_hp : tuple, optional
        Fields to read from a HEALPix map.
    convert_K_to_muK : bool, optional
        Convert K to muK.
    geometry : enmap.geometry, optional
        Enmap geometry.
    car_template: str
        Path to CAR geometry template.

    Returns
    -------
    map_out : np.ndarray
        Loaded map.
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        kwargs = {"field": fields_hp} if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    else:
        if geometry is None:
            geometry = enmap.read_map_geometry(car_template)
        m = enmap.read_map(map_file, geometry=geometry)

    return conv*m


def write_map(map_file, map, dtype=np.float64, pix_type='hp',
              convert_muK_to_K=False):
    """
    Write a map to a file, regardless of the pixellization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    map : np.ndarray
        Map to write.
    dtype : np.dtype, optional
        Data type.
    pix_type : str, optional
        Pixellization type.
    convert_muK_to_K : bool, optional
        Convert muK to K.
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
    Apply a Gaussian smoothing to a map with
    a given FWHM in degrees.

    Parameters
    ----------
    map : np.ndarray
        Input map.
    fwhm_deg : float
        FWHM in degrees.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    map_out : np.ndarray
        Smoothed map.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        return hp.smoothing(map, fwhm=np.deg2rad(fwhm_deg))
    else:
        sigma_deg = fwhm_deg / np.sqrt(8 * np.log(2))
        return enmap.smooth_gauss(map, np.deg2rad(sigma_deg))


def _plot_map_hp(map, lims=None, file_name=None, title=None):
    """
    Hidden function to plot HEALPIX maps and either show it
    or save it to a file.


    Parameters
    ----------
    map : np.ndarray
        Input map.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    file_name : str, optional
        Output file name.
    title : str, optional
        Plot title.
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
    if ncomp != 1 and lims is not None:
        range_args = [
            {
                "min": lims[i][0],
                "max": lims[i][1]
            } for i in lims(3)
        ]
    for i in range(ncomp):
        if ncomp != 1:
            f = "TQU"[i]
        print("np.shape(map)", np.shape(np.atleast_2d(map)))
        hp.mollview(
            np.atleast_2d(map)[i],
            cmap=cmap,
            title=title,
            **range_args[i],
            cbar=True
        )
        if file_name:
            if ncomp == 1:
                plt.savefig(f"{file_name}.png", bbox_inches="tight")
            else:
                plt.savefig(f"{file_name}_{f}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def _plot_map_car(map, lims=None, file_name=None):
    """
    Hidden function to plot CAR maps and either show it
    or save it to a file.

    Parameters
    ----------
    map : np.ndarray
        Input map.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    file_name : str, optional
        Output file name.
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


def plot_map(map, file_name=None, lims=None, title=None, pix_type="hp"):
    """
    Plot a map regardless of the pixellization type.

    Parameters
    ----------
    map : np.ndarray
        Input map.
    file_name : str, optional
        Output file name.
    lims : list, optional
        Color scale limits.
        If map is a single component, lims is a list [min, max].
        If map is a 3-component map, lims is a list of 2-element lists.
    title : str, optional
        Plot title.
    pix_type : str, optional
        Pixellization type.
    """
    _check_pix_type(pix_type)

    if pix_type == "hp":
        _plot_map_hp(map, lims, file_name=file_name, title=title)
    else:
        _plot_map_car(map, lims, file_name=file_name)


def apodize_mask(mask, apod_radius_deg, apod_type, pix_type="hp"):
    """
    Apodize a mask with a given apod radius and type regardless
    of the pixellization type.
    CAR apodization code inspired from pspy.

    Parameters
    ----------
    mask : np.ndarray or enmap.ndmap
        Input mask.
    apod_radius_deg : float
        Apodization radius in degrees.
    apod_type : str
        Apodization type
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    mask_apo : np.ndarray or enmap.ndmap
        Apodized mask.
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


def template_from_map(map, ncomp, pix_type="hp"):
    """
    Generate a template from a map regardless of the pixellization type.

    Parameters
    ----------
    map : np.ndarray or enmap.ndmap
        Input map.
    ncomp : int
        Number of components of the output template.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    template : np.ndarray or enmap.ndmap
        Template.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        if map.shape > 1:
            new_shape = (ncomp,) + map.shape[1:]
        else:
            new_shape = (ncomp, len(map))
        return np.zeros(new_shape)

    else:
        shape, wcs = map.geometry
        new_shape = (ncomp,) + shape[-2:]

        return enmap.zeros(new_shape, wcs)


def binary_mask_from_map(map, pix_type="hp", geometry=None):
    """
    Generate a binary mask from a map.
    Parameters
    ----------
    map : np.ndarray or enmap.ndmap
        Input map.
    ncomp : int
        Number of components of the output template.
    pix_type : str, optional
        Pixellization type.

    Returns
    -------
    binary_mask : np.ndarray or enmap.ndmap
        Binary mask.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        if isinstance(map, str):
            map = read_map(map, pix_type=pix_type)
        if map.shape > 1:
            map = np.sum(map, axis=0)
    else:
        if isinstance(map, str):
            shape, wcs = enmap.read_map_geometry(map)
            map = read_map(map, pix_type=pix_type)
        else:
            shape, wcs = map.geometry
        if len(shape) == 3:
            map = np.sum(map, axis=0)
            shape = shape[-2:]
        map = enmap.ndmap(map, wcs)
        binary = enmap.zeros(map.shape, wcs=wcs)

    map = smooth_map(map, fwhm_deg=1, pix_type=pix_type)
    hits_proxy = map
    hits_proxy = np.abs(map)
    hits_proxy /= np.amax(hits_proxy, axis=(0, 1))
    binary[hits_proxy > 0.1] = 1.

    return binary
