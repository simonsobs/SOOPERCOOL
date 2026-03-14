import numpy as np
import healpy as hp
from pixell import enmap, enplot, curvedsky, uharm
import matplotlib.pyplot as plt
import pymaster as nmt


def _check_pix_type(pix_type):
    """
    Error handling for pixelization types.

    Parameters
    ----------
    pix_type : str
        Pixelization type.
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
    Only support the healpix pixelization type.

    Parameters
    ----------
    map_in : np.ndarray
        Input map.
    nside_out : int
        Output nside.
    power : float, optional
        Set to -2 to keep the sum invariant (for hits)
    pix_type : str, optional
        Pixelization type.

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
    Transform map of either HEALPix or CAR pixelization into alms.

    Parameters
    ----------
    map: numpy.array or enmap.ndmap, shape (ndim, map_shape)
        Input map. ndim is either 1 or 3 (for polarized input). map_shape is
        either npix (for hp) or (nx, ny) (for car).
    pix_type: str
        Pixelization scheme. Must be "hp" or "car".

    Returns
    -------
    numpy.array, shape (ndim, nalm)
        Spherical harmonic coefficients.
    """
    _check_pix_type(pix_type)

    if isinstance(map, str):
        map = read_map(map, pix_type=pix_type)
    lmax = lmax_from_map(map, pix_type=pix_type)

    if pix_type == "hp":
        return hp.map2alm(map, lmax=lmax)
    else:
        return curvedsky.map2alm(map, lmax=lmax)


def alm2map(alm, pix_type="hp", nside=None, car_template=None,
            geometry=None):
    """
    Transform alms into maps of either HEALPix or CAR pixelization.

    Parameters
    ----------
    alm: numpy.array, shape (ndim, nalm) 
        Spherical harmonic coefficients. ndim must be 1 or 3 (for polarized
        input).
    pix_type: str
        Pixelization scheme. Must be "hp" or "car".
    nside: int
        HEALPix NSIDE parameter.
    car_map_template: enamp.ndmap, shape (ndim, nx, ny)
        CAR map template. ndim can be 1 or 3.
    Returns
    -------
    numpy.array or enmap.ndmap, shape (3, map_shape)
        Output map. map_shape is either npix (for hp) or (nx, ny) (for car).
    """
    _check_pix_type(pix_type)
    if isinstance(alm, list):
        alm = np.array(alm, dtype=np.complex128)

    if pix_type == "hp":
        assert nside is not None, "nside is required"
        return hp.alm2map(alm, nside=nside)
    else:
        if isinstance(car_template, str):
            shape, wcs = enmap.read_map_geometry(car_template)
        elif car_template is None:
            shape, wcs = geometry
        else:
            shape, wcs = car_template.geometry
        if alm.ndim == 1:
            map = enmap.zeros(shape, wcs)
        else:
            ndim = alm.shape[0]
            map = enmap.zeros((ndim,) + shape, wcs)
        return curvedsky.alm2map(alm, map)


def _lmax_from_car_geometry(geometry):
    """
    Determine the maximum multipole from a CAR map.

    Parameters
    ----------
    geometry: tuple of (shape, wcs)
        CAR map shape (nx, ny) and wcs (instance of astropy.wcs.WCS)
    pix_type : str, optional
        Pixelization type.

    Returns
    -------
    int
        Maximum multipole.
    """
    _, wcs = geometry
    res = np.deg2rad(np.min(np.abs(wcs.wcs.cdelt)))

    return uharm.res2lmax(res)


def lmax_from_map(map, pix_type="hp"):
    """
    Determine the maximum multipole from a map and its pixelization type.

    Parameters
    ----------
    map : str or np.ndarray or enmap.ndmap
        Input filename or map.
    pix_type : str, optional
        Pixelization type.

    Returns
    -------
    int
        Maximum multipole.
    """
    _check_pix_type(pix_type)

    if isinstance(map, str):
        if pix_type == "car":
            geometry = enmap.read_map_geometry(map)
            lmax = _lmax_from_car_geometry(geometry)
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
    Determine the pixelization type from a map file.
    Since `read_fits_header` does not handle compression,
    assign the HEALPIX type to compressed files.

    Parameters
    ----------
    map_file : str
        Map file name.

    Returns
    -------
    str
        Pixelization type.
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
    Read a map from a file, regardless of the pixelization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    pix_type : str, optional
        Pixelization type. Accepts "hp", "car", None. If None, infer the
        pixel type on the fly. Default: "hp".
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
    if pix_type is None:
        pix_type = _get_pix_type(map_file)
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        kwargs = {"field": fields_hp} if fields_hp is not None else {}
        m = hp.read_map(map_file, **kwargs)
    elif pix_type == "car":
        if (geometry, car_template) == (None, None):
            geometry = None
        if car_template is not None:
            geometry = enmap.read_map_geometry(car_template)
        m = enmap.read_map(map_file, geometry=geometry)

    return conv*m


def write_map(map_file, map, dtype=np.float64, pix_type='hp',
              convert_muK_to_K=False):
    """
    Write a map to a file, regardless of the pixelization type.

    Parameters
    ----------
    map_file : str
        Map file name.
    map : np.ndarray
        Map to write.
    dtype : np.dtype, optional
        Data type.
    pix_type : str, optional
        Pixelization type.
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
        pixelization type.

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


def crop_borders(map, crop_size, smooth_scale, pix_type="hp"):
    """
    Crop the borders of a map by setting to zero all pixels
    within a given distance from the border. Both distance
    and smoothing scale are given in degrees.

    Parameters
    ----------
    map : enmap.ndmap
        Input map.
    crop_size : float
        Distance from the border to crop in degrees.
    smooth_scale : float
        Smoothing scale in degrees.
    pix_type : str, optional
        Pixelization type.
    """
    out_map = map.copy()
    _check_pix_type(pix_type)
    if pix_type == "hp":
        raise NotImplementedError(
            "Border cropping not implemented for HEALPix maps."
        )
    else:
        dist = enmap.distance_transform(map)
        dist_smooth = enmap.smooth_gauss(dist, np.deg2rad(smooth_scale))
        out_map[(dist_smooth < np.deg2rad(crop_size)) & (map != 0)] = 0.

    return out_map


def apply_box_mask(map, box):
    """
    Apply box mask to a given map (hp or car).

    Parameters
    ----------
    map: np.array or enmap.ndmap, shape (ndim, shape)
        Input map.
    box: list
        Contains [[dec_lo, ra_lo], [dec_hi, ra_hi]], in degrees.

    Returns
    -------
    map: np.array or enmap.ndmap, shape (ndim, shape)
        Output map.
    """
    if hasattr(map, "wcs"):
        box = np.deg2rad(box)
        decs, ras = map.posmap()
        mask_box = enmap.zeros(shape=map.shape, wcs=map.wcs, dtype=np.float64)
    else:
        npix = map.shape[-1]
        ras, decs = hp.pix2ang(hp.npix2nside(npix), np.arange(npix),
                               lonlat=True)
        ras[ras > 180.] -= 360.
        mask_box = np.zeros(npix)
    mask_box[(box[0][0] < decs) & (decs < box[1][0])
             & (box[0][1] < ras) & (ras < box[1][1])] = 1.

    return map * mask_box


def get_fsky_from_hits(map, mode="NN"):
    """
    Infer effective sky fraction from a hits map assuming one of the three
    approximate limits (signal-signal, signal-noise, noise-noise),

    Parameters
    ----------
    map: np.array or enmap.ndmap, shape (map_shape, )
        Input hits maps.
    mode: str
        Approximation. Accept either "SS", "SN", or "NN.

    Returns
    -------
    float
        Effective sky fraction.
    """
    if hasattr(map, "wcs"):
        pixsize = enmap.pixsizemap(map.shape, map.wcs)
    else:
        npix = len(map) if map.ndim == 1 else len(map[0])
        pixsize = 4. * np.pi / npix * np.ones(npix)
    if mode == "SS":
        fsky = np.sum(pixsize*map**2)**2 / np.sum(pixsize*map**4)
    elif mode == "SN":
        fsky = (np.sum(pixsize*map)*np.sum(pixsize*map**2)
                / np.sum(pixsize*map**3))
    elif mode == "NN":
        fsky = np.sum(pixsize*map)**2 / np.sum(pixsize*map**2)
    else:
        raise ValueError(f"Mode {mode} unknown. Choose either SS, SN, or NN.")

    # Translate from square radians into sky fraction
    return fsky / 4. / np.pi


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
    Plot a map regardless of the pixelization type.

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
        Pixelization type.
    """
    _check_pix_type(pix_type)

    if pix_type == "hp":
        _plot_map_hp(map, lims, file_name=file_name, title=title)
    else:
        _plot_map_car(map, lims, file_name=file_name)


def apodize_mask(mask, apod_radius_deg, apod_type, pix_type="hp"):
    """
    Apodize a mask with a given apod radius and type regardless
    of the pixelization type.
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
        Pixelization type.

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
        if apod_type == "C1":
            mask_apo = enmap.apod_mask(
                mask, width=np.deg2rad(apod_radius_deg)
            )
        else:
            raise NotImplementedError(f"Unknown apodization type {apod_type}")  # noqa

    return mask_apo


def template_from_map(map, ncomp, pix_type="hp"):
    """
    Generate a template from a map regardless of the pixelization type.

    Parameters
    ----------
    map : np.ndarray or enmap.ndmap
        Input map.
    ncomp : int
        Number of components of the output template.
    pix_type : str, optional
        Pixelization type.

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


def sky_average(map, pix_type="hp"):
    """
    Compute the sky average of a map
    depending on its pixelization type.
    Parameters
    ----------
    map : np.ndarray or enmap.ndmap
        Input map.
    pix_type : str, optional
        Pixelization type. Either 'hp' or 'car'.
    Returns
    -------
    float
        Sky average of the map.
    """
    _check_pix_type(pix_type)
    if pix_type == "hp":
        if len(map.shape) > 1:
            return np.mean(map, axis=1)
        else:
            return np.mean(map)
    else:
        shape, wcs = map.geometry
        pixel_area_sr = enmap.pixsizemap(shape, wcs)

        if len(map.shape) > 2:
            avg = []
            for i in range(map.shape[0]):
                weighted_sum = np.sum(
                    map[i] * pixel_area_sr,
                )
                avg.append(weighted_sum / np.sum(pixel_area_sr))
            return np.array(avg)
        else:
            weighted_sum = np.sum(
                map * pixel_area_sr,
            )
            return weighted_sum / np.sum(pixel_area_sr)


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
        Pixelization type.

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


def get_spin_derivatives(map):
    """
    First and second spin derivatives of a given spin-0 map.

    Parameters
    ----------
    map : np.ndarray or enmap.ndmap, shape (map_shape,)
        Input map (spin-0).

    Returns
    -------
    first : np.ndarray or enmap.ndmap, shape (map_shape,)
        First map spin derivative.
    second : np.ndarray or enmap.ndmap, shape (map_shape,)
        Second map spin derivative.
    """
    pix_type = "hp"
    if hasattr(map, "wcs"):
        pix_type = "car"
    ell = np.arange(lmax_from_map(map, pix_type=pix_type) + 1)
    alpha1i = np.sqrt(ell*(ell + 1.))
    alpha2i = np.sqrt((ell - 1.)*ell*(ell + 1.)*(ell + 2.))

    if pix_type == "car":
        if map.ndim > 2:
            raise ValueError(
                f"Input CAR map has too many ({map.ndim}) dimensions.")
        temp = enmap.zeros(map.shape, wcs=map.wcs)
        nside = None
    else:
        if map.ndim > 2:
            raise ValueError(
                f"Input HP map has too many ({map.ndim}) dimensions.")
        nside = hp.npix2nside(np.shape(map)[-1])
        temp = None

    def almtomap(alm):
        return alm2map(alm, pix_type=pix_type, nside=nside,
                       car_map_template=temp)

    def maptoalm(map):
        return map2alm(map, pix_type=pix_type)

    first = almtomap(hp.almxfl(maptoalm(map), alpha1i))
    second = almtomap(hp.almxfl(maptoalm(map), alpha2i))

    return first, second


def get_binary_mask_from_nhits(nhits_map, nside, zero_threshold=1e-3):
    """
    Make binary mask by smoothing, normalizing and thresholding nhits map.

    NOTE: This function has been deprecated and is no longer in use.
    """
    nhits_smoothed = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi/180)
    nhits_smoothed[nhits_smoothed < 0] = 0
    nhits_smoothed /= np.amax(nhits_smoothed)
    binary_mask = np.zeros_like(nhits_smoothed)
    binary_mask[nhits_smoothed > zero_threshold] = 1

    return binary_mask


def get_apodized_mask_from_nhits(nhits_map, nside,
                                 galactic_mask=None,
                                 point_source_mask=None,
                                 zero_threshold=1e-3,
                                 apod_radius=10.,
                                 apod_radius_point_source=4.,
                                 apod_type="C1"):
    """
    Produce an appropriately apodized mask from an nhits map as used in
    the BB pipeline paper (https://arxiv.org/abs/2302.04276).

    NOTE: This function has been deprecated and is no longer in use.

    Procedure:
    * Make binary mask by smoothing, normalizing and thresholding nhits map
    * (optional) multiply binary mask by galactic mask
    * Apodize (binary * galactic)
    * (optional) multiply (binary * galactic) with point source mask
    * (optional) apodize (binary * galactic * point source)
    * Multiply everything by (smoothed) nhits map
    """
    import pymaster as nmt

    # Smooth and normalize hits map
    nhits_map = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi/180)
    nhits_map /= np.amax(nhits_map)

    # Get binary mask
    binary_mask = get_binary_mask_from_nhits(nhits_map, nside, zero_threshold)

    # Multiply by Galactic mask
    if galactic_mask is not None:
        binary_mask *= hp.ud_grade(galactic_mask, nside)

    # Apodize the binary mask
    binary_mask = nmt.mask_apodization(binary_mask, apod_radius,
                                       apotype=apod_type)

    # Multiply with point source mask
    if point_source_mask is not None:
        binary_mask *= hp.ud_grade(point_source_mask, nside)
        binary_mask = nmt.mask_apodization(binary_mask,
                                           apod_radius_point_source,
                                           apotype=apod_type)

    return nhits_map * binary_mask
