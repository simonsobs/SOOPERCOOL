from pixell import curvedsky
import healpy as hp
import numpy as np


def get_alm_ordering(fields="TEB", components=None):
    """
    Build an iterator that yields the ordering of the alm components
    given fields and components (e.g. frequencies).
    Expected behavior for T,E,B fields and f1, f2 is :
        almT_f1, almE_f1, almB_f1, almT_f2, almE_f2, almB_f2

    Parameters
    ----------
    fields : str
        The fields to consider. Default is "TEB".
    components : list
        The components to consider. Default is None.
    """
    if components is not None:
        for c in components:
            for f in fields:
                yield c, f
    else:
        for f in fields:
            yield f


def get_ps_matrix_for_sim(ps_dict, lmax, components=None, fields="TEB"):
    """
    Build a matrix of power spectra for simulation of correlated alms.

    Parameters
    ----------
    ps_dict : dict
        A dictionary containing the power spectra.
    lmax : int
        The maximum multipole to consider.
    components : list
        The components to consider. Default is None.
    fields : str
        The fields to consider. Default is "TEB".

    Returns
    -------
    ps_matrix : np.ndarray
        A 3D array containing the power spectra with dimension
        (nfields * ncomponents, nfields * ncomponents, lmax).
    """
    ordering = list(get_alm_ordering(fields=fields, components=components))

    nalms = len(components) * len(fields) if components is not None else len(fields)
    ps_matrix = np.zeros((nalms, nalms, lmax))

    for i, lab1 in enumerate(ordering):
        for j, lab2 in enumerate(ordering):
            if components is not None:
                c1, f1 = lab1
                c2, f2 = lab2
                ps_matrix[i, j, :] = ps_dict[c1, c2][f1 + f2][:lmax]
            else:
                f1 = lab1
                f2 = lab2
                ps_matrix[i, j, :] = ps_dict[f1 + f2][:lmax]
    return ps_matrix


def get_alms_from_cls(ps_dict, lmax, fields="TEB", components=None, seed=None):
    """
    Generate alms from power spectra.

    Parameters
    ----------
    ps_dict : dict
        A dictionary containing the power spectra.
    lmax : int
        The maximum multipole to consider.
    fields : str
        The fields to consider. Default is "TEB".
    components : list
        The components to consider. Default is None.
    seed : int
        The seed for the random number generator. Default is None.
    """
    ps_matrix = get_ps_matrix_for_sim(
        ps_dict, lmax, components=components, fields=fields
    )
    alms = curvedsky.rand_alm(ps_matrix, lmax=lmax, seed=seed)
    ordering = list(get_alm_ordering(fields, components))
    alms_dict = {}
    if components is not None:
        for comp in components:
            alms_dict[comp] = {comp: [alms[ordering.index((comp, f))] for f in fields]}
        return alms_dict

    else:
        return alms


def beam_alms(alms, beam, fields="TEB"):
    """
    Apply a beam to the alms.

    Parameters
    ----------
    alms : dict
        A dictionary containing the alms.
    beam : np.ndarray
        The beam to apply.
    fields : str
        The fields to consider. Default is "TEB".

    Returns
    -------
    beamed_alms : dict
        A dictionary containing the beam-convolved alms.
    """
    for i, field in enumerate(fields):
        alms[i] = curvedsky.almxfl(alms[i], beam)
    return alms


def get_map_from_alms(alms, template):
    """
    Get a map from alms and a template.

    Parameters
    ----------
    alms : np.ndarray
        The alms to use.
    template : np.ndarray
        The template to use.

    Returns
    -------
    map : np.ndarray
        Returns the corresponding map.
    """
    if hasattr(template, "geometry"):
        pix_type = "car"
    else:
        pix_type = "hp"

    if pix_type == "hp":
        map = hp.alm2map(alms, nside=hp.npix2nside(template.shape[-1]))
    else:
        map = curvedsky.alm2map(
            alms,
            template.copy(),
        )
    return map
