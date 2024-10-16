#!/bin/sh python3

import numpy as np
import healpy as hp
import h5py


def read_hdf5_map(fname, to_nest=False):
    """
    """
    f = h5py.File(fname, "r")
    dset = f["map"]
    header = dict(dset.attrs)

    if header["ORDERING"] == "NESTED":
        file_nested = True
    elif header["ORDERING"] == "RING":
        file_nested = False

    _, npix = dset.shape

    if file_nested and not to_nest:
        mapdata = hp.reorder(dset[:], n2r=True)
    elif not file_nested and to_nest:
        mapdata = hp.reorder(dset[:], r2n=True)
    else:
        mapdata = dset

    return mapdata


def write_hdf5_map(fname, nside, dict_maps, list_of_obsid,
                   nest_or_ring='RING'):
    """
    """
    with h5py.File(fname, 'w') as f:
        f.attrs['NSIDE'] = nside
        f.attrs['ORDERING'] = nest_or_ring
        f.attrs['OBS_ID'] = list_of_obsid
        for k, v in dict_maps.items():
            f.create_dataset(k, data=v)


def gen_masks_of_given_atomic_map_list_for_bundles(nmaps, nbundles):
    """
    Makes a list (length nbundles) of boolean lists (length nmaps)
    corresponding to the atomic maps that have to be coadded to make up a
    given bundle. This is done by uniformly distributing atomic maps into each
    bundle and, if necessary, looping through the bundles until the remainders
    have gone.

    Parameters
    ----------
    nmaps: int
        Number of atomic maps to distribute.
    nbundles: int
        Number of map bundles to be generated.

    Returns
    -------
    boolean_mask_list: list of list of str
        List of lists of booleans indicating the atomics to be coadded for
        each bundle.
    """

    n_per_bundle = nmaps // nbundles
    nremainder = nmaps % nbundles
    boolean_mask_list = []

    for idx in range(nbundles):
        if idx < nremainder:
            _n_per_bundle = n_per_bundle + 1
        else:
            _n_per_bundle = n_per_bundle

        i_begin = idx * _n_per_bundle
        i_end = (idx+1) * _n_per_bundle

        _m = np.zeros(nmaps, dtype=np.bool_)
        _m[i_begin:i_end] = True

        boolean_mask_list.append(_m)

    return boolean_mask_list
