import numpy as np
import healpy as hp
from pixell import enmap
import matplotlib.pyplot as plt


def _check_pix_type(pix_type):
    if not (pix_type in ['hp', 'car']):
        raise ValueError(f"Unknown pixelisation type {pix_type}.")


def ud_grade(map_in, nside_out, power=None, pix_type='hp'):
    if pix_type != 'hp':
        raise ValueError("Can't U/D-grade non-HEALPix maps")
    return hp.ud_grade(map_in, nside_out=nside_out, power=power)


def read_map(map_file, field=0, pix_type='hp', convert_K_to_muK=False):
    """
    """
    conv = 1
    if convert_K_to_muK:
        conv = 1.e6
    _check_pix_type(pix_type)
    if pix_type == 'hp':
        return conv*np.array(hp.read_map(map_file, field=field))
    else:
        # Read all maps
        mp = enmap.read_map(map_file)
        # Slice up
        if isinstance(field, int):
            field = [field]
        elif isinstance(field, str):
            raise KeyError("Can't select CAR maps based on column name")
        return conv*mp[field]


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
        map.write(map_file, fmt=dtype)


def plot_map(map, title=None, file_name=None, lims=None):
    """
    """
    ncomp = map.shape[0] if len(map.shape) == 2 else 1
    cmap = "YlOrRd" if ncomp == 1 else "RdYlBu_r"
    kwargs = {"title": title, "cmap": cmap}

    if lims is None:
        range_args = [{} for i in range(ncomp)]

    if ncomp == 1:
        if lims is not None:
            range_args = [{
                "min": lims[0],
                "max": lims[1]
            }]
        to_plot = [map]

    elif ncomp == 3:
        if lims is not None:
            range_args = [
                {
                    "min": lims[i][0],
                    "max": lims[i][1]
                } for i in lims(3)
            ]
    for i in range(ncomp):
        hp.mollview(to_plot[i], **kwargs, **range_args[i])

        if file_name:
            if ncomp == 1:
                plt.savefig(f"{file_name}.png", bbox_inches="tight")
            else:
                plt.savefig(f"{file_name}_{'TQU'[i]}.png", bbox_inches="tight")
        else:
            plt.show()
