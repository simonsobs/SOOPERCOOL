import healpy as hp
import matplotlib.pyplot as plt


def read_map(map_file, ncomp):
    """
    """
    return hp.read_map(map_file, field=[i for i in range(ncomp)])


def write_map(map_file, map, dtype=None):
    """
    """
    hp.write_map(map_file, map, overwrite=True, dtype=dtype)


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
