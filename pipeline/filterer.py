import argparse
import healpy as hp
from bbmaster import utils
import numpy as np
from bbmaster import BBmeta

def filter(args):
    """
    Filtering main routine. Calls the appropriate
    filterer depending on the type of filterer specified
    in the yaml file.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)
    if meta.filtering_type == "m_filterer":
        filter_func = utils.m_filter_map
        kwargs = {"m_cut": meta.m_cut}

    elif meta.filtering_type == "toast":
        filter_func = utils.toast_filter_map
        kwargs = {}

    else:
        raise NotImplementedError(f"Filterer type {meta.filtering_type} not implemented")
    
    # Read the mask
    mask = meta.read_mask("analysis")

    if args.transfer:
        for cl_type in ["cosmo", "tf_est", "tf_val"]:
            cases_list = ["pureE", "pureB"] if cl_type == "tf_est" else [None]
            for id_sim in range(meta.tf_est_num_sims):
                for case in cases_list:
                    map_file = meta.get_map_filename_transfer2(id_sim, cl_type, pure_type=case)
                    map = hp.read_map(map_file, field=[0,1,2])
                    filtered_map = filter_func(map, mask, **kwargs)
                    print(map_file)
                    hp.write_map(map_file.replace(".fits", "_filtered.fits"), filtered_map, overwrite=True, dtype=np.float32)

    if args.sims or args.data:
        Nsims = meta.num_sims if args.sims else 1

        for map_name in meta.maps_list:
            map_set, id_split = map_name.split("__")
            for id_sim in range(Nsims):
                map_file = meta.get_map_filename(map_set, id_split, id_sim=id_sim if Nsims > 1 else None)
                map = hp.read_map(map_file, field=[0,1,2])
                filtered_map = filter_func(map, mask, **kwargs)
                hp.write_map(map_file.replace(".fits", "_filtered.fits"), filtered_map, overwrite=True, dtype=np.float32)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filterer stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transfer", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--data", action="store_true")

    args = parser.parse_args()

    filter(args)
