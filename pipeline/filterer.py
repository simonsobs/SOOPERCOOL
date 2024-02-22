import argparse
import healpy as hp
from soopercool import BBmeta


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

    # Read the mask
    mask = meta.read_mask("binary")

    meta.timer.start(f"Filter {meta.tf_est_num_sims} sims for TF estimation.")
    if args.transfer:

        filtering_tags = meta.get_filtering_tags()
        filter_funcs = {ftag: meta.get_filter_function(ftag) for ftag in filtering_tags}

        for cl_type in ["cosmo", "tf_est", "tf_val"]:
            cases_list = ["pureE", "pureB"] if cl_type == "tf_est" else [None]
            for id_sim in range(meta.tf_est_num_sims):
                for case in cases_list:

                    for ftag in filtering_tags:
                        map_file = meta.get_map_filename_transfer(
                            id_sim,
                            cl_type,
                            pure_type=case,
                            filter_tag=ftag
                        )
                        map = hp.read_map(map_file, field=[0, 1, 2])
                        filter_map = filter_funcs[ftag]
                        filter_map(map, map_file, mask)
    meta.timer.stop(f"Filter {meta.tf_est_num_sims} sims for TF estimation.",
                    verbose=True)

    if args.sims or args.data:
        Nsims = meta.num_sims if args.sims else 1
        meta.timer.start(f"Filter {Nsims} sims.")
        for map_name in meta.maps_list:
            map_set, id_split = map_name.split("__")
            for id_sim in range(Nsims):
                map_file = meta.get_map_filename(
                    map_set,
                    id_split,
                    id_sim=id_sim if Nsims > 1 else None
                )
                map = hp.read_map(map_file, field=[0, 1, 2])
                filter_map(map, map_file, mask)
        meta.timer.stop(f"Filter {Nsims} sims.", verbose=True)


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
