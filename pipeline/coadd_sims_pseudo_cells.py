from soopercool import BBmeta
from soopercool import mpi_utils as mpi
from itertools import product
import numpy as np
import argparse
import pymaster as nmt


def main(args):
    """
    This script is used to coadd the cross-split power spectra
    (e.g. SAT1_f093__0 x SAT_f093__1) into cross-map-set power
    spectra (e.g. SAT1_f093 x SAT_f093). It will produce both
    cross and auto map-set power spectra from which we derive
    the noise power spectra.
    """
    meta = BBmeta(args.globals)
    # do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_sims"

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)
    lb = nmt_bins.get_effective_ells()
    field_pairs = [m1 + m2 for m1, m2 in product("TEB", repeat=2)]

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False),
    }

    cross_split_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    # Load split C_ells
    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):
        # Initialize output dictionary
        cells_coadd = {
            "cross": {
                (ms1, ms2): {fp: [] for fp in field_pairs}
                for ms1, ms2 in cross_map_set_list
            },
            "auto": {
                (ms1, ms2): {fp: [] for fp in field_pairs}
                for ms1, ms2 in cross_map_set_list
            },
        }

        # Loop over all map set pairs
        for map_name1, map_name2 in cross_split_list:
            if verbose:
                print(f"# {id_sim} | {map_name1} x {map_name2}")

            map_set1, _ = map_name1.split("__")
            map_set2, _ = map_name2.split("__")

            cells_dict = np.load(
                f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz"  # noqa
            )

            if (map_name1, map_name2) in ps_names["cross"]:
                type = "cross"
            elif (map_name1, map_name2) in ps_names["auto"]:
                type = "auto"

            for field_pair in field_pairs:

                cells_coadd[type][map_set1, map_set2][field_pair] += [
                    cells_dict[field_pair]
                ]  # noqa

        # Average the cross-split power spectra
        cells_coadd["noise"] = {}
        for map_set1, map_set2 in cross_map_set_list:
            cells_coadd["noise"][(map_set1, map_set2)] = {}
            for field_pair in field_pairs:
                for type in ["cross", "auto"]:
                    cells_coadd[type][map_set1, map_set2][field_pair] = np.mean(
                        cells_coadd[type][map_set1, map_set2][field_pair], axis=0
                    )

                cells_coadd["noise"][(map_set1, map_set2)][field_pair] = (
                    cells_coadd["auto"][map_set1, map_set2][field_pair]
                    - cells_coadd["cross"][map_set1, map_set2][field_pair]
                )

            for type in ["cross", "auto", "noise"]:
                cells_to_save = {
                    fp: cells_coadd[type][map_set1, map_set2][fp] for fp in field_pairs
                }
                np.savez(
                    f"{cells_dir}/decoupled_{type}_pcls_{map_set1}_x_{map_set2}_{id_sim:04d}.npz",  # noqa
                    lb=lb,
                    **cells_to_save,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--no-plots", action="store_true", help="Do not make plots")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    mode = parser.add_mutually_exclusive_group()

    args = parser.parse_args()

    main(args)
