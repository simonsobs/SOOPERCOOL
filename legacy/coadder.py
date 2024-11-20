from soopercool import BBmeta
from itertools import product
import numpy as np
import argparse
import soopercool.mpi_utils as mpi_utils


def coadder(args):
    """
    This script is used to coadd the cross-split power spectra
    (e.g. SAT1_f093__0 x SAT_f093__1) into cross-map-set power
    spectra (e.g. SAT1_f093 x SAT_f093). It will produce both
    cross and auto map-set power spectra from which we derive
    the noise power spectra.
    """
    meta = BBmeta(args.globals)
    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()
    field_pairs = [m1 + m2 for m1, m2 in product("TEB", repeat=2)]

    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory

    # Set the number of sims to loop over
    Nsims = meta.num_sims if args.sims else 1

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False),
    }

    cross_split_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    # Initialize MPI
    use_mpi4py = args.sims
    mpi_utils.init(use_mpi4py)

    # Load split C_ells
    for id_sim in mpi_utils.taskrange(Nsims - 1):

        sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""

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

            map_set1, _ = map_name1.split("__")
            map_set2, _ = map_name2.split("__")

            cells_dict = np.load(
                f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz"  # noqa
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
                    fp: cells_coadd[type][(map_set1, map_set2)][fp]
                    for fp in field_pairs
                }
                np.savez(
                    f"{cl_dir}/decoupled_{type}_pcls_nobeam_{map_set1}_{map_set2}{sim_label}.npz",  # noqa
                    lb=lb,
                    **cells_to_save,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")

    args = parser.parse_args()

    coadder(args)
