from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
import argparse
import numpy as np
import pymaster as nmt
import re


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    # do_plots = not args.no_plots
    # verbose = args.verbose

    out_dir = meta.output_directory
    print(out_dir)
    cells_dir = f"{out_dir}/cells"
    couplings_dir = f"{out_dir}/couplings"

    BBmeta.make_dir(cells_dir)

    mask = mu.read_map(meta.masks["analysis_mask"], ncomp=1)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    # Create namaster fields
    fields = {}
    for map_name in meta.maps_list:
        map_set, id_bundle = map_name.split("__")

        # Load maps
        map_dir = meta.map_dir_from_map_set(map_set)
        map_template = meta.map_template_from_map_set(map_set)

        map_file = map_template.replace(
            "{id_bundle}",
            str(id_bundle)
        )
        type_options = [f for f in re.findall(r"\{.*?\}", map_template) if "|" in f][0] # noqa
        # Select the hitmap
        option = type_options.replace("{", "").replace("}", "").split("|")[0]

        map_file = map_file.replace(
                type_options,
                option
            )

        m = mu.read_map(f"{map_dir}/{map_file}", ncomp=3)
        field_spin0 = nmt.NmtField(mask, m[:1])
        field_spin2 = nmt.NmtField(mask, m[1:], purify_b=meta.pure_B)

        fields[map_set, id_bundle] = {
            "spin0": field_spin0,
            "spin2": field_spin2
        }

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        inv_couplings_beamed[ms1, ms2] = np.load(f"{couplings_dir}/couplings_{ms1}_{ms2}.npz")["inv_coupling"].reshape([n_bins*9, n_bins*9]) # noqa

    for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                       coadd=False):
        map_set1, id_split1 = map_name1.split("__")
        map_set2, id_split2 = map_name2.split("__")
        pcls = pu.get_coupled_pseudo_cls(
                fields[map_set1, id_split1],
                fields[map_set2, id_split2],
                nmt_bins
                )

        decoupled_pcls = pu.decouple_pseudo_cls(
                pcls, inv_couplings_beamed[map_set1, map_set2]
                )

        np.savez(f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}.npz",  # noqa
                 **decoupled_pcls, lb=nmt_bins.get_effective_ells())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no-plots", action="store_false", help="Do not make plots.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
