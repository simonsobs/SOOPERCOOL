from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
from soopercool import mpi_utils as mpi
import argparse
import numpy as np
import pymaster as nmt


def main(args):
    """ """
    meta = BBmeta(args.globals)
    # do_plots = not args.no_plots
    verbose = args.verbose

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells_sims"
    couplings_dir = f"{out_dir}/couplings"
    sims_dir = f"{out_dir}/cov_sims"

    BBmeta.make_dir(cells_dir)

    mask = mu.read_map(meta.masks["analysis_mask"], pix_type=meta.pix_type)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"], binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        inv_couplings_beamed[ms1, ms2] = np.load(
            f"{couplings_dir}/couplings_{ms1}_{ms2}.npz"
        )["inv_coupling"].reshape(
            [n_bins * 9, n_bins * 9]
        )  # noqa

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):
        base_dir = f"{sims_dir}/{id_sim:04d}"

        # Create namaster fields
        fields = {}
        for map_name in meta.maps_list:
            map_set, id_bundle = map_name.split("__")
            map_fname = f"{base_dir}/cov_sims_{map_set}_bundle{id_bundle}.fits"

            m = mu.read_map(
                map_fname,
                field=[0, 1, 2],
                pix_type=meta.pix_type,
                convert_K_to_muK=True,
            )
            field_spin0 = nmt.NmtField(mask, m[:1])
            field_spin2 = nmt.NmtField(mask, m[1:], purify_b=meta.pure_B)

            fields[map_set, id_bundle] = {"spin0": field_spin0, "spin2": field_spin2}

        for map_name1, map_name2 in meta.get_ps_names_list(type="all", coadd=False):
            map_set1, id_split1 = map_name1.split("__")
            map_set2, id_split2 = map_name2.split("__")
            if verbose:
                print(
                    f"# {id_sim+1} | ({map_set1}, split {id_split1}) x "
                    f"({map_set2}, split {id_split2})"
                )
            pcls = pu.get_coupled_pseudo_cls(
                fields[map_set1, id_split1], fields[map_set2, id_split2], nmt_bins
            )

            decoupled_pcls = pu.decouple_pseudo_cls(
                pcls, inv_couplings_beamed[map_set1, map_set2]
            )

            np.savez(
                f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}_{id_sim:04d}.npz",  # noqa
                **decoupled_pcls,
                lb=nmt_bins.get_effective_ells(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument(
        "--no-plots", action="store_false", help="Do not make plots."
    )  # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
