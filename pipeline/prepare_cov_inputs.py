from soopercool import BBmeta
from itertools import product
import numpy as np
import argparse
import pymaster as nmt
from soopercool import utils as su
from scipy.interpolate import interp1d


def main(args):
    """
    The main purpose of this script is to prepare signal
    and noise power spectra used in the analytic covariance
    prescription. These are computed on the data from
    coupled cross-bundle power spectra.
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells"

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    lb = nmt_bins.get_effective_ells()
    ell = np.arange(nmt_bins.lmax + 1)
    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False)
    }

    cross_split_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    # Load split C_ells

    # Initialize output dictionary
    cells_coadd = {
        "cross": {
            (ms1, ms2): {
                fp: [] for fp in field_pairs
            } for ms1, ms2 in cross_map_set_list
        },
        "auto": {
            (ms1, ms2): {
                fp: [] for fp in field_pairs
            } for ms1, ms2 in cross_map_set_list
        }
    }

    # Load beams
    beams = {}
    for map_set in meta.map_sets_list:
        beam_dir = meta.beam_dir_from_map_set(map_set)
        beam_file = meta.beam_file_from_map_set(map_set)

        _, bl = su.read_beam_from_file(
            f"{beam_dir}/{beam_file}",
            lmax=nmt_bins.lmax
        )
        beams[map_set] = bl

    # Loop over all map set pairs
    for map_name1, map_name2 in cross_split_list:

        map_set1, _ = map_name1.split("__")
        map_set2, _ = map_name2.split("__")

        cells_dict = np.load(
            f"{cells_dir}/weighted_pcls_{map_name1}_x_{map_name2}.npz"
        )

        if (map_name1, map_name2) in ps_names["cross"]:
            type = "cross"
        elif (map_name1, map_name2) in ps_names["auto"]:
            type = "auto"

        for field_pair in field_pairs:

            cells_coadd[type][map_set1, map_set2][field_pair] += [cells_dict[field_pair]] # noqa

    # Average the cross-split power spectra
    cells_coadd["noise"] = {}
    for map_set1, map_set2 in cross_map_set_list:
        cells_coadd["noise"][(map_set1, map_set2)] = {}
        for field_pair in field_pairs:
            for type in ["cross", "auto"]:
                if len(cells_coadd[type][map_set1, map_set2][field_pair]) != 0:
                    cells_coadd[type][map_set1, map_set2][field_pair] = \
                        np.mean(
                            cells_coadd[type][map_set1, map_set2][field_pair],
                            axis=0
                        )

            if len(cells_coadd["auto"][map_set1, map_set2][field_pair]) == 0:
                cells_coadd["noise"][map_set1, map_set2][field_pair] = np.zeros_like(cells_coadd["cross"][map_set1, map_set2][field_pair])  # noqa
                cells_coadd["auto"][map_set1, map_set2][field_pair] = np.zeros_like(cells_coadd["cross"][map_set1, map_set2][field_pair])  # noqa
            else:
                cells_coadd["noise"][(map_set1, map_set2)][field_pair] = \
                    cells_coadd["auto"][map_set1, map_set2][field_pair] - \
                    cells_coadd["cross"][map_set1, map_set2][field_pair]

        # First attempt at introducing filtering corrections
        # This is a handwavy approach to interpolate the per
        # multipole TF
        #
        # Robust interpolation and clipping of per-multipole TF
        tf_settings = meta.transfer_settings
        tf_dir = tf_settings["transfer_directory"]
        ftag1 = meta.filtering_tag_from_map_set(map_set1)
        ftag2 = meta.filtering_tag_from_map_set(map_set2)
        tf = np.load(f"{tf_dir}/transfer_function_{ftag1}_x_{ftag2}.npz")

        tf_interp = {}
        cap_ell = min(600, nmt_bins.lmax)  # don't index past lmax
        tiny = 1e-8

        for fp in field_pairs:
            name = f"{fp}_to_{fp}"
            # interpolate TF defined at bandpower centers lb onto all ell
            tfint_fn = interp1d(
                lb, tf[name],
                kind="linear",
                bounds_error=False,
                fill_value=(tf[name][0], tf[name][-1])  # edge-hold instead of extrapolate
            )
            tfint = tfint_fn(ell)

            # sanitize NaNs/Infs
            tfint = np.nan_to_num(tfint, nan=tiny, posinf=np.max(tfint[np.isfinite(tfint)]), neginf=tiny)

            # cap high-ell tail to a constant to avoid noise blow-up
            tfint[ell > cap_ell] = tfint[cap_ell]

            # if any negatives slipped in, flatten up to last-negative; else just clip to tiny
            neg_idx = np.where(tfint < 0)[0]
            if neg_idx.size > 0:
                k = neg_idx.max()
                rep = min(k + 1, tfint.size - 1)
                tfint[:k+1] = max(tfint[rep], tiny)
            else:
                tfint = np.maximum(tfint, tiny)

            tf_interp[fp] = tfint

        for type in ["cross", "noise"]:
            cells_to_save = {
                # ansatz for the fitlering to be improved.
                # This is a temporary placeholder
                # Need to fit for the TF exponent
                fp: cells_coadd[type][map_set1, map_set2][fp] / tf_interp[fp] * tf_interp[fp] ** 0.75  # noqa
                for fp in field_pairs
            }
            # Note that we don't deconvolve the beam above to
            # avoid huge numerical instabilities when deconvolving
            # the MCMs
            if type == "cross":
                for fp in ["TB", "EB", "BE", "BT"]:
                    cells_to_save[fp] = np.zeros_like(cells_to_save[fp])
            elif type == "noise":
                for fp in field_pairs:
                    if fp != fp[::-1]:
                        cells_to_save[fp] = np.zeros_like(cells_to_save[fp])
            np.savez(
                f"{cells_dir}/weighted_{type}_pcls_{map_set1}_x_{map_set2}.npz",  # noqa
                ell=np.arange(len(cells_to_save["TT"])),
                **cells_to_save
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare covariance inputs for SOOPERCOOL pipeline"
    )
    parser.add_argument(
        "--globals",
        type=str,
        required=True,
        help="Path to the globals file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    main(args)
