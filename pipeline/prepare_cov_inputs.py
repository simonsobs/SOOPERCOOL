from soopercool import BBmeta
from itertools import product
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from scipy.stats import binned_statistic


def gp_fit(x, y, noise_std, xpred):
    """
    """
    kernel = 1.0 * gp.kernels.Matern(
        length_scale=10,
        length_scale_bounds=(1e-4, 1e5),
        nu=5/2
    )
    # kernel = 1.0 * gp.kernels.RBF(
    #     length_scale=10,
    #     length_scale_bounds=(1e-4, 1e4)
    # )
    model = gp.GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=noise_std**2
    )
    model.fit(
        x.reshape(-1, 1),
        y.reshape(-1, 1)
    )
    ypred = model.predict(xpred.reshape(-1, 1))
    return ypred


def bin_data(x, y, bins):
    """
    """
    mu, edges, _ = binned_statistic(
        x, y,
        statistic="mean",
        bins=bins
    )
    std = binned_statistic(
        x, y,
        statistic="std",
        bins=bins
    )[0] / np.sqrt(
        binned_statistic(
            x, y,
            statistic="count",
            bins=bins
        )[0]
    )
    return (edges[1:] + edges[:-1]) / 2, mu, std


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
    cells_tf_dir = f"{out_dir}/cells_tf_est"
    if args.plots:
        plot_dir = f"{out_dir}/plots/cov_inputs"
        meta.make_dir(plot_dir)

    tf_settings = meta.transfer_settings

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    ps_names = {
        "cross": meta.get_ps_names_list(type="cross", coadd=False),
        "auto": meta.get_ps_names_list(type="auto", coadd=False)
    }

    cross_split_list = meta.get_ps_names_list(type="all", coadd=False)
    cross_map_set_list = meta.get_ps_names_list(type="all", coadd=True)

    # Initialize output Cells dict
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

        # Below, we try to build a filtering correction at power spectrum level
        # to model the increased variance induced by filtering
        # We will build this based on a set of simulations which has
        # already been generated for transfer function estimation
        ftag1 = meta.filtering_tag_from_map_set(map_set1)
        ftag2 = meta.filtering_tag_from_map_set(map_set2)

        sim_ids = range(
            tf_settings["sim_id_start"],
            tf_settings["sim_id_start"] + tf_settings["tf_est_num_sims"]
        )

        ps_mat_filtered = []
        ps_mat_unfiltered = []
        for sim_id in sim_ids:
            ps_mat_filtered.append(
                np.load(
                    f"{cells_tf_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_filtered_unbinned_{sim_id:04d}.npz" # noqa
                )["pcls_mat"]
            )
            ps_mat_unfiltered.append(
                np.load(
                    f"{cells_tf_dir}/pcls_mat_tf_est_{ftag1}_x_{ftag2}_unfiltered_unbinned_{sim_id:04d}.npz" # noqa
                )["pcls_mat"]
            )

        ps_mat_filtered = np.array(ps_mat_filtered)
        ps_mat_unfiltered = np.array(ps_mat_unfiltered)

        var_filtered = np.std(ps_mat_filtered, axis=0) ** 2
        var_unfiltered = np.std(ps_mat_unfiltered, axis=0) ** 2

        # Compute 4pt and 2pt diagonal
        # transfer functions
        T4 = np.nan_to_num(var_filtered / var_unfiltered)
        T2 = np.nan_to_num(
            ps_mat_filtered / ps_mat_unfiltered
        )
        invT2sq = np.nan_to_num(
            np.mean(
                1 / T2**2,
                axis=0
            )
        )
        T4_over_T2sq = np.nan_to_num(T4 * invT2sq)

        ps_mat_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        pure_pairs = list(product(
            ["pureT", "pureE", "pureB"],
            ["pureT", "pureE", "pureB"]
        ))
        correction = {}
        for ps_idx, fp in enumerate(ps_mat_pairs):
            pure_idx = pure_pairs.index((f"pure{fp[0]}", f"pure{fp[1]}"))
            y = T4_over_T2sq[pure_idx, ps_idx, :]
            nl = y.shape[0]
            x = np.arange(nl)
            x, y, ystd = bin_data(
                x,
                y,
                [0, 2, 6, 10, 30, 50, 70, 90, 100, 200,
                 300, 400, 500, 600, 700, 800, 900, nl]
            )
            msk_fit = (x >= 2) & (x <= 650)
            corr = gp_fit(
                np.log10(x[msk_fit]),
                np.nan_to_num(y)[msk_fit],
                np.nan_to_num(ystd)[msk_fit],
                np.log10(np.arange(1, nl))
            )
            correction[fp] = np.concatenate(([0], corr))

            if args.plots:
                plt.figure(figsize=(8, 6))
                plt.plot(
                    np.arange(nl),
                    T4_over_T2sq[pure_idx, ps_idx, :],
                    color="k"
                )
                plt.plot(
                    np.arange(1, nl),
                    corr,
                    color="DodgerBlue"
                )
                plt.errorbar(
                    x, y, ystd,
                    fmt=".",
                    color="DodgerBlue",
                    markerfacecolor="white"
                )
                plt.xlim(2, 650)
                plt.ylim(0.8, 1.5)
                plt.savefig(
                    f"{plot_dir}/4pt_correction_{fp}.pdf",
                    bbox_inches="tight"
                )

        # Save out coadded cells with filtering correction applied
        for type in ["cross", "noise"]:
            cells_to_save = {
                fp: cells_coadd[type][map_set1, map_set2][fp] *
                np.nan_to_num(np.sqrt(correction[fp]))
                for fp in field_pairs
            }

            np.savez(
                f"{cells_dir}/weighted_{type}_pcls_{map_set1}_x_{map_set2}.npz",  # noqa
                ell=np.arange(len(cells_to_save["TT"])),
                **cells_to_save
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare covariance inputs"
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
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save plots"
    )
    args = parser.parse_args()
    main(args)
