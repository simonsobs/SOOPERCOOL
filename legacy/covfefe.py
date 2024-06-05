import argparse
from soopercool import BBmeta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def covFeFe(args):
    """
    This function handles the covariance matrix computation.
    We can use a set of simulations to build a numerical estimate
    of the covariance matrices.
    TODO: implement an analytical estimate of the covariances
    """
    meta = BBmeta(args.globals)
    cov_dir = meta.covmat_directory
    n_bins = meta.get_n_bandpowers()

    Nsims = meta.num_sims
    cl_dir = meta.cell_sims_directory

    field_pairs = [f"{m1}{m2}" for m1 in "TEB" for m2 in "TEB"]

    cross_ps_names = meta.get_ps_names_list(type="all", coadd=True)

    # Build the covariance matrix elements to compute
    cov_names = []
    for i, (ms1, ms2) in enumerate(cross_ps_names):
        for j, (ms3, ms4) in enumerate(cross_ps_names):
            if i > j:
                continue
            cov_names.append((ms1, ms2, ms3, ms4))

    # Load the simulations
    cl_dict = {}
    for ms1, ms2 in cross_ps_names:
        cl_dict[ms1, ms2] = []
        for iii in range(Nsims):
            cells_dict = np.load(
                f"{cl_dir}/decoupled_cross_pcls_nobeam_{ms1}_{ms2}_{iii:04d}.npz", # noqa
            )
            cl_vec = np.concatenate(
                [
                    cells_dict[field_pair] for field_pair in field_pairs
                ]
            )
            cl_dict[ms1, ms2].append(cl_vec)

        cl_dict[ms1, ms2] = np.array(cl_dict[ms1, ms2])

    full_cov_dict = {}
    for ms1, ms2, ms3, ms4 in cov_names:

        cl12 = cl_dict[ms1, ms2]
        cl34 = cl_dict[ms3, ms4]

        cl12_mean = np.mean(cl12, axis=0)
        cl34_mean = np.mean(cl34, axis=0)
        cov = np.mean(
            np.einsum("ij,ik->ijk", cl12-cl12_mean, cl34-cl34_mean),
            axis=0
        )
        full_cov_dict[ms1, ms2, ms3, ms4] = cov

        cov_dict = {}
        for i, field_pair_1 in enumerate(field_pairs):
            for j, field_pair_2 in enumerate(field_pairs):

                cov_block = cov[i*n_bins:(i+1)*n_bins, j*n_bins:(j+1)*n_bins]
                cov_dict[field_pair_1 + field_pair_2] = cov_block

        np.savez(
            f"{cov_dir}/mc_cov_{ms1}_{ms2}_{ms3}_{ms4}.npz",
            **cov_dict
        )

    if args.plots:
        plot_dir = meta.plot_dir_from_output_dir(
            meta.covmat_directory_rel
        )
        n_fields = len(field_pairs)
        n_spec = len(cross_ps_names)

        full_size = n_spec*n_fields*n_bins
        full_cov = np.zeros((full_size, full_size))

        for i, (ms1, ms2) in enumerate(cross_ps_names):
            for j, (ms3, ms4) in enumerate(cross_ps_names):
                if i > j:
                    continue

                full_cov[
                    i*n_fields*n_bins:(i+1)*n_fields*n_bins,
                    j*n_fields*n_bins:(j+1)*n_fields*n_bins
                ] = full_cov_dict[ms1, ms2, ms3, ms4]

        # Symmetrize
        full_cov = np.triu(full_cov)
        full_cov += full_cov.T - np.diag(full_cov.diagonal())
        covdiag = full_cov.diagonal()
        full_corr = full_cov / np.outer(np.sqrt(covdiag), np.sqrt(covdiag))

        plt.figure(figsize=(8, 8))
        im = plt.imshow(full_corr, vmin=-1, vmax=1, cmap="RdBu_r")
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size=0.3, pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.savefig(f"{plot_dir}/full_corr.png", dpi=300,
                    bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Covariance matrix calculator"
    )
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true", help="Generate plots")

    args = parser.parse_args()
    covFeFe(args)
