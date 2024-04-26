import argparse
from soopercool import BBmeta
import sacc
from itertools import product
import numpy as np
import soopercool.mpi_utils as mpi_utils


def multi_eye(size, k_list):
    """
    """
    return np.sum([np.eye(size, k=k) for k in k_list], axis=0)


def thin_covariance(cov, n_bins, n_fields, order=None):
    """
    """
    if order is None:
        return cov
    else:
        k_list = list(range(-order, order+1))
        eye = multi_eye(size=n_bins, k_list=k_list)
        eye = np.tile(eye, (n_fields, n_fields))

        return eye * cov


def saccer(args):
    """
    This script will compile outputs of `coadder.py`
    and `covfefe.py` into a single `sacc` file for
    the data and a `sacc` file for each simulation.
    """

    meta = BBmeta(args.globals)

    nmt_binning = meta.read_nmt_binning()
    ls = np.arange(nmt_binning.lmax+1)
    lb = nmt_binning.get_effective_ells()
    n_bins = len(lb)
    lwin = np.zeros((len(ls), n_bins))

    for id_bin in range(n_bins):
        weights = np.array(nmt_binning.get_weight_list(id_bin))
        multipoles = np.array(nmt_binning.get_ell_list(id_bin))
        for il, l in enumerate(multipoles):
            lwin[l, id_bin] = weights[il]

    field_pairs = [m1+m2 for m1, m2 in product("TEB", repeat=2)]

    Nsims = meta.num_sims if args.sims else 1

    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory

    cov_dir = meta.covmat_directory
    sacc_dir = meta.sacc_directory

    data_types = {"T": "0", "E": "e", "B": "b"}
    map_sets = meta.map_sets_list
    ps_names = meta.get_ps_names_list(type="all", coadd=True)

    covs = {}
    for i, (ms1, ms2) in enumerate(ps_names):
        for j, (ms3, ms4) in enumerate(ps_names):

            if i > j:
                continue
            cov_dict = np.load(f"{cov_dir}/mc_cov_{ms1}_{ms2}_{ms3}_{ms4}.npz")

            cov_size = len(field_pairs)*len(lb)
            cov = np.zeros((cov_size, cov_size))
            for ifp1, fp1 in enumerate(field_pairs):
                for ifp2, fp2 in enumerate(field_pairs):
                    cov[ifp1*len(lb):(ifp1+1)*len(lb),
                        ifp2*len(lb):(ifp2+1)*len(lb)] = cov_dict[fp1+fp2]

            covs[ms1, ms2, ms3, ms4] = thin_covariance(
                cov, len(lb), len(field_pairs), order=3
            )

    full_cov_size = len(ps_names)*len(lb)*len(field_pairs)
    full_cov = np.zeros((full_cov_size, full_cov_size))

    for i, (ms1, ms2) in enumerate(ps_names):
        for j, (ms3, ms4) in enumerate(ps_names):
            if i > j:
                continue

            full_cov[
                i*len(field_pairs)*len(lb):(i+1)*len(field_pairs)*len(lb),
                j*len(field_pairs)*len(lb):(j+1)*len(field_pairs)*len(lb)
            ] = covs[ms1, ms2, ms3, ms4]

    # Symmetrize
    full_cov = np.triu(full_cov)
    full_cov += full_cov.T - np.diag(full_cov.diagonal())

    # Initialize MPI
    use_mpi4py = args.sims
    mpi_utils.init(use_mpi4py)

    for id_sim in mpi_utils.taskrange(Nsims - 1):

        sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""

        s_wins = sacc.BandpowerWindow(ls, lwin)

        s = sacc.Sacc()

        for ms in map_sets:
            f = float(meta.freq_tag_from_map_set(ms))
            for spin, qty in zip(
                [0, 2],
                ["cmb_temperature", "cmb_polarization"]
            ):

                s.add_tracer(**{
                    "tracer_type": "NuMap",
                    "name": f"{ms}",
                    "quantity": qty,
                    "spin": spin,
                    "nu": [f-1., f, f+1],  # TODO
                    "ell": lb,
                    "beam": np.ones_like(lb),  # TODO,
                    "bandpass": [0., 1., 0.]  # TODO
                })

        for i, (ms1, ms2) in enumerate(ps_names):

            cl_file = f"{cl_dir}/decoupled_cross_pcls_nobeam_{ms1}_{ms2}{sim_label}.npz" # noqa
            cells = np.load(cl_file)
            for fp in field_pairs:

                f1, f2 = fp
                s.add_ell_cl(**{
                    "data_type": f"cl_{data_types[f1]}{data_types[f2]}",
                    "tracer1": f"{ms1}",
                    "tracer2": f"{ms2}",
                    "ell": lb,
                    "x": cells[fp],
                    "window": s_wins
                })

        s.add_covariance(full_cov)

        s.save_fits(
            f"{sacc_dir}/cl_and_cov_sacc{sim_label}.fits",
            overwrite=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sacc compilation of power spectra and covariances."
    )

    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--data", action="store_true")
    args = parser.parse_args()
    saccer(args)
