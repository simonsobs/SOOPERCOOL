import argparse
from soopercool import BBmeta
from soopercool import map_utils as mu
from soopercool import mpi_utils as mpi
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def build_noise_ps_matrix(nl_dict, map_sets_list):
    """
    """
    ms_and_fields = [(ms, f) for ms in map_sets_list for f in "TEB"]
    cls_matrix = []
    for i, (ms1, f1) in enumerate(ms_and_fields):
        for j, (ms2, f2) in enumerate(ms_and_fields):
            if j < i:
                continue
            cls_matrix.append(nl_dict[ms1, ms2][f1+f2])
    return cls_matrix


def get_auto_cls_from_dict(cl_dict, map_set):
    """
    """
    cls = []
    for fp in ["TT", "TE", "TB", "EE", "EB", "BB"]:
        cl = cl_dict[map_set, map_set][fp]
        cls.append(cl)

    return np.array(cls)


def generate_alms_from_cls(cls, seed=None):
    """
    """
    if seed is not None:
        np.random.seed(seed)
    return hp.synalm(cls)


def generate_noise_maps_from_alms(alms, nside):
    """
    """
    return hp.alm2map(alms, nside=nside)


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    out_dir = meta.output_directory
    nl_dir = f"{out_dir}/noise_interp"

    noise_sims_dir = f"{out_dir}/noise_sims"
    plot_dir = f"{out_dir}/plots/noise_sims"
    BBmeta.make_dir(noise_sims_dir)
    BBmeta.make_dir(plot_dir)

    sat_map_sets_list = [
        ms for ms in meta.map_sets_list
        if "sat" in meta.exp_tag_from_map_set(ms).lower()
    ]
    sat_map_set_pairs_list = [
        (ms1, ms2)
        for (ms1, ms2) in meta.get_ps_names_list(type="all", coadd=True)
        if ("sat" in meta.exp_tag_from_map_set(ms1).lower() and
            "sat" in meta.exp_tag_from_map_set(ms2).lower())
    ]

    # Load noise power spectra
    nl_dict = {}
    for ms1, ms2 in sat_map_set_pairs_list:
        nl_file = f"{nl_dir}/nl_{ms1}_x_{ms2}.npz"
        nl = np.load(nl_file)
        nl_dict[ms1, ms2] = {k: nl[k] for k in nl.keys()}
    ll = nl_dict[ms1, ms2]["ell"]

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):

        for ms in sat_map_sets_list:
            if verbose:
                print(f" # {id_sim} | {ms}")

            nls = get_auto_cls_from_dict(nl_dict, ms)
            n_bundles = meta.n_bundles_from_map_set(ms)
            for id_bundle in range(n_bundles):
                noise_alms = generate_alms_from_cls(
                    nls, seed=id_sim*n_bundles+id_bundle
                )
                noise_maps = hp.alm2map(noise_alms, nside=meta.nside)

                fname = f"homogeneous_noise_{ms}_bundle{id_bundle}_{id_sim:04d}.fits" # noqa
                mu.write_map(f"{noise_sims_dir}/{fname}", noise_maps[ms],
                             dtype=np.float32, convert_muK_to_K=True)
                for ip, fp in enumerate(["TT", "EE", "BB", "TE"]):
                    plt.title(f"{ms} | {fp}")
                    plt.plot(hp.anafast(noise_maps)[ip], c="navy",
                             label="Data")
                    plt.plot(ll, nl_dict[ms, ms][fp], "k--", label="Input")
                    plt.yscale("log")
                    plt.ylabel(r"$C_\ell$", fontsize=14)
                    plt.xlabel(r"\ell$", fontsize=14)
                    plt.legend()
                    plt.savefig(
                        f"{plot_dir}/noise_cl_{ms}_{fp}_{id_sim:04d}.png"
                    )
                    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
