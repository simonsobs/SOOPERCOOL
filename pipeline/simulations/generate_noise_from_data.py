import argparse
from soopercool import BBmeta
from soopercool import map_utils as mu
import numpy as np
import healpy as hp


def build_noise_ps_matrix(nl_dict, map_sets_list):
    """
    """
    ms_and_fields = [(ms, f) for ms in map_sets_list for f in "TEB"]
    cls = []
    for i, (ms1, f1) in enumerate(ms_and_fields):
        for j, (ms2, f2) in enumerate(ms_and_fields):
            if j < i:
                continue
            cls.append(nl_dict[ms1, ms2][f1+f2])
    return cls


def generate_noise_alms_from_cls(cls, map_sets_list):
    """
    """
    noise_alms = hp.synalm(cls)

    final_alms = {}
    for i, ms in enumerate(map_sets_list):
        alms = noise_alms[i*3:(i+1)*3]
        final_alms[ms] = alms

    return final_alms


def generate_noise_maps_from_alms(final_alms, map_sets_list, nside):
    """
    """
    noise_maps = {
        ms: hp.alm2map(final_alms[ms], nside=nside)
        for ms in map_sets_list
    }
    return noise_maps


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    nl_dir = f"{out_dir}/noise_interp"

    noise_sims_dir = f"{out_dir}/noise_sims"
    BBmeta.make_dir(noise_sims_dir)

    # Load noise power spectra
    nl_dict = {}
    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        nl_file = f"{nl_dir}/nl_{ms1}_x_{ms2}.npz"
        nl = np.load(nl_file)
        nl_dict[ms1, ms2] = {k: nl[k] for k in nl.keys()}

    cls = build_noise_ps_matrix(nl_dict, meta.map_sets_list)

    for id_sim in range(meta.covariance["cov_num_sims"]):
        for id_bundle in range(4):
            noise_alms = generate_noise_alms_from_cls(cls, meta.map_sets_list)
            noise_maps = generate_noise_maps_from_alms(
                noise_alms,
                meta.map_sets_list,
                meta.nside
            )

            for ms in meta.map_sets_list:
                fname = f"homogeneous_noise_{ms}_bundle{id_bundle}_{id_sim:04d}.fits" # noqa
                mu.write_map(f"{noise_sims_dir}/{fname}", noise_maps[ms],
                             dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    args = parser.parse_args()
    main(args)
