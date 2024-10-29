import argparse
from soopercool import BBmeta
import healpy as hp
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
import numpy as np


def get_noise_cls(noise_kwargs, lmax, freq_tag, fsky=0.1,
                  is_beam_deconvolved=False):
    """
    Load polarization noise from SO SAT noise model.
    Assume polarization noise is half of that.
    """
    oof_dict = {"pessimistic": 0, "optimistic": 1}
    oof_mode = noise_kwargs["one_over_f_mode"]
    oof_mode = oof_dict[oof_mode]

    sensitivity_mode = noise_kwargs["sensitivity_mode"]

    noise_model = noise_calc.SOSatV3point1(
        sensitivity_mode=sensitivity_mode,
        one_over_f_mode=oof_mode,
        survey_years=noise_kwargs["survey_years"]
    )
    lth, _, nlth_P = noise_model.get_noise_curves(
        fsky,
        lmax + 1,
        delta_ell=1,
        deconv_beam=is_beam_deconvolved
    )
    lth = np.concatenate(([0, 1], lth))[:]
    nlth_P = np.array(
        [np.concatenate(([0, 0], nl))[:] for nl in nlth_P]
    )
    freq_tags = {int(f): i_f
                 for (i_f, f) in enumerate(noise_model.get_bands())}

    if freq_tag not in freq_tags:
        raise ValueError(f"{freq_tag} GHz is not an SO-SAT frequency")

    idx_f = freq_tags[freq_tag]
    nl_th_dict = {pq: nlth_P[idx_f]
                  for pq in ["EE", "EB", "BE", "BB"]}
    nl_th_dict["TT"] = 0.5*nlth_P[idx_f]
    nl_th_dict["TE"] = 0.*nlth_P[idx_f]
    nl_th_dict["TB"] = 0.*nlth_P[idx_f]

    return lth, nl_th_dict


def generate_noise_map(nl_T, nl_P, n_bundles, nside, seed):
    """
    """
    noise_mat = np.array([nl_T, nl_P, nl_P, np.zeros_like(nl_P)])
    noise_mat *= n_bundles

    np.random.seed(seed)
    return hp.synfast(noise_mat, nside, pol=True, new=True)


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose
    out_dir = meta.output_directory

    sims_dir = f"{out_dir}/noise_sims"
    BBmeta.make_dir(sims_dir)

    lmax_sim = 3*meta.nside - 1

    noise_kwargs = {
        "one_over_f_mode": "optimistic",
        "sensitivity_mode": "baseline",
        "survey_years": 5,
    }
    nlth = {ms: get_noise_cls(noise_kwargs, lmax_sim,
                              meta.freq_tag_from_map_set(ms))
            for ms in meta.map_sets_list}
    nlth["l"] = np.arange(lmax_sim + 1)

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):
        BBmeta.make_dir(f"{sims_dir}/{id_sim:04d}")
        for ms in meta.map_sets_list:
            if verbose:
                print(f"# {id_sim+1} | {ms}")
            n_bundles = meta.n_bundles_from_map_set(ms)
            for id_bundle in range(n_bundles):
                seed = id_sim*n_bundles + id_bundle + 4582
                noise = generate_noise_map(
                    nlth[ms][1]["TT"], nlth[ms][1]["EE"],
                    n_bundles, meta.nside, seed
                )
                mu.write_map(
                    f"{sims_dir}/{id_sim:04d}/noise_sims_{ms}_{id_sim:04d}_bundle{id_bundle}.fits",  # noqa
                    noise, dtype=np.float32, convert_muK_to_K=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
