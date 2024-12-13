import argparse
from soopercool import BBmeta
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import numpy as np


def beam_gaussian(ll, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.
    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.
    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ll * (ll + 1) * sigma_rad**2)


def get_sat_beam(freq_tag, lmax):
    """
    Compute and save dictionary with beam window functions for each map set.
    """
    noise_model = noise_calc.SOSatV3point1(
        survey_years=5, sensitivity_mode="baseline", one_over_f_mode=1
    )
    freq_tags = {int(f): i_f for (i_f, f) in enumerate(noise_model.get_bands())}

    if freq_tag not in freq_tags:
        raise ValueError(f"{freq_tag} GHz is not an SO-SAT frequency")

    ll = np.arange(lmax + 1)
    beams_list = {
        int(freq_band): beam_arcmin
        for freq_band, beam_arcmin in zip(
            noise_model.get_bands(), noise_model.get_beams()
        )
    }
    return ll, beam_gaussian(ll, beams_list[freq_tag])


def main(args):
    """ """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    lmax_sim = 3000

    for ms in meta.map_sets_list:
        freq_tag = meta.freq_tag_from_map_set(ms)

        out_dir = meta.output_directory
        beam_dir = f"{out_dir}/gaussian_beams"
        BBmeta.make_dir(beam_dir)

        fname = f"{beam_dir}/beam_{ms}.dat"

        if verbose:
            print(f"Written to {fname}.")
        l, bl = get_sat_beam(freq_tag, lmax_sim)
        np.savetxt(fname, np.transpose([l, bl]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
