import argparse
import healpy as hp
from soopercool.utils import (get_noise_cls, generate_noise_map)
import numpy as np
from soopercool import BBmeta, utils
import warnings
import os
import soopercool.SO_Noise_Calculator_Public_v3_1_2 as noise_calc


def mocker(args):
    """
    Implement a very basic simulation routine
    to generate mock maps or a set of simulation
    to estimate the covariance (if --sims is set to True).

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)

    ps_th = meta.load_fiducial_cl(cl_type="cosmo")

    # Load binary mask
    binary_mask = meta.read_mask("binary")
    fsky = np.mean(binary_mask)
    lmax_sim = 3*meta.nside - 1

    # Create fiducial noise power spectra
    meta.timer.start("Computing noise cls")
    lth, nlth_deconvolved_dict = get_noise_cls(
        noise_kwargs=meta.noise,
        lmax=lmax_sim,
        fsky=fsky,
        is_beam_deconvolved=False
    )

    # Save the noise power spectra
    for ms in meta.map_sets_list:
        ftag = meta.freq_tag_from_map_set(ms)
        np.savez(f"{meta.mock_directory}/noise_{ms}.npz",
                 lth=lth, **nlth_deconvolved_dict[ftag])
    meta.timer.stop("Computing noise cls")

    # Load beams for each map set
    beam_windows = {
        map_set: meta.read_beam(map_set)[1]
        for map_set in meta.map_sets_list
    }

    # Load hitmap
    hitmap = meta.read_hitmap()

    # Load and save beams
    meta.timer.start("Generating beams")

    noise_model = noise_calc.SOSatV3point1()

    beam_arcmin = {freq_band: beam_arcmin
                   for freq_band, beam_arcmin in zip(noise_model.get_bands(),
                                                     noise_model.get_beams())}
    # WMAP and Planck beams in arcmin
    noise_dir = "/global/cfs/cdirs/sobs/users/cranucci/"
    noise_dir_planck = f"{noise_dir}/npipe/nside{meta.nside}_coords_eq"
    noise_dir_wmap = f"{noise_dir}/wmap/nside{meta.nside}_coords_eq/noise"
    splits_dict_planck = {0: 'A', 1: 'B'}
    bands_dict_wmap = {23: 'K1', 33: 'Ka1'}
    # beam_arcmin_ext = {'023': 52.8, '033': 39.6, '030': 32.34, '100': 9.66,
    #                    '143': 7.27, '217': 5.01, '353': 4.86}
    beam_arcmin_ext = {23.: 52.8, 33.: 39.6, 30.: 32.34, 100.: 9.66,
                       143.: 7.27, 217.: 5.01, 353.: 4.86}
    beam_arcmin.update(beam_arcmin_ext)

    beams = {}
    for map_set in meta.map_sets_list:
        freq_tag = meta.freq_tag_from_map_set(map_set)
        file_root = meta.file_root_from_map_set(map_set)
        beam_dir = meta.beam_directory
        beam_fname = f"{beam_dir}/beam_{file_root}.dat"
        # TODO: when refactoring, this part might be unnecessary
        if 'SAT' not in map_set and os.path.isfile(beam_fname):
            print("reading beam", file_root)
            _, beams[map_set] = meta.read_beam(map_set)
        else:
            beams[map_set] = utils.beam_gaussian(lth, beam_arcmin[freq_tag])

            # Save beams
            if not os.path.exists(beam_fname):
                np.savetxt(beam_fname, np.transpose([lth, beams[map_set]]))
    meta.timer.stop("Generating beams")

    hp_ordering = ["TT", "TE", "TB", "EE", "EB", "BB"]

    Nsims = meta.num_sims if args.sims else 1

    for id_sim in range(Nsims):
        alms_T, alms_E, alms_B = hp.synalm([ps_th[k] for k in hp_ordering],
                                           lmax=lmax_sim)
        if meta.null_e_modes:
            cmb_map = hp.alm2map([alms_T, alms_E*0, alms_B],
                                 meta.nside, lmax=lmax_sim)
        else:
            cmb_map = hp.alm2map([alms_T, alms_E, alms_B],
                                 meta.nside, lmax=lmax_sim)

        for map_set in meta.map_sets_list:

            meta.timer.start(f"Generate map set {map_set} split maps")
            cmb_map_beamed = hp.smoothing(
                cmb_map, beam_window=beam_windows[map_set]
            )
            n_splits = meta.n_splits_from_map_set(map_set)
            freq_tag = meta.freq_tag_from_map_set(map_set)

            for id_split in range(n_splits):
                if 'SAT' in map_set:
                    noise_map = generate_noise_map(
                        nlth_deconvolved_dict["T"][freq_tag],
                        nlth_deconvolved_dict["P"][freq_tag],
                        hitmap,
                        n_splits,
                        is_anisotropic=meta.anisotropic_noise
                    )
                else:
                    if 'planck' in map_set:
                        to_muk = 1e6
                        s = splits_dict_planck[id_split]
                        fname_noise = f"{noise_dir_planck}/npipe6v20{s}_sim/{id_sim:04d}/residual"  # noqa
                        fname_noise += f"/residual_npipe6v20{s}_{freq_tag:03d}_{id_sim:04d}.fits"  # noqa
                    elif 'wmap' in map_set:
                        to_muk = 1e3
                        s = id_split + 1
                        fname_noise = f"{noise_dir_wmap}/{id_sim:04d}"
                        fname_noise += f"/noise_maps_mK_band{bands_dict_wmap[freq_tag]}_yr{id_split+1}.fits"  # noqa

                    noise_map = to_muk * hp.read_map(fname_noise,
                                                     field=[0, 1, 2])

                split_map = cmb_map_beamed + noise_map

                split_map *= binary_mask

                map_file_name = meta.get_map_filename(
                    map_set, id_split,
                    id_sim if Nsims > 1 else None
                )
                hp.write_map(
                    map_file_name,
                    split_map,
                    overwrite=True,
                    dtype=np.float32
                )

                if args.plots:
                    if Nsims == 1:
                        plot_dir = meta.plot_dir_from_output_dir(
                            meta.map_directory_rel
                        )
                        utils.plot_map(split_map,
                                       f"{plot_dir}/map_{map_set}__{id_split}",
                                       title=map_set,
                                       TQU=True)
            meta.timer.stop(f"Generate map set {map_set} split maps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--sims", action="store_true",
                        help="Generate a set of sims if True.")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.sims and args.plots:
        warnings.warn("Both --sims and --plot are set to True. "
                      "Too many plots will be generated. "
                      "Set --plot to False")
        args.plots = False

    mocker(args)
