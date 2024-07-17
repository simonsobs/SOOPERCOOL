import argparse
from soopercool import BBmeta, utils
from soopercool import map_utils as mu
import numpy as np
import os
import healpy as hp
import tarfile
import shutil
import wget
import astropy.io.fits as fits


def pre_processer(args):
    """
    Pre-process external data (Planck and WMAP)
    to be compatible with SOOPERCOOL.
    Main operations:
        - compute alms of original maps (expensive part)
        - rotate alms from galactic to equatorial coords
        - downgrade maps to nside defined in paramfile
    Simulations are read (already downgraded and rotated) from NERSC.
    If not found, they will be processed starting
    from original maps as done with the data.
    ----------
    Runs in three modes: "data", "sims", "noise".
    * "data": operate on real data
    * "sims": operate on simulations
    * "noise": operate on noise simulations
    ----------
    Parameters
    ----------
    args: dictionary
        Global parameters and command line arguments.
    ----------
    outputs:
        - Beams, temperature (both) and polarization (Planck only)
        - Spherical armonics transforms (alm) of data maps
        - Downgraded data maps, rotated in equatorial coords
        - Downgraded sims/noise maps, rotated in equatorial coords
    """

    meta = BBmeta(args.globals)

    data_dir = "../data"
    outputs_dir = "../outputs"
    sims_dir = meta.sims_directory
    map_dir = meta.map_directory
    map_plots_dir = f"{outputs_dir}/plots/maps"
    beam_dir = meta.beam_directory
    prep_dir = meta.pre_process_directory
    alms_dir = f"{prep_dir}/external_data/alms"
    plots_dir = f"{prep_dir}/external_data/plots"
    ext_maps_dir = f"{prep_dir}/external_data/maps"
    mask_dir = meta.mask_directory
    directories = [data_dir, outputs_dir, sims_dir,
                   map_dir, map_plots_dir, beam_dir,
                   prep_dir, alms_dir, ext_maps_dir,
                   plots_dir, mask_dir]
    for dirs in directories:
        os.makedirs(dirs, exist_ok=True)

    freqs = []
    binary_mask = meta.read_mask("binary")
    nside_out = meta.nside
    lmax_out = 3*nside_out - 1
    ells_beam = np.arange(3*nside_out + 1)

    if args.planck:
        npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"
        splits = ['A', 'B']
        splits_dict = {'A': 0, 'B': 1}
        to_muk = 1e6

        print("Beams")
        beam_windows_dir = f"{beam_dir}/simulated_maps/npipe_aux/beam_window_functions"  # noqa
        # Download Planck NPIPE RIMO from URL and save temporarily
        url_pref = "https://portal.nersc.gov/cfs/cmb/planck2020/misc"
        url = f"{url_pref}/PLANCK_RIMO_TF_R4.00.tar.gz"
        fname_rimo = f"{prep_dir}/external_data/PLANCK_RIMO_TF_R4.00.tar.gz"
        if not os.path.isfile(fname_rimo):
            print("Downloading Planck RIMO...")
            wget.download(url, fname_rimo)
            print("\n")
        tf = tarfile.open(fname_rimo)
        tf.extractall(beam_dir)
        tf.close()

        spectra = ["TT", "EE", "BB", "TE"]
        leakage_term = {}
        for spec in spectra:
            leakage_term[spec] = [f"{spec}_2_TT", f"{spec}_2_EE",
                                  f"{spec}_2_BB", f"{spec}_2_TE",
                                  f"{spec}_2_TB", f"{spec}_2_EB",
                                  f"{spec}_2_ET", f"{spec}_2_BT",
                                  f"{spec}_2_BE"]

        print("- frequency-only dependent beams")
        beams = {}
        for map_set in meta.map_sets_list:
            if 'planck' in map_set:
                print(map_set)
                f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
                freqs.append(f)
                file_root = meta.file_root_from_map_set(map_set)
                beam_fname = f"{beam_dir}/beam_{file_root}.dat"
                beam_fname_T = f"{beam_dir}/beam_T_{file_root}.dat"
                beam_fname_pol = f"{beam_dir}/beam_pol_{file_root}.dat"

                if os.path.isfile(beam_fname):
                    print("beams found in", beam_dir)
                    _, beams[map_set] = np.loadtxt(beam_fname, unpack=True)
                    continue

                wl = fits.open(f"{beam_windows_dir}/full_frequency/Wl_npipe6v20_{f}GHzx{f}GHz.fits")  # noqa
                wl_dict = {}
                num = 1
                for spec in spectra:
                    for leak in leakage_term[spec]:
                        wl_dict[leak] = wl[num].data[leak]
                    num += 1

                # Planck beam: sqrt of XX_2_XX term of the beam leakage matrix
                bl_T = np.sqrt(wl_dict["TT_2_TT"][0])[:(3*nside_out+1)]
                bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])[:(3*nside_out+1)]
                beams[map_set] = bl_pol

                # Save beams
                np.savetxt(beam_fname_T, np.transpose([ells_beam, bl_T]))
                np.savetxt(beam_fname_pol, np.transpose([ells_beam, bl_pol]))
                np.savetxt(beam_fname, np.transpose([ells_beam, bl_pol]))

        print("- frequency-split dependent beams")
        for split in splits:
            for f in freqs:
                print(f"planck_f{f}{split}")
                beam_fname_T = f"{beam_dir}/beam_T_planck_{f}{split}.dat"
                beam_fname_pol = f"{beam_dir}/beam_pol_planck_{f}{split}.dat"

                if os.path.isfile(beam_fname_pol):
                    print("beams found in", beam_dir)
                    continue
                wl = fits.open(f"{beam_windows_dir}/AB/Wl_npipe6v20_{f}{split}x{f}{split}.fits")  # noqa
                wl_dict = {}
                num = 1
                for spec in spectra:
                    for leak in leakage_term[spec]:
                        wl_dict[leak] = wl[num].data[leak]
                    num += 1

                # Planck beam: sqrt of XX_2_XX term of the beam leakage matrix
                bl_T = np.sqrt(wl_dict["TT_2_TT"][0])[:(3*nside_out+1)]
                bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])[:(3*nside_out+1)]

                np.savetxt(beam_fname_T, np.transpose([ells_beam, bl_T]))
                np.savetxt(beam_fname_pol, np.transpose([ells_beam, bl_pol]))

        if os.path.isdir(f"{beam_dir}/simulated_maps"):
            shutil.rmtree(f"{beam_dir}/simulated_maps")

        if not args.noise:
            print("---------------------")
            print("Generating dipole template")
            # dipole amplitude, Galactic coordinates of the dipole direction
            # from NPIPE paper, Solar dipole measurements, Table 11
            dipole_amplitude = 3366.6  # [muK_cmb]
            dipole_longitude = np.radians(263.986)  # l [deg]
            dipole_latitude = np.radians(48.247)  # b [deg]
            # Convert Galactic coordinates to Cartesian coordinates
            x = np.cos(dipole_latitude) * np.cos(dipole_longitude)
            y = np.cos(dipole_latitude) * np.sin(dipole_longitude)
            z = np.sin(dipole_latitude)
            # Generate a dipole template (for the two possible nside)
            dipole_template = {}
            dipole_template[1024] = dipole_amplitude * np.dot([x, y, z], hp.pix2vec(1024, np.arange(hp.nside2npix(1024))))  # noqa
            dipole_template[2048] = dipole_amplitude * np.dot([x, y, z], hp.pix2vec(2048, np.arange(hp.nside2npix(2048))))  # noqa

    elif args.wmap:
        to_muk = 1e3
        bands_dict = {'023': 'K1', '033': 'Ka1'}
        wmap_dir = f"{prep_dir}/external_data/maps"

        for map_set in meta.map_sets_list:
            if 'wmap' in map_set:
                f = meta.freq_tag_from_map_set(map_set)
                freqs.append(str(f).zfill(3))
                nsplits = meta.n_splits_from_map_set(map_set)
        splits = list(range(1, nsplits+1))

        if args.data:
            meta.timer.start("Download WMAP data")
            url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/1yr/raw"  # noqa
            # Download WMAP single-year maps from URL and store
            print("Downloading WMAP single-year maps")
            for f in freqs:
                for yr in splits:
                    fname_map = f"wmap_iqumap_r9_yr{yr}_{bands_dict[f]}_v5.fits"  # noqa
                    print("-", fname_map)
                    fname_data_map = f"{map_dir}/wmap_f{f}_split_{yr-1}.fits"
                    if os.path.isfile(fname_data_map):
                        print("  data maps already on disk")
                        continue
                    url = f"{url_pref}/{fname_map}"
                    fname_out = f"{wmap_dir}/{fname_map}"
                    if os.path.isfile(fname_out):
                        print("  data already downloaded, at", wmap_dir)
                        continue
                    wget.download(url, fname_out)
                    print("\n")

            print("Mask")
            # Download WMAP temperature analysis mask from URL and store
            url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/masks"  # noqa
            url = f"{url_pref}/wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits"  # noqa
            fname_mask = f"{mask_dir}/wmap_temperature_analysis_mask.fits"
            if not os.path.isfile(fname_mask):
                wget.download(url, fname_mask)
                print("\n")
            mask = mu.read_map(fname_mask, field=['N_OBS'])

            print("Beams")
            print("Downloading WMAP beam window functions")
            ext_dir = f"{prep_dir}/external_data"
            url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/beams"  # noqa

            for f in freqs:
                print(f"wmap_f{f}")
                beam_fname = f"{beam_dir}/beam_wmap_f{f}.dat"
                if os.path.isfile(beam_fname):
                    print(f"beams found at {beam_fname}")
                    continue
                beam_tf = f"{ext_dir}/wmap_beam_{bands_dict[f]}.txt"
                if not os.path.isfile(beam_tf):
                    url = f"{url_pref}/wmap_ampl_bl_{bands_dict[f]}_9yr_v5p1.txt"  # noqa
                    wget.download(url, beam_tf)
                    print("\n")
                # read beams
                # if last ell < 3*nside_out
                # extend beams repeating the last value
                bl = np.zeros(3*nside_out+1)
                l, b, _ = np.loadtxt(beam_tf, unpack=True)
                lmax_file = int(l[-1])
                if lmax_file < 3*nside_out:
                    bl[:(lmax_file+1)] = b
                    bl[lmax_file:] = b[-1]
                else:
                    bl = b[:(3*nside_out+1)]
                np.savetxt(beam_fname, np.transpose([ells_beam, bl]))
            meta.timer.stop("Download WMAP data")

    if args.data:
        # rotating and downgrading data maps
        # both operations carried out in harmonic space (alms)
        # to conserve harmonic properties
        print("---------------------")
        print("Processing external maps")
        meta.timer.start("Process external maps")
        angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])
        for nu in freqs:
            if args.planck:
                # original nside
                nside_in = 1024 if nu == '030' else 2048
            elif args.wmap:
                nside_in = 512

            dgrade = (nside_in != nside_out)
            if dgrade:  # if downgrading
                # which alms indices to clip before downgrading
                lmax_in = 3*nside_in - 1
                clip_indices = []
                for m in range(lmax_out+1):
                    clip_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m))  # noqa
                clip_indices = np.concatenate(clip_indices)

            for split in splits:
                print("---------------------")
                print(f"- channel {nu} - split {split}")
                if args.planck:
                    fname_in = f"{npipe_dir}/npipe6v20{split}/npipe6v20{split}_{nu}_map.fits"  # noqa
                    fname_alms = f"{alms_dir}/npipe6v20{split}/alms_npipe6v20{split}_{nu}_map_ns{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                    os.makedirs(f"{alms_dir}/npipe6v20{split}", exist_ok=True)
                elif args.wmap:
                    fname_in = f"{wmap_dir}/wmap_iqumap_r9_yr{split}_{bands_dict[nu]}_v5.fits"  # noqa
                    fname_alms = f"{alms_dir}/wmap_{bands_dict[nu]}/alms_wmap_{bands_dict[nu]}_yr{split}_map_{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/wmap_f{nu}_split_{split-1}.fits"
                    os.makedirs(f"{alms_dir}/wmap_{bands_dict[nu]}",
                                exist_ok=True)

                if os.path.isfile(fname_out):
                    print("maps already processed, found in", map_dir)
                    continue
                if os.path.isfile(fname_alms):
                    print("reading alms at", fname_alms)
                    alm_file = np.load(fname_alms)
                    alm = np.array([alm_file['alm_T'],
                                    alm_file['alm_E'],
                                    alm_file['alm_B']])
                else:
                    print("reading input maps at", fname_in)
                    # read TQU, convert to muK
                    map_in = to_muk * mu.read_map(fname_in, field=[0, 1, 2])
                    if args.planck:
                        print("removing dipole")
                        map_in[0] -= dipole_template[nside_in]
                    elif args.wmap:
                        # remove mean temperature outside mask
                        map_in_masked = hp.ma(map_in[0])
                        map_in_masked.mask = np.logical_not(mask)
                        map_in[0] -= np.mean(map_in_masked)

                    print("computing alms")
                    alm = hp.map2alm(map_in, pol=True, use_pixel_weights=True)
                    del map_in
                    print("writing alms to disk")
                    np.savez(fname_alms, alm_T=alm[0],
                             alm_E=alm[1], alm_B=alm[2])

                if dgrade:
                    print(f"downgrading to nside={nside_out}")
                    # clipping alms at l>lmax_out to avoid artifacts
                    alm = [each[clip_indices] for each in alm]

                print("rotating alms from galactic to equatorial coords")
                hp.rotate_alm(alm, *angles)

                print("projecting alms to maps")
                map_out = hp.alm2map(alm, nside=nside_out, pol=True)

                # mask external maps with SO binary mask
                map_out *= binary_mask

                print("writing maps to disk")
                mu.write_map(fname_out, map_out, dtype=np.float32)

                if args.plots:
                    print("plotting maps")
                    if args.planck:
                        utils.plot_map(map_out,
                                       f"{map_plots_dir}/map_planck_f{nu}__{splits_dict[split]}",  # noqa
                                       title=f'planck_f{nu}', TQU=True)
                    elif args.wmap:
                        utils.plot_map(map_out,
                                       f"{map_plots_dir}/map_wmap_f{nu}__{split-1}",  # noqa
                                       title=f'wmap_f{nu}', TQU=True)
        if 'm_out' in locals():
            del map_out
        if 'alm' in locals():
            del alm
        meta.timer.stop("Process external maps")

    if args.sims or args.noise:
        print("---------------------")
        print("Processing simulations")
        meta.timer.start("Process simulations")
        nsims = meta.num_sims
        # path to already downgraded and rotated npipe sims
        ext_sims_dir = "/global/cfs/cdirs/sobs/users/cranucci"
        ext_sims_dir += "/npipe" if args.planck else "/wmap"
        ext_sims_dir += f"/nside{nside_out}_coords_eq"

        if os.path.isdir(ext_sims_dir) or args.wmap:
            process_sims = False
        else:
            # process original npipe sims maps
            process_sims = True
            # path to original maps
            ext_sims_dir = npipe_dir

        for sim_id in range(nsims):
            # starting seed from 0200 to 0000 (if processing original maps)
            sim_id_in = sim_id+200 if process_sims else sim_id
            if args.noise:
                # noise output dirs
                noise_dir = f"{sims_dir}/{sim_id:04d}/noise"
                os.makedirs(noise_dir, exist_ok=True)
            else:
                # sims output dirs
                os.makedirs(f"{sims_dir}/{sim_id:04d}", exist_ok=True)

            for nu in freqs:
                nside_in = 1024 if nu == '030' else 2048  # original nside
                dgrade = (nside_in != nside_out)

                if process_sims and dgrade:
                    # which alms indices to clip before downgrading
                    lmax_in = 3*nside_in - 1
                    clip_indices = []
                    for m in range(lmax_out+1):
                        clip_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m)) # noqa
                    clip_indices = np.concatenate(clip_indices)

                for split in splits:
                    print("---------------------")
                    print(f"sim {sim_id:04d} - channel {nu} - split {split}")
                    if args.noise:
                        # noise fnames (Planck and WMAP)
                        if args.planck:
                            fname_in = f"{ext_sims_dir}/npipe6v20{split}_sim"
                            fname_in += f"/{sim_id_in:04d}/residual"
                            fname_in += f"/residual_npipe6v20{split}_{nu}_{sim_id_in:04d}.fits"  # noqa
                            fname_out = f"{noise_dir}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                        elif args.wmap:
                            fname_in = f"{ext_sims_dir}/noise/{sim_id_in:04d}"
                            fname_in += f"/noise_maps_mK_band{bands_dict[nu]}_yr{split}.fits"  # noqa
                            fname_out = f"{noise_dir}/wmap_f{nu}_split_{split-1}.fits"  # noqa
                    else:
                        # sims fnames (only Planck)
                        fname_in = f"{ext_sims_dir}/npipe6v20{split}_sim"
                        fname_in += f"/{sim_id_in:04d}/npipe6v20{split}_{nu}_map.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id:04d}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa

                    if os.path.isfile(fname_out):
                        print("sims already processed, found in", sims_dir)
                        continue

                    print("reading input maps at", fname_in)
                    # read TQU, convert to muK
                    map_in = to_muk * mu.read_map(fname_in, field=[0, 1, 2])

                    if process_sims:
                        if not args.noise:
                            # Subtract the dipole (not from noise)
                            map_in[0] -= dipole_template[nside_in]
                        print("computing alms")
                        alm = hp.map2alm(map_in,
                                         pol=True, use_pixel_weights=True)
                        del map_in

                        if dgrade:
                            print(f"downgrading to nside={nside_out}")
                            # clipping alms at l>lmax_out to avoid artifacts
                            alm = [each[clip_indices] for each in alm]
                        print("rotating alms, galactic -> equatorial")
                        hp.rotate_alm(alm, *angles)
                        print("projecting alms to map")
                        map_out = hp.alm2map(alm, nside=nside_out, pol=True)
                    else:
                        map_out = map_in

                    print("masking")
                    map_out *= binary_mask
                    print("writing maps to disk")
                    mu.write_map(fname_out, map_out, dtype=np.float32)

        meta.timer.stop("Process simulations")
        print("---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing external data for the pipeline")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--planck", action="store_true",
                        help="Pass to process Planck maps")
    parser.add_argument("--wmap", action="store_true",
                        help="Pass to process WMAP maps")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--noise", action="store_true")

    args = parser.parse_args()

    if args.wmap and args.sims:
        print("WMAP simulations not available yet.")
        print("Only noise sims are available.")
        print("exiting...")
    else:
        pre_processer(args)
