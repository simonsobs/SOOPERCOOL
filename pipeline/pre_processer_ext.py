import argparse
from soopercool import BBmeta, utils
import numpy as np
import os
import healpy as hp
import tarfile
import shutil
import wget
import astropy.io.fits as fits
import matplotlib.pyplot as plt


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
        - Beams, temperature and polarization, per freq and split
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

                if args.plots:
                    plt.plot(bl_T, label=r'$b_\ell$ (T)')
                    plt.plot(bl_pol, label=r'$b_\ell$ (E)')
                    plt.title(f"{f} GHz")
                    plt.legend(frameon=False)
                    plt.savefig(f"{plots_dir}/beam_{file_root}.png",
                                bbox_inches='tight')
                    # plt.show()
                    plt.clf()

        if args.plots:
            for map_set in meta.map_sets_list:
                if 'planck' in map_set:
                    plt.plot(beams[map_set], label=str(map_set))
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$B_\ell$')
            plt.title("Planck beams")
            plt.legend(frameon=False)
            plt.savefig(f"{plots_dir}/beams_planck.png")
            plt.clf()

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

                if args.plots:
                    plt.plot(bl_T, label=r'$b_\ell$ (T)')
                    plt.plot(bl_pol, label=r'$b_\ell$ (E)')
                    plt.title(f"{f}{split}")
                    plt.legend(frameon=False)
                    plt.savefig(f"{plots_dir}/beam_f{f}_{split}.png",
                                bbox_inches='tight')
                    # plt.show()
                    plt.clf()

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
        url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/1yr/raw"
        # Download WMAP single-year maps from URL and save temporarily
        print("Downloading WMAP single-year maps")
        for map_set in meta.map_sets_list:
            if 'wmap' in map_set:
                f = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
                freqs.append(f)
                nsplits = meta.n_splits_from_map_set(map_set)
                for yr in range(1, nsplits+1):
                    fname_map = f"wmap_iqumap_r9_yr{yr}_{bands_dict[f]}_v5.fits"  # noqa
                    print("-", fname_map)
                    fname_data_map = f"{map_dir}/wmap_f{f}_split_{yr-1}.fits"
                    if os.path.isfile(fname_data_map):
                        print("data maps already on disk")
                        continue
                    url = f"{url_pref}/{fname_map}"
                    fname_out = f"{wmap_dir}/{fname_map}"
                    if os.path.isfile(fname_out):
                        print("original data already downloaded:", fname_out)
                        continue
                    wget.download(url, fname_out)
                    print("\n")
        splits = list(range(1, nsplits+1))

        print("Beams")
        print("Downloading WMAP beam window functions")
        ext_dir = f"{prep_dir}/external_data"
        url_pref = "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/beams"

        for f in freqs:
            print(f"wmap_f{f}")
            beam_fname = f"{beam_dir}/beam_wmap_f{f}.dat"
            if os.path.isfile(beam_fname):
                print(f"beams found at {beam_fname}")
                continue
            beam_tf = f"{ext_dir}/wmap_beam_{bands_dict[f]}.txt"
            if not os.path.isfile(beam_tf):
                url = f"{url_pref}/wmap_ampl_bl_{bands_dict[f]}_9yr_v5p1.txt"
                wget.download(url, beam_tf)
                print("\n")
            # read beams
            # WMAP max l is 750, so we extend beams repeating the last value
            bl = np.zeros(3*nside_out+1)
            l, b, _ = np.loadtxt(beam_tf, unpack=True)
            lmax_file = int(l[-1])
            bl[:(lmax_file+1)] = b
            bl[lmax_file:] = b[-1]
            np.savetxt(beam_fname, np.transpose([ells_beam, bl]))

    if args.data:
        # rotating and downgrading data maps
        # both operations carried out in harmonic space (alms)
        # to conserve harmonic properties
        print("---------------------")
        print("Rotating/downgrading external data maps")
        meta.timer.start("Rotate/downgrade external data maps")
        angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])
        for nu in freqs:
            if args.planck:
                # original nside
                nside_in = 1024 if nu == '030' else 2048
            elif args.wmap:
                nside_in = 512

            if nside_in != nside_out:  # if downgrading
                # which alms indices to clip before downgrading
                lmax_in = 3*nside_in - 1
                clipping_indices = []
                for m in range(lmax_out+1):
                    clipping_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m))  # noqa
                clipping_indices = np.concatenate(clipping_indices)

            for split in splits:
                print("---------------------")
                print(f"- channel {nu} - split {split}")
                if args.planck:
                    fname_in = f"{npipe_dir}/npipe6v20{split}/npipe6v20{split}_{nu}_map.fits"  # noqa
                    fname_alms = f"{alms_dir}/npipe6v20{split}/alms_npipe6v20{split}_{nu}_map_ns{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                    os.makedirs(f"{alms_dir}/npipe6v20{split}",
                                exist_ok=True)
                elif args.wmap:
                    fname_in = f"{wmap_dir}/wmap_iqumap_r9_yr{split}_{bands_dict[nu]}_v5.fits"  # noqa
                    fname_alms = f"{alms_dir}/wmap_{bands_dict[nu]}/alms_wmap_r9_yr{split}_{bands_dict[nu]}_map_{nside_in}.npz"  # noqa
                    fname_out = f"{map_dir}/wmap_f{nu}_split_{split-1}.fits"
                    os.makedirs(f"{alms_dir}/wmap_{bands_dict[nu]}",
                                exist_ok=True)

                if os.path.isfile(fname_out):
                    print("map already rotated/downgraded, found at",
                          fname_out)
                    continue
                if os.path.isfile(fname_alms):
                    print("reading alms at", fname_alms)
                    alm_file = np.load(fname_alms)
                    alm = np.array([alm_file['alm_T'],
                                    alm_file['alm_E'],
                                    alm_file['alm_B']])
                else:
                    print("reading input maps at", fname_in)
                    # TQU, -> muK
                    m = to_muk * hp.read_map(fname_in, field=[0, 1, 2])
                    if args.planck:
                        print("removing dipole")
                        m[0] -= dipole_template[nside_in]

                    print("computing alms")
                    alm = hp.map2alm(m, pol=True, use_pixel_weights=True)
                    print("writing alms to disk")
                    np.savez(fname_alms, alm_T=alm[0],
                             alm_E=alm[1], alm_B=alm[2])
                    del m

                if nside_in != nside_out:
                    print("clipping alms")
                    # clipping alms at l>lmax_out to avoid artifacts
                    alm = [each[clipping_indices] for each in alm]

                print("rotating alms from galactic to equatorial coords")
                hp.rotate_alm(alm, *angles)

                print("projecting alms to map")
                if nside_in != nside_out:
                    print(f"and downgrading to nside={nside_out}")
                m_out = hp.alm2map(alm, nside=nside_out, pol=True)

                # mask external maps with SO binary mask
                # m_out *= binary_mask

                print("writing maps to disk")
                hp.write_map(fname_out, m_out,
                             overwrite=True, dtype=np.float32)

                if args.plots:
                    print("plotting maps")
                    if args.planck:
                        utils.plot_map(m_out,
                                       f"{map_plots_dir}/map_planck_f{nu}_split_{splits_dict[split]}",  # noqa
                                       title=f'planck_f{nu}', TQU=True)
                    elif args.wmap:
                        utils.plot_map(m_out,
                                       f"{map_plots_dir}/map_wmap_f{nu}_split_{split-1}",  # noqa
                                       title=f'wmap_f{nu}', TQU=True)
        if 'm_out' in locals():
            del m_out
        if 'alm' in locals():
            del alm
        meta.timer.stop("Rotate/downgrade external data maps")

    if args.sims or args.noise:
        print("---------------------")
        print("Processing simulations")
        meta.timer.start("Process simulations")
        nsims = meta.num_sims
        # path to already downgraded and rotated npipe sims
        if args.planck:
            ext_sims_dir = f"/global/cfs/cdirs/sobs/users/cranucci/npipe/npipe_nside{nside_out}_coords_eq"  # noqa
        elif args.wmap:
            ext_sims_dir = f"/global/cfs/cdirs/sobs/users/cranucci/wmap/noise_sims_nside{nside_out}_coords_eq"  # noqa
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
                if args.planck:
                    noise_dir = f"{sims_dir}/{sim_id:04d}/npipe_residual"
                elif args.wmap:
                    noise_dir = f"{sims_dir}/{sim_id:04d}/wmap_noise"
                os.makedirs(noise_dir, exist_ok=True)
            else:
                os.makedirs(f"{sims_dir}/{sim_id:04d}", exist_ok=True)

            for nu in freqs:
                nside_in = 1024 if nu == '030' else 2048  # original nside

                if process_sims:
                    # which alms indices to clip before downgrading
                    lmax_in = 3*nside_in - 1
                    clipping_indices = []
                    for m in range(lmax_out+1):
                        clipping_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m)) # noqa
                    clipping_indices = np.concatenate(clipping_indices)

                for split in splits:
                    print("---------------------")
                    print(f"-sim {sim_id:04d} - channel {nu} - split {split}")
                    if args.noise:
                        if args.planck:
                            fname_in = f"{ext_sims_dir}/npipe6v20{split}_sim/{sim_id_in:04d}/residual/residual_npipe6v20{split}_{nu}_{sim_id_in:04d}.fits"  # noqa
                            fname_out = f"{noise_dir}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                        elif args.wmap:
                            fname_in = f"{ext_sims_dir}/{sim_id_in:04d}/wmap_noise_mK_bandK_yr{split-1}_nside{nside_out}.fits"  # noqa
                            fname_out = f"{noise_dir}/wmap_f{nu}_split_{split-1}.fits"  # noqa
                    else:
                        fname_in = f"{ext_sims_dir}/npipe6v20{split}_sim/{sim_id_in:04d}/npipe6v20{split}_{nu}_map.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id:04d}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa

                    if os.path.isfile(fname_out):
                        print("sims already rotated/downgraded, found at",
                              fname_out)
                        continue

                    print("reading input maps at", fname_in)
                    m = to_muk * hp.read_map(fname_in, field=[0, 1, 2])

                    if process_sims:
                        if not args.noise:
                            # Subtract the dipole (not from noise)
                            m[0] -= dipole_template[nside_in]
                        print("computing alms")
                        alm = hp.map2alm(m, pol=True, use_pixel_weights=True)
                        print("clipping alms")
                        # clipping alms at l>lmax_out to avoid artifacts
                        alm_clipped = [each[clipping_indices] for each in alm]
                        print("rotate alms, galactic to equatorial coords")
                        hp.rotate_alm(alm_clipped, *angles)
                        print(f"downgrade to nside={nside_out}")
                        m = hp.alm2map(alm_clipped, nside=nside_out, pol=True)

                    print("masking")
                    m *= binary_mask
                    print("writing maps to disk")
                    hp.write_map(fname_out, m,
                                 overwrite=True, dtype=np.float32)
        meta.timer.stop("Process simulations")
        print("---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing Planck NPIPE maps for the pipeline")
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

    if args.wmap and (args.sims or args.noise):
        print("WMAP simulations not available yet.")
        print("exiting...")
    else:
        pre_processer(args)
