import argparse
from soopercool import BBmeta, utils
import numpy as np
import os
import sys
import healpy as hp
import tarfile
import shutil
import urllib.request
import astropy.io.fits as fits
import matplotlib.pyplot as plt


def pre_processer(args):
    """
    Pre-process external data (Planck and WMAP [not yet])
    to be compatible with SOOPERCOOL.
    3 main operations:
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

    npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"
    data_dir = "../data"
    outputs_dir = "../outputs"
    sims_dir = meta.sims_directory
    map_dir = meta.map_directory
    map_plots_dir = f"{outputs_dir}/plots/maps"
    beam_dir = meta.beam_directory
    prep_dir = meta.pre_process_directory
    alms_dir = f"{prep_dir}/external_data/alms"
    plots_dir = f"{prep_dir}/external_data/plots"
    mask_dir = meta.mask_directory
    directories = [data_dir, outputs_dir, sims_dir,
                   map_dir, map_plots_dir,
                   beam_dir, prep_dir, alms_dir,
                   plots_dir, mask_dir]
    for dirs in directories:
        os.makedirs(dirs, exist_ok=True)

    binary_mask = meta.read_mask("binary")
    nside_out = meta.nside
    lmax_out = 3*nside_out - 1
    ells = np.arange(3*nside_out + 1)

    freqs = []  # ['030', '100', '143', '217', '353']
    splits = ['A', 'B']
    splits_dict = {'A': 0, 'B': 1}

    print("Beams")
    timeout_seconds = 300  # Set the timeout [sec] for the socket
    beam_windows_dir = f"{beam_dir}/simulated_maps/npipe_aux/beam_window_functions"  # noqa
    # Download Planck NPIPE RIMO from URL with a timeout and save temporarily
    urlpref = "https://portal.nersc.gov/cfs/cmb/planck2020/misc"
    url = f"{urlpref}/PLANCK_RIMO_TF_R4.00.tar.gz"
    fname_rimo = f"{prep_dir}/external_data/rimo.tar.gz"

    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        with open(fname_rimo, 'w+b') as f:
            f.write(response.read())

    tf = tarfile.open(fname_rimo)
    tf.extractall(beam_dir)
    tf.close()
    os.remove(fname_rimo)

    spectra = ["TT", "EE", "BB", "TE"]
    leakage_term = {}
    for spec in spectra:
        leakage_term[spec] = [f"{spec}_2_TT", f"{spec}_2_EE", f"{spec}_2_BB",
                              f"{spec}_2_TE", f"{spec}_2_TB", f"{spec}_2_EB",
                              f"{spec}_2_ET", f"{spec}_2_BT", f"{spec}_2_BE"]

    # Frequency-only dependent beams
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

            lmax = len(wl_dict["TT_2_TT"][0])

            bl_T = np.zeros(lmax)
            bl_pol = np.zeros(lmax)

            # Planck beam: sqrt or the XX_2_XX term of the beam leakage matrix
            bl_T = np.sqrt(wl_dict["TT_2_TT"][0])
            bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])
            beams[map_set] = bl_pol

            # Save beams
            np.savetxt(beam_fname_T,
                       np.transpose([ells, bl_T[:(3*nside_out+1)]]))
            np.savetxt(beam_fname_pol,
                       np.transpose([ells, bl_pol[:(3*nside_out+1)]]))
            np.savetxt(beam_fname,
                       np.transpose([ells, bl_pol[:(3*nside_out+1)]]))

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

    # Frequency-split dependent beams
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

            lmax = len(wl_dict["TT_2_TT"][0])

            bl_T = np.zeros(lmax)
            bl_pol = np.zeros(lmax)

            # Planck beam: sqrt or the XX_2_XX term of the beam leakage matrix
            bl_T = np.sqrt(wl_dict["TT_2_TT"][0])
            bl_pol = np.sqrt(wl_dict["EE_2_EE"][0])

            np.savetxt(beam_fname_T,
                       np.transpose([ells, bl_T[:(3*nside_out+1)]]))
            np.savetxt(beam_fname_pol,
                       np.transpose([ells, bl_pol[:(3*nside_out+1)]]))

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

    if args.data:
        # alms of original maps
        print("Computing alms of original data maps")
        meta.timer.start("Compute alms of original data maps")
        for nu in freqs:
            nside_in = 1024 if nu == '030' else 2048  # original maps nside

            for split in splits:
                print("---------------------")
                print(f"channel {nu} - split {split}")
                fname_in = f"{npipe_dir}/npipe6v20{split}/npipe6v20{split}_{nu}_map.fits"  # noqa
                fname_out = f"{alms_dir}/npipe6v20{split}/alms_npipe6v20{split}_{nu}_map_ns{nside_in}.npz"  # noqa
                if os.path.isfile(fname_out):
                    print("alms already computed, found at", fname_out)
                    continue
                os.makedirs(f"{alms_dir}/npipe6v20{split}", exist_ok=True)

                print("reading input maps at", fname_in)
                m = 1e6*hp.read_map(fname_in, field=[0, 1, 2])  # TQU, K->muK
                print("removing dipole")
                m[0] -= dipole_template[nside_in]

                print("computing alms")
                alm = hp.map2alm(m, pol=True, use_pixel_weights=True)
                print("writing alms to disk")
                np.savez(fname_out, alm_T=alm[0], alm_E=alm[1], alm_B=alm[2])
        meta.timer.stop("Compute alms of original data maps")

        # rotating and downgrading
        # both operations carried out in harmonic space (alms)
        # to conserve harmonic properties
        print("---------------------")
        print("Rotating and downgrading data maps")
        meta.timer.start("Rotate and downgrade data maps")
        angles = hp.rotator.coordsys2euler_zyz(coord=["G", "C"])
        for nu in freqs:
            nside_in = 1024 if nu == '030' else 2048  # original alms nside

            # which alms indices to clip before downgrading
            lmax_in = 3*nside_in - 1
            clipping_indices = []
            for m in range(lmax_out+1):
                clipping_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m))  # noqa
            clipping_indices = np.concatenate(clipping_indices)

            for split in splits:
                print("---------------------")
                print(f"channel {nu} - split {split}")
                fname_in = f"{alms_dir}/npipe6v20{split}/alms_npipe6v20{split}_{nu}_map_ns{nside_in}.npz"  # noqa
                fname_out = f"{map_dir}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                if os.path.isfile(fname_out):
                    print("maps already rotated/downgraded, found at")
                    print(fname_out)
                    continue

                print("reading alms at", fname_in)
                alm_file = np.load(fname_in)
                alm = np.array([alm_file['alm_T'],
                                alm_file['alm_E'],
                                alm_file['alm_B']])

                print("clipping alms")
                # clipping alms at l>lmax_out to avoid artifacts
                alm_clipped = [each[clipping_indices] for each in alm]

                print("rotating alms from galactic to equatorial coords")
                hp.rotate_alm(alm_clipped, *angles)

                print(f"downgrading to nside={nside_out}")
                m_dg_rot = hp.alm2map(alm_clipped, nside=nside_out, pol=True)

                # mask Planck maps with SO binary mask
                m_masked = binary_mask * m_dg_rot

                print("writing maps to disk")
                hp.write_map(fname_out, m_masked,
                             overwrite=True, dtype=np.float32)

                if args.plots:
                    print("plotting maps")
                    utils.plot_map(m_masked,
                                   f"{map_plots_dir}/map_planck_f{nu}_split_{splits_dict[split]}", # noqa
                                   title=f'planck_f{nu}', TQU=True)
                    plt.close()
        meta.timer.stop("Rotate and downgrade data maps")

    if args.sims or args.noise:
        print("---------------------")
        print("Processing simulations")
        meta.timer.start("Process simulations")
        nsims = meta.num_sims
        # path to already downgraded and rotated npipe sims
        npipe_sims_dir = f"/global/cfs/cdirs/sobs/users/cranucci/npipe/npipe_nside{nside_out}_coords_eq"  # noqa
        if os.path.isdir(npipe_sims_dir):
            process_sims = False
            # sim_ids = [str(s).zfill(4) for s in range(nsims)]
        else:
            # process original npipe sims maps
            process_sims = True
            # path to original maps
            npipe_sims_dir = npipe_dir
            # npipe sims starting seed is 0200
            # sim_ids = [str(s).zfill(4) for s in range(200, 200+nsims)]

        for sim_id in range(nsims):
            # starting seed from 0200 to 0000 (if processing original maps)
            sim_id_in = sim_id+200 if process_sims else sim_id
            if args.noise:
                os.makedirs(f"{sims_dir}/{sim_id:04d}/npipe_residual",
                            exist_ok=True)
            else:
                os.makedirs(sims_dir + f"/{sim_id:04d}", exist_ok=True)
            for nu in freqs:
                nside_in = 1024 if nu == '030' else 2048  # original maps nside

                # which alms indices to clip before downgrading
                lmax_in = 3*nside_in - 1
                clipping_indices = []
                for m in range(lmax_out+1):
                    clipping_indices.append(hp.Alm.getidx(lmax_in, np.arange(m, lmax_out+1), m)) # noqa
                clipping_indices = np.concatenate(clipping_indices)

                for split in splits:
                    print("---------------------")
                    print(f"sim {sim_id:04d} - channel {nu} - split {split}")
                    if args.noise:
                        fname_in = f"{npipe_sims_dir}/npipe6v20{split}_sim/{sim_id_in:04d}/residual/residual_npipe6v20{split}_{nu}_{sim_id_in:04d}.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id:04d}/npipe_residual/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                    else:
                        fname_in = f"{npipe_sims_dir}/npipe6v20{split}_sim/{sim_id_in:04d}/npipe6v20{split}_{nu}_map.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id:04d}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa

                    if os.path.isfile(fname_out):
                        print("sims already rotated/downgraded, found at",
                              fname_out)
                        continue

                    print("reading input maps at", fname_in)
                    m = 1e6 * hp.read_map(fname_in, field=[0, 1, 2])

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
                    m_masked = binary_mask * m
                    print("writing maps to disk")
                    hp.write_map(fname_out, m_masked,
                                 overwrite=True, dtype=np.float32)
        if not args.noise:
            del (dipole_template)
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
                        help="Pass to process WMAP maps (not yet)")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--noise", action="store_true")

    args = parser.parse_args()

    if args.wmap:
        print("ERROR: WMAP pre-processing not yet implemented")
        print("Please choose another experiment")
        print("exiting...")
        sys.exit(0)

    pre_processer(args)
