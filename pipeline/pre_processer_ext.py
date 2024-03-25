import argparse
from soopercool import BBmeta, utils
import numpy as np
import os, sys
import healpy as hp


def pre_processer(args):
    """
    Pre-process Planck data to be compatible with SOOPERCOOL.
    3 main operations:
        - compute alms of original Planck maps (expensive part)
        - rotate alms from galactic to equatorial coords
        - downgrade maps to nside defined in paramfile
    Planck sims are read (already downgraded and rotated) from NERSC.
    If not found, they will be processed starting
    from original maps as done with the data
    ---------------------------------------------------------
    output:
        - Planck beams (for now, gaussian beams from FWHM)
        - Planck downgraded data maps, rotated in equatorial coords
        - (optional) Planck sims
    """

    print("Pre-processing Planck NPIPE data")
    meta = BBmeta(args.globals)

    npipe_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe"
    data_dir = "../data"
    sims_dir = meta.sims_directory
    map_dir = meta.map_directory
    beam_dir = meta.beam_directory
    prep_dir = meta.pre_process_directory
    alms_dir = prep_dir + "/planck/alms"
    plots_dir = prep_dir + "/planck/plots"
    mask_dir = meta.mask_directory
    directories = [data_dir, sims_dir, map_dir,
                   beam_dir, prep_dir, alms_dir,
                   plots_dir, mask_dir]
    for dirs in directories:
        os.makedirs(dirs, exist_ok=True)

    binary_mask = meta.read_mask("binary")

    freqs = []  # ['030', '100', '143', '217', '353']
    nside_out = meta.nside
    lmax_out = 3*nside_out - 1

    print("Beams")
    ells = np.arange(0, 3*nside_out+1, 1)
    beam_arcmin = {'023': 0., '030': 32.34, '100': 9.66,
                   '143': 7.27, '217': 5.01, '353': 4.86}
    beams = {}

    for map_set in meta.map_sets_list:
        if 'planck' in map_set:
            freq_tag = str(meta.freq_tag_from_map_set(map_set)).zfill(3)
            freqs.append(freq_tag)
            beams[map_set] = utils.beam_gaussian(ells, beam_arcmin[freq_tag])
            # Save beams
            file_root = meta.file_root_from_map_set(map_set)
            if not os.path.exists(file_root):
                np.savetxt(f"{beam_dir}/beam_{file_root}.dat",
                           np.transpose([ells, beams[map_set]]))
    if args.plots:
        import matplotlib.pyplot as plt
        for map_set in meta.map_sets_list:
            if 'planck' in map_set:
                plt.plot(ells, beams[map_set], label=str(map_set))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$B_\ell$')
        plt.title('Planck beams')
        plt.legend(frameon=False)
        plt.savefig(f'{plots_dir}/beams_planck.png')
        plt.clf()

    splits = ['A', 'B']
    splits_dict = {'A': 0, 'B': 1}

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
        print("Computing alms of original maps")
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
                m = 1e6 * hp.read_map(fname_in, field=[0, 1, 2])  # TQU, K to muK
                print("removing dipole")
                m[0] -= dipole_template[nside_in]

                print(f"computing alms, input map has nside={nside_in}")
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
                    print("maps already rotated/downgraded, found at", fname_out)
                    continue

                print("reading alms at", fname_in)
                alm_file = np.load(fname_in)
                alm = np.array([alm_file['alm_T'], alm_file['alm_E'], alm_file['alm_B']])  # noqa

                print("clipping alms")
                # clipping alms at l>lmax_out to avoid artifacts
                alm_clipped = [each[clipping_indices] for each in alm]

                print("rotating alms from galactic to equatorial coords")
                hp.rotate_alm(alm_clipped, *angles)

                print(f"downgrading to nside={nside_out}")
                m_dg_rot = hp.alm2map(alm_clipped, nside=nside_out, pol=True)

                # mask Planck map with SO binary mask
                m_masked = binary_mask * m_dg_rot

                print("writing maps to disk")
                hp.write_map(fname_out, m_masked,
                             overwrite=True, dtype=np.float32)

                if args.plots:
                    print("plotting maps")
                    utils.plot_map(m_masked,
                                   f"{plots_dir}/map_planck_f{nu}_split_{splits_dict[split]}", # noqa
                                   title=f'planck_f{nu}', TQU=True)
                    plt.close()
        meta.timer.stop("Rotate and downgrade data maps")

    if args.sims or args.noise:
        print("---------------------")
        print("Processing simulations" if not args.noise else "Processing noise simulations")  # noqa
        meta.timer.start("Process simulations")
        # path to already downgraded and rotated npipe sims
        npipe_sims_dir = f"/global/cfs/cdirs/sobs/users/cranucci/npipe/npipe_nside{nside_out}_coords_eq"  # noqa
        if os.path.isdir(npipe_sims_dir):
            process_sims = False
            sim_ids = [str(s).zfill(4) for s in range(meta.num_sims)]
        else:
            # process original npipe sims maps
            process_sims = True
            # path to original maps
            npipe_sims_dir = npipe_dir
            # npipe sims starting seed is 0200
            sim_ids = [str(s).zfill(4) for s in range(200, 200+meta.num_sims)]

        for sim_id in sim_ids:
            # starting seed from 0200 to 0000 (if processing original maps)
            sim_id_out = str(int(sim_id) - 200).zfill(4) if process_sims else sim_id  # noqa
            if args.noise:
                os.makedirs(f"{sims_dir}/{sim_id_out}/npipe_residual",
                            exist_ok=True)
            else:
                os.makedirs(sims_dir + f"/{sim_id_out}", exist_ok=True)
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
                    print(f"sim {sim_id} - channel {nu} - split {split}")
                    if args.noise:
                        fname_in = f"{npipe_sims_dir}/npipe6v20{split}_sim/{sim_id}/residual/residual_npipe6v20{split}_{nu}_{sim_id}.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id_out}/npipe_residual/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa
                    else:
                        fname_in = f"{npipe_sims_dir}/npipe6v20{split}_sim/{sim_id}/npipe6v20{split}_{nu}_map.fits"  # noqa
                        fname_out = f"{sims_dir}/{sim_id_out}/planck_f{nu}_split_{splits_dict[split]}.fits"  # noqa

                    if os.path.isfile(fname_out):
                        print("sims already rotated/downgraded, found at",
                              fname_out)
                        continue

                    print("reading input maps at", fname_in)
                    m = 1e6 * hp.read_map(fname_in, field=[0, 1, 2])

                    if process_sims:
                        if not args.noise:
                            # Subtract the dipole
                            m[0] -= dipole_template[nside_in]
                        print("computing alms")
                        alm = hp.map2alm(m, pol=True, use_pixel_weights=True)
                        print("clip alms")
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
        del(dipole_template)
        meta.timer.stop("Process simulations")
        print("---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processing Planck NPIPE maps for the pipeline")
    parser.add_argument("--globals", type=str,
                        help="Path to the yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Pass to generate plots")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--noise", action="store_true")

    args = parser.parse_args()
    pre_processer(args)
