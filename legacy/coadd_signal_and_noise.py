import argparse
from soopercool import BBmeta
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
import healpy as hp
import numpy as np
import os


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    do_plots = not args.no_plots

    out_dir = meta.output_directory
    binary_mask_dir = meta.masks["analysis_mask"].replace("analysis",
                                                          "binary")
    assert os.path.isfile(binary_mask_dir), f"Binary mask {binary_mask_dir} doesn't exist."  # noqa

    sims_dir = f"{out_dir}/cov_sims"
    BBmeta.make_dir(sims_dir)

    mu._check_pix_type(meta.pix_type)
    lmax = meta.lmax

    # MPI parallelization
    rank, size, comm = mpi.init(True, logger=None)
    id_start = meta.covariance["cov_id_start"]
    nsims = meta.covariance["cov_num_sims"]
    mpi_shared_list = [(id_sim, ms)
                       for id_sim in range(id_start, nsims+id_start)
                       for ms in meta.map_sets_list]

    use_alms = False
    use_maps = False
    if "signal_alm_sims_dir" in meta.covariance:
        use_alms = meta.covariance["signal_alm_sims_dir"] is not None
    if not use_alms and "signal_map_sims_dir" in meta.covariance:
        use_maps = meta.covariance["signal_map_sims_dir"] is not None
    if use_alms:
        mpi.print_rnk0("Using alms for signal covariance", rank)
        signal_alm_dirs = meta.covariance["signal_alm_sims_dir"]
        signal_alm_templates = meta.covariance["signal_alm_sims_template"]
    if use_maps:
        mpi.print_rnk0("Using maps for signal covariance", rank)
        mpi.print_rnk0("WARNING: Assuming signal map sims "
                       "have been beam convolved.", rank)
        signal_map_dirs = meta.covariance["signal_map_sims_dir"]
        signal_map_templates = meta.covariance["signal_map_sims_template"]
    if not use_alms and not use_maps:
        mpi.print_rnk0("Using noise sims only for covariance", rank)

    binary = mu.read_map(binary_mask_dir,
                         pix_type=meta.pix_type,
                         car_template=meta.car_template)
    beams = {
        ms: meta.read_beam(ms)[1][:lmax+1]
        for ms in meta.map_sets_list
    }

    if do_plots:
        field_pairs = ["TT", "EE", "BB", "TE"]
        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]
        clth = {}

        for ms in meta.map_sets_list:
            nu = meta.freq_tag_from_map_set(ms)
            for i, fp in enumerate(field_pairs):
                clth[ms, fp] = hp.read_cl(fiducial_cmb)[i, :lmax+1]
                if fiducial_dust is not None:
                    clth[ms, fp] += hp.read_cl(
                        fiducial_dust.format(nu1=nu, nu2=nu)
                    )[i, :lmax+1]
                if fiducial_synch is not None:
                    clth[ms, fp] += hp.read_cl(
                        fiducial_synch.format(nu1=nu, nu2=nu)
                    )[i, :lmax+1]
                clth[ms, fp] *= beams[ms]**2

    # Every rank must have the same shared list
    mpi_shared_list = comm.bcast(mpi_shared_list, root=0)
    task_ids = mpi.distribute_tasks(size, rank, len(mpi_shared_list))
    local_mpi_list = [mpi_shared_list[i] for i in task_ids]

    for id_sim, ms in local_mpi_list:
        base_dir = f"{sims_dir}/{id_sim:04d}"
        BBmeta.make_dir(base_dir)

        if verbose:
            print(f"# {id_sim} | {ms}")
        if do_plots:
            plots_dir = f"{out_dir}/plots/cov_sims/{id_sim:04d}"
            BBmeta.make_dir(plots_dir)

        ft = meta.freq_tag_from_map_set(ms)
        noise_map_dir = meta.covariance["noise_map_sims_dir"][ms]
        noise_map_template = meta.covariance["noise_map_sims_template"][ms]

        # Add signal if exists, otherwise just use noise
        if use_alms:
            fname = signal_alm_templates[ms].format(id_sim=id_sim,
                                                    freq_tag=ft)
            alms = hp.read_alm(f"{signal_alm_dirs[ms]}/{fname}",
                               hdu=(1, 2, 3))  # signal alms are in muK_CMB
            alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]
            signal = mu.alm2map(alms_beamed,
                                pix_type=meta.pix_type,
                                nside=meta.nside,
                                car_map_template=meta.car_template)

        if use_maps:
            fname = signal_map_templates[ms].format(id_sim=id_sim,
                                                    freq_tag=ft)
            signal = mu.read_map(
                f"{signal_map_dirs[ms]}/{fname}",
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                car_template=meta.car_template,
                convert_K_to_muK=False,  # signal maps are in muK_CMB
            )
        if (use_maps or use_alms) and do_plots:
            mu.plot_map(
                signal,
                file_name=f"{plots_dir}/cov_sims_{ms}_signal",  # noqa
                pix_type=meta.pix_type,
                lims=[[-500, 500], [-20, 20], [-20, 20]]
            )

        for id_bundle in range(meta.n_bundles_from_map_set(ms)):
            map_name = f"cov_sims_{ms}_bundle{id_bundle}.fits"
            fname = noise_map_template.format(id_sim=id_sim, map_set=ms,
                                              id_bundle=id_bundle)
            noise_map = mu.read_map(
                f"{noise_map_dir}/{fname}",
                pix_type=meta.pix_type,
                fields_hp=[0, 1, 2],
                convert_K_to_muK=True,  # noise maps are in K_CMB
                car_template=meta.car_template
            )
            if do_plots:
                noise_masked = noise_map.copy()
                mu.multiply_map(binary, noise_masked, meta.pix_type)
                mu.plot_map(
                    noise_masked,
                    file_name=f"{plots_dir}/cov_sims_{ms}_bundle{id_bundle}_noise",  # noqa
                    pix_type=meta.pix_type,
                    lims=[[-500, 500], [-20, 20], [-20, 20]]
                )

            coadded_map = noise_map.copy()
            if (use_maps or use_alms):
                mu.add_map(signal, coadded_map, meta.pix_type)

            mu.multiply_map(binary, coadded_map, meta.pix_type)
            mu.write_map(
                f"{base_dir}/{map_name}",
                coadded_map,
                pix_type=meta.pix_type
            )

            if do_plots:
                mu.plot_map(
                    coadded_map,
                    file_name=f"{plots_dir}/cov_sims_{ms}_bundle{id_bundle}_coadd",  # noqa
                    pix_type=meta.pix_type,
                    lims=[[-500, 500], [-20, 20], [-20, 20]]
                    )
                print(f"Pol map variance: {np.std(coadded_map[1:]):.2E}")
        if verbose and do_plots:
            mpi.print_rnk0(f" PLOTS: {plots_dir}", rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true", help="Do not plot the maps.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
