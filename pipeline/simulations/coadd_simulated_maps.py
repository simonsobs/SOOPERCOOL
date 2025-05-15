import argparse
from soopercool import BBmeta
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
import healpy as hp


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    verbose = args.verbose

    do_plots = not args.no_plots

    out_dir = meta.output_directory
    masks_dir = f"{out_dir}/masks"

    sims_dir = f"{out_dir}/cov_sims"
    BBmeta.make_dir(sims_dir)

    signal_alm_dirs = meta.covariance["signal_alm_sims_dir"]
    signal_alm_templates = meta.covariance["signal_alm_sims_template"]

    binary = mu.read_map(f"{masks_dir}/binary_mask.fits",
                         pix_type=meta.pix_type,
                         car_template=meta.car_template)

    mu._check_pix_type(meta.pix_type)
    lmax = meta.lmax

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
                cmb_cl = hp.read_cl(fiducial_cmb)[i, :lmax+1]
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu, nu2=nu)
                )[i, :lmax+1]
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu, nu2=nu)
                )[i, :lmax+1]
                clth[ms, fp] = (cmb_cl + dust_cl + synch_cl) * beams[ms]**2

    mpi.init(True)

    for id_sim in mpi.taskrange(meta.covariance["cov_num_sims"] - 1):

        base_dir = f"{sims_dir}/{id_sim:04d}"
        BBmeta.make_dir(base_dir)
        if do_plots:
            plots_dir = f"{out_dir}/plots/cov_sims/{id_sim:04d}"
            BBmeta.make_dir(plots_dir)

        for ms in meta.map_sets_list:
            if verbose:
                print(f"# {id_sim} | {ms}")

            noise_map_dir = meta.covariance["noise_map_sims_dir"][ms]
            noise_map_template = meta.covariance["noise_map_sims_template"][ms]

            for id_bundle in range(meta.n_bundles_from_map_set(ms)):

                fname = noise_map_template.format(id_sim=id_sim, map_set=ms,
                                                  id_bundle=id_bundle)
                bundle_map = mu.read_map(
                    f"{noise_map_dir}/{fname}",
                    pix_type=meta.pix_type,
                    fields_hp=[0, 1, 2],
                    car_template=meta.car_template
                )
                if do_plots and id_bundle == 0:
                    pass
                    # # TODO: Generalize to masked noise realisation
                    # noise_cl = {
                    #     fp: hp.anafast(bundle_map)[i]
                    #     for i, fp in enumerate(field_pairs)
                    # }
                    # for i, fp in enumerate(field_pairs):
                    #     plt.title(f"{ms} | bundle {id_bundle} | {fp}")
                    #     plt.loglog(ll, cl2dl*signal_cl[fp], c="navy",
                    #                label="Signal")
                    #     plt.loglog(ll, cl2dl*noise_cl[fp], c="grey", alpha=0.5,  # noqa
                    #                label="Noise")
                    #     plt.loglog(ll, cl2dl*clth[ms, fp], c="darkorange",
                    #                label="Theory (beamed)")
                    #     plt.xlabel(r"$\ell$", fontsize=15)
                    #     plt.ylabel(r"$D_\ell$", fontsize=15)
                    #     plt.legend()
                    #     plt.savefig(f"{plots_dir}/cl_{ms}_bundle{id_bundle}_{fp}.png")  # noqa
                    #     plt.close()

                # Add signal if exists, otherwise just use noise
                if signal_alm_templates[ms]:
                    ft = meta.freq_tag_from_map_set(ms)
                    fname = signal_alm_templates[ms].format(id_sim=id_sim,
                                                            freq_tag=ft)
                    alms = hp.read_alm(f"{signal_alm_dirs[ms]}/{fname}",
                                       hdu=(1, 2, 3))
                    alms_beamed = [hp.almxfl(alm, beams[ms]) for alm in alms]

                    # # TODO
                    # if do_plots:
                    #     signal_cl = {fp: hp.alm2cl(alms_beamed)[i]
                    #                  for i, fp in enumerate(field_pairs)}
                    signal = mu.alm2map(alms_beamed,
                                        pix_type=meta.pix_type,
                                        nside=meta.nside,
                                        car_map_template=meta.car_template)

                    mu.add_map(signal, bundle_map, meta.pix_type)

                map_name = f"cov_sims_{ms}_bundle{id_bundle}.fits"
                map_masked = bundle_map.copy()
                mu.multiply_map(binary, map_masked, meta.pix_type)
                mu.write_map(
                    f"{base_dir}/{map_name}",
                    map_masked,
                    pix_type=meta.pix_type
                )

                if do_plots:
                    mu.plot_map(map_masked, pix_type=meta.pix_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true", help="Do not plot the maps.") # noqa
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    main(args)
