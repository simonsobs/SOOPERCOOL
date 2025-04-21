from soopercool import BBmeta
from soopercool import ps_utils as pu
from soopercool import map_utils as mu
import argparse
import numpy as np
import pymaster as nmt
import re


def main(args):
    """
    """
    meta = BBmeta(args.globals)
    do_plots = not args.no_plots

    # DEBUG
    print("pix_type", meta.pix_type)

    out_dir = meta.output_directory
    cells_dir = f"{out_dir}/cells"

    BBmeta.make_dir(cells_dir)

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type,
                       car_template=meta.car_template)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)

    if do_plots:
        import healpy as hp
        import matplotlib.pyplot as plt

        lmax = meta.lmax
        lb = nmt_bins.get_effective_ells()
        lmax_bins = nmt_bins.get_ell_max(nmt_bins.get_n_bands() - 1)
        lb_msk = lb < lmax + 10
        cb2db = lb*(lb+1)/2./np.pi
        field_pairs = ["TT", "EE", "BB", "TE"]

        cl_plot_dir = f"{out_dir}/plots/cells"
        map_plot_dir = f"{out_dir}/plots/maps"
        BBmeta.make_dir(cl_plot_dir)
        BBmeta.make_dir(map_plot_dir)

        fiducial_cmb = meta.covariance["fiducial_cmb"]
        fiducial_dust = meta.covariance["fiducial_dust"]
        fiducial_synch = meta.covariance["fiducial_synch"]

    # Create namaster fields
    fields = {}
    for map_name in meta.maps_list:
        map_set, id_bundle = map_name.split("__")

        # Load maps
        map_dir = meta.map_dir_from_map_set(map_set)
        map_template = meta.map_template_from_map_set(map_set)

        map_file = map_template.replace(
            "{id_bundle}",
            str(id_bundle)
        )
        type_options = [f for f in re.findall(r"\{.*?\}", map_template) if "|" in f][0] # noqa
        # Select the hitmap
        option = type_options.replace("{", "").replace("}", "").split("|")[0]

        map_file = map_file.replace(type_options, option)

        m = mu.read_map(f"{map_dir}/{map_file}",
                        pix_type=meta.pix_type,
                        fields_hp=[0, 1, 2],
                        car_template=meta.car_template,
                        convert_K_to_muK=True)
        if do_plots:
            fname = f"{map_plot_dir}/map_{map_set}_bundle{id_bundle}"
            lims = [[-5000, 5000], [-300, 300], [-300, 300]]
            title = f"({map_set}, bundle {id_bundle})"
            mu.plot_map(m, file_name=fname, lims=lims, title=title,
                        pix_type=meta.pix_type)

        wcs = None
        if hasattr(m, 'wcs'):
            # This is a patch. Reproject mask and map onto template geometry.
            from pixell import enmap
            tshape, twcs = enmap.read_map_geometry(meta.car_template)
            shape, wcs = enmap.overlap(m.shape, m.wcs, tshape, twcs)
            shape, wcs = enmap.overlap(mask.shape, mask.wcs, shape, wcs)
            flat_template = enmap.zeros((3, shape[0], shape[1]), wcs)
            mask = enmap.insert(flat_template.copy()[0], mask)
            m = enmap.insert(flat_template.copy(), m)

        field_spin0 = nmt.NmtField(mask, m[:1], wcs=wcs)
        field_spin2 = nmt.NmtField(mask, m[1:], wcs=wcs, purify_b=meta.pure_B)

        fields[map_set, id_bundle] = {
            "spin0": field_spin0,
            "spin2": field_spin2
        }

    inv_couplings_beamed = meta.get_inverse_couplings(filtered=True)

    for map_name1, map_name2 in meta.get_ps_names_list(type="all",
                                                       coadd=False):
        map_set1, id_bundle1 = map_name1.split("__")
        map_set2, id_bundle2 = map_name2.split("__")
        pcls = pu.get_coupled_pseudo_cls(
                fields[map_set1, id_bundle1],
                fields[map_set2, id_bundle2],
                nmt_bins
                )

        decoupled_pcls = pu.decouple_pseudo_cls(
                pcls, inv_couplings_beamed[map_set1, map_set2]
                )

        np.savez(f"{cells_dir}/decoupled_pcls_{map_name1}_x_{map_name2}.npz",
                 **decoupled_pcls, lb=nmt_bins.get_effective_ells())

        if do_plots:
            nu1 = meta.freq_tag_from_map_set(map_set1)
            nu2 = meta.freq_tag_from_map_set(map_set2)

            clb_th = None
            if fiducial_cmb:
                cmb_cl = hp.read_cl(fiducial_cmb)[:, :lmax_bins+1]
                cmb_clb = nmt_bins.bin_cell(cmb_cl)[:, lb_msk]
                clb_th = cmb_clb
            if fiducial_dust:
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu1, nu2=nu2)
                )[:, :lmax_bins+1]
                dust_clb = nmt_bins.bin_cell(dust_cl)[:, lb_msk]
                if clb_th:
                    clb_th += dust_clb
                else:
                    clb_th = dust_clb
            if fiducial_synch:
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu1, nu2=nu2)
                )[:, :lmax_bins+1]
                synch_clb = nmt_bins.bin_cell(synch_cl)[:, lb_msk]
                if clb_th:
                    clb_th += synch_clb
                else:
                    clb_th = synch_clb

            for i, fp in enumerate(field_pairs):
                plt.title(f"({map_set1}, bundle {id_bundle1}) "
                          f"x ({map_set2}, bundle {id_bundle2}) | {fp}",
                          fontsize=9)
                plt.loglog(lb[lb_msk],
                           cb2db[lb_msk]*decoupled_pcls[fp][lb_msk],
                           c="navy", ls="-", label="Data")
                if clb_th is not None:
                    plt.loglog(lb[lb_msk], cb2db[lb_msk]*clb_th[i],
                               c="k", ls="--", label="Theory")
                plt.ylabel(
                    r"$D_\ell^\mathrm{%s} \; [\mu K_\mathrm{CMB}^2]$" % fp,
                    fontsize=15
                )
                plt.xlabel(r"$\ell$", fontsize=15)
                plt.legend(fontsize=13)
                plt.savefig(f"{cl_plot_dir}/pcls_{map_set1}_"
                            f"bundle{id_bundle1}_{map_set2}_"
                            f"bundle{id_bundle2}_{fp}.png")
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not make plots.")
    args = parser.parse_args()
    main(args)
