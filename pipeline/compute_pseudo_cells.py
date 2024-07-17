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

    out_dir = meta.output_directory

    cells_dir = f"{out_dir}/cells"
    couplings_dir = f"{out_dir}/couplings"

    BBmeta.make_dir(cells_dir)

    mask = mu.read_map(meta.masks["analysis_mask"],
                       pix_type=meta.pix_type)

    binning = np.load(meta.binning_file)
    nmt_bins = nmt.NmtBin.from_edges(binning["bin_low"],
                                     binning["bin_high"] + 1)
    n_bins = nmt_bins.get_n_bands()

    if do_plots:
        import healpy as hp
        import matplotlib.pyplot as plt

        lmax = 3*meta.nside - 1
        ll = np.arange(lmax + 1)
        cl2dl = ll*(ll+1)/2./np.pi
        lb = nmt_bins.get_effective_ells()
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

        m = mu.read_map(f"{map_dir}/{map_file}", field=[0, 1, 2],
                        pix_type=meta.pix_type)
        if do_plots:
            for i, f in enumerate(["T", "Q", "U"]):
                hp.mollview(m[i],
                            cmap="RdYlBu_r",
                            title=f"({map_set}, bundle {id_bundle}) | {f}",
                            min=-300 if f == "T" else -10,
                            max=300 if f == "T" else 10)
                plt.savefig(f"{map_plot_dir}/map_{map_set}_"
                            f"bundle{id_bundle}_{f}.png")
                plt.close()

        field_spin0 = nmt.NmtField(mask, m[:1])
        field_spin2 = nmt.NmtField(mask, m[1:], purify_b=meta.pure_B)

        fields[map_set, id_bundle] = {
            "spin0": field_spin0,
            "spin2": field_spin2
        }

    inv_couplings_beamed = {}

    for ms1, ms2 in meta.get_ps_names_list(type="all", coadd=True):
        inv_couplings_beamed[ms1, ms2] = np.load(f"{couplings_dir}/couplings_{ms1}_{ms2}.npz")["inv_coupling"].reshape([n_bins*9, n_bins*9]) # noqa

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

            for i, fp in enumerate(field_pairs):
                cmb_cl = hp.read_cl(fiducial_cmb)[i, :lmax+1]
                dust_cl = hp.read_cl(
                    fiducial_dust.format(nu1=nu1, nu2=nu2)
                )[i, :lmax+1]
                synch_cl = hp.read_cl(
                    fiducial_synch.format(nu1=nu1, nu2=nu2)
                )[i, :lmax+1]
                clth = cmb_cl + dust_cl + synch_cl

                plt.title(f"({map_set1}, bundle {id_bundle1}) "
                          f"x ({map_set2}, bundle {id_bundle2}) | {fp}")
                plt.loglog(lb, cb2db*decoupled_pcls[fp],
                           c="navy", marker="o", mfc="w", ls="", label="Data")
                plt.loglog(ll, cl2dl*clth,
                           c="darkorange", ls="--", label="Theory")
                plt.ylabel(r"$D_\ell$", fontsize=15)
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
