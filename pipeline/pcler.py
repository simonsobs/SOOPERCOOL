import argparse
import healpy as hp
import numpy as np
import os
from bbmaster import BBmeta
from bbmaster.utils import *
import pymaster as nmt
from itertools import product
import matplotlib.pyplot as plt
import warnings


def get_coupled_pseudo_cls(fields1, fields2, nmt_binning):
    """
    """
    spins = list(fields1.keys())

    pcls = {}
    for spin1 in spins:
        for spin2 in spins:

            f1 = fields1[spin1]
            f2 = fields2[spin2]

            coupled_cell = nmt.compute_coupled_cell(f1, f2)
            coupled_cell = coupled_cell[:, :nmt_binning.lmax+1]

            pcls[f"{spin1}x{spin2}"] = nmt_binning.bin_cell(coupled_cell)
    
    return pcls

def decouple_pseudo_cls(coupled_pseudo_cells, coupling_inv):
    """
    """
    decoupled_pcls = {}
    for spin_comb, coupled_pcl in coupled_pseudo_cells.items():
        n_bins = coupled_pcl.shape[-1]
        decoupled_pcl = coupling_inv[spin_comb] @ coupled_pcl.flatten()
        if spin_comb == "spin0xspin0": size = 1
        elif spin_comb == "spin0xspin2" or spin_comb == "spin2xspin0": size = 2
        elif spin_comb == "spin2xspin2": size = 4 
        decoupled_pcl = decoupled_pcl.reshape((size, n_bins))

        decoupled_pcls[spin_comb] = decoupled_pcl
    
    decoupled_pcls = field_pairs_from_spins(decoupled_pcls)

    return decoupled_pcls


def field_pairs_from_spins(cls_in_dict):
    """
    """
    cls_out_dict = {}

    field_spin_mapping = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin0": ["ET", "BT"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }

    for spin_pair in cls_in_dict:
        for index, field_pair in enumerate(field_spin_mapping[spin_pair]):

            cls_out_dict[field_pair] = cls_in_dict[spin_pair][index]

    return cls_out_dict


def get_pcls_mat_transfer(fields, nmt_binning):
    """
    attempt to get TF for T and Pol.
    """
    n_bins = nmt_binning.get_n_bands()
    pcls_mat_00 = np.zeros((1, 1, n_bins))
    pcls_mat_02 = np.zeros((2, 2, n_bins))
    pcls_mat_22 = np.zeros((4, 4, n_bins))

    index = 0
    cases = ["pureE", "pureB"]
    for index, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls = get_coupled_pseudo_cls(fields[pure_type1], fields[pure_type2], nmt_binning)
        pcls_mat_22[index] = pcls["spin2xspin2"]

        pcls_mat_02[cases.index(pure_type2)] = pcls["spin0xspin2"]
    
    pcls_mat_00[0] = pcls["spin0xspin0"]

    return {"spin0xspin0": pcls_mat_00,
            "spin0xspin2": pcls_mat_02,
            "spin2xspin2": pcls_mat_22}

def pcler(args):
    """
    """
    meta = BBmeta(args.globals)

    mask = meta.read_mask("analysis")
    nmt_binning = meta.read_nmt_binning()
    n_bins = nmt_binning.get_n_bands()
    binary_mask = meta.read_mask("binary")
    fsky = np.mean(binary_mask)

    if args.tf_est or args.tf_val:
        cl_dir = meta.cell_transfer_directory
    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory

    if args.data or args.sims:

        # Set the number of sims to loop over
        Nsims = meta.num_sims if args.sims else 1

        # Load the inverse coupling matrix
        inv_couplings = {}
        for map_set1, map_set2 in meta.get_ps_names_list(type="all", coadd=True):
            couplings = np.load(f"{meta.coupling_directory}/couplings_{map_set1}_{map_set2}.npz")
            coupling_dict = {
                "spin0xspin0": couplings["inv_coupling_spin0xspin0"].reshape([n_bins, n_bins]),
                "spin0xspin2": couplings["inv_coupling_spin0xspin2"].reshape([2*n_bins, 2*n_bins]),
                "spin2xspin0": couplings["inv_coupling_spin0xspin2"].reshape([2*n_bins, 2*n_bins]),
                "spin2xspin2": couplings["inv_coupling_spin2xspin2"].reshape([4*n_bins, 4*n_bins])
            }
            inv_couplings[map_set1, map_set2] = coupling_dict
            if map_set1 != map_set2:
                # the only map set dependance is on the beam
                inv_couplings[map_set2, map_set1] = coupling_dict
        
        if args.plots:
            el = nmt_binning.get_effective_ells()
            cells_plots = {}
            
            for map_name1, map_name2 in meta.get_ps_names_list(type="all", coadd=False):
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                split_label = f"split{id_split1}_x_split{id_split2}"
                for pol1, pol2 in ['EE', 'EB', 'BB']:
                    plot_label = f"{map_set1}__{map_set2}__{pol1}{pol2}"
                    has_noise_bias = (id_split1==id_split2) and (map_set1 == map_set2)
                    if not plot_label in cells_plots:
                        cells_plots[plot_label] = {}
                    if not split_label in cells_plots[plot_label]:
                        cells_plots[plot_label][split_label] = {}
                    if has_noise_bias:
                        cells_plots[plot_label][split_label]['noisy'] = np.zeros((Nsims, n_bins))
                    else:
                        cells_plots[plot_label][split_label]['noiseless'] = np.zeros((Nsims, n_bins))

        for id_sim in range(Nsims):
            fields = {}
            # Load namaster fields that will be used
            for map_name in meta.maps_list:
                map_set, id_split = map_name.split("__")

                # Load maps
                map_file = meta.get_map_filename(
                    map_set, id_split, id_sim=id_sim if Nsims > 1 else None
                )
                map_file = map_file.replace(".fits", "_filtered.fits")
                map = hp.read_map(map_file, field=[0,1,2])

                # Include beam in namaster fields to deconvolve it
                field_spin0 = nmt.NmtField(mask, [map[0]])
                field_spin2 = nmt.NmtField(mask, [map[1], map[2]])

                fields[map_set, id_split] = {
                    "spin0": field_spin0, 
                    "spin2": field_spin2
                }
            
            for map_name1, map_name2 in meta.get_ps_names_list(type="all", 
                                                               coadd=False):
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                pcls = get_coupled_pseudo_cls(
                    fields[map_set1, id_split1], 
                    fields[map_set2, id_split2], 
                    nmt_binning
                )

                decoupled_pcls = decouple_pseudo_cls(
                    pcls, inv_couplings[map_set1, map_set2]
                )
                
                sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                np.savez(f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz", 
                         **decoupled_pcls, lb=nmt_binning.get_effective_ells())
                
                if args.plots:
                    split_label = f"split{id_split1}_x_split{id_split2}"
                    for field_pair in ['EE', 'EB', 'BB']:
                        plot_label = f"{map_set1}__{map_set2}__{field_pair}"
                        has_noise_bias = (id_split1==id_split2) and (map_set1 == map_set2)
                        
                        if has_noise_bias:
                            cells_plots[plot_label][split_label]['noisy'][id_sim] = decoupled_pcls[field_pair]
                        else:
                            cells_plots[plot_label][split_label]['noiseless'][id_sim] = decoupled_pcls[field_pair]
     
        if args.plots:
            plot_dir_rel = meta.sims_directory_rel if Nsims > 1 else meta.map_directory_rel 
            plot_dir = meta.plot_dir_from_output_dir(plot_dir_rel)
            
            for plot_label in cells_plots:
                fig = plt.figure(figsize=(6,4))
                plt.title(plot_label)
                
                # Load theory CMB spectra
                map_set1, map_set2, field_pair = plot_label.split("__")
                el_th, cl_th_dict = theory_cls(meta.cosmology, 
                                               meta.lmax, lmin=meta.lmin)
                cl_th = cl_th_dict[field_pair]
                do_theory_plot = True
                for split_label in cells_plots[plot_label]:
                    cells_splits = cells_plots[plot_label][split_label]
                    
                    if do_theory_plot:
                        # Plot noiseless theory power spectra
                        plt.plot(el_th, 
                                 el_th*(el_th+1)/2./np.pi*(cl_th),
                                 lw=1, ls='--', c='darkred', 
                                 label=f"theory (noiseless)")
                        if 'noisy' in cells_splits:
                            # Plot noisy theory with beam-deconvolved noise
                            _, nl_th_beam_dict = get_noise_cls(
                                fsky, meta.lmax, lmin=meta.lmin, 
                                sensitivity_mode='baseline', 
                                oof_mode='optimistic',
                                is_beam_deconvolved=True
                            )
                            freq_tag = meta.freq_tag_from_map_set(map_set1)
                            n_splits = meta.n_splits_from_map_set(map_set1)
                            nl_th_beam = nl_th_beam_dict["P"][freq_tag]*float(n_splits)
                            plt.plot(el_th, 
                                     el_th*(el_th+1)/2./np.pi*(cl_th+nl_th_beam),
                                     lw=1, ls='-.', c='k', 
                                     label=f"theory (noisy)")
                        do_theory_plot = False
                        
                    # Plot splits power spectra
                    for has_noise_bias, cells_sims in cells_splits.items():
                        cells_sum = 0
                        for id_sim in range(Nsims):
                            cells = cells_sims[id_sim]
                            cells_sum += cells
                            color = 'k' if (has_noise_bias=="noisy") else 'tab:red'
                            if Nsims > 1:
                                plt.plot(el, el*(el+1)/2./np.pi*cells, 
                                         lw=0.3, c=color, alpha=0.5)
                        plt.plot(el, el*(el+1)/2./np.pi*cells_sum/Nsims, 
                                 lw=1, c=color, label=split_label)
                        
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel(r'$\ell(\ell+1)\,C_\ell/2\pi$', fontsize=14)
                    plt.xlabel(r'$\ell$', fontsize=14)
                    plt.legend(bbox_to_anchor=(1,1))
                    plt.savefig(
                        os.path.join(
                            plot_dir, 
                            f"cells_{map_set1}_{map_set2}_{field_pair}.png"
                        ), bbox_inches='tight'
                    )


    if args.tf_est:
        for id_sim in range(meta.tf_est_num_sims):
            
            fields = {"filtered": {}, "unfiltered": {}}
            for pure_type in ["pureE", "pureB"]:
                map_file = meta.get_map_filename_transfer2(id_sim, "tf_est", pure_type=pure_type)
                map_file_filtered = map_file.replace(".fits", "_filtered.fits")

                map = hp.read_map(map_file, field=[0,1,2])
                map_filtered = hp.read_map(map_file_filtered, field=[0,1,2])
                
                # TO-DO: filter temperature only once !
                field = {
                    "spin0": nmt.NmtField(mask, map[:1]),
                    "spin2": nmt.NmtField(mask, map[1:])
                }

                if meta.filtering_type == "toast":
                    pw = hp.pixwin(meta.nside, pol=True, lmax=3*meta.nside-1)
                else:
                    pw = [None, None]
                field_filtered = {
                    "spin0": nmt.NmtField(mask, map_filtered[:1], beam=pw[0]),
                    "spin2": nmt.NmtField(mask, map_filtered[1:], beam=pw[1])
                }

                fields["unfiltered"][pure_type] = field
                fields["filtered"][pure_type] = field_filtered

            pcls_mat_filtered = get_pcls_mat_transfer(fields["filtered"], nmt_binning)
            pcls_mat_unfiltered = get_pcls_mat_transfer(fields["unfiltered"], nmt_binning)

            np.savez(f"{cl_dir}/pcls_mat_tf_est_filtered_{id_sim:04d}.npz",
                     **pcls_mat_filtered)
            np.savez(f"{cl_dir}/pcls_mat_tf_est_unfiltered_{id_sim:04d}.npz",
                     **pcls_mat_unfiltered)
            

    if args.tf_val:
        
        inv_couplings = {}
        for filter_flag in ["filtered", "unfiltered"]:
            couplings = np.load(f"{meta.coupling_directory}/couplings_{filter_flag}.npz")
            inv_couplings[filter_flag] = {
                "spin0xspin0": couplings["inv_coupling_spin0xspin0"].reshape([n_bins, n_bins]),
                "spin0xspin2": couplings["inv_coupling_spin0xspin2"].reshape([2*n_bins, 2*n_bins]),
                "spin2xspin0": couplings["inv_coupling_spin0xspin2"].reshape([2*n_bins, 2*n_bins]),
                "spin2xspin2": couplings["inv_coupling_spin2xspin2"].reshape([4*n_bins, 4*n_bins])
            }

        for id_sim in range(meta.tf_est_num_sims):
            
            for cl_type in ["tf_val", "cosmo"]:

                for filter_flag in ["filtered", "unfiltered"]:
                    map_file = meta.get_map_filename_transfer2(id_sim, cl_type=cl_type)
                    print(map_file)
                    if filter_flag == "filtered":
                        map_file = map_file.replace(".fits", "_filtered.fits")

                    map = hp.read_map(map_file, field=[0,1,2])


                    if (filter_flag == "filtered") and (meta.filtering_type == "toast"):
                        pw = hp.pixwin(meta.nside, pol=True, lmax=3*meta.nside-1)
                    else:
                        pw = [None, None]
                    field = {
                        "spin0": nmt.NmtField(mask, map[:1], beam=pw[0]),
                        "spin2": nmt.NmtField(mask, map[1:], beam=pw[1])
                    }

                    pcls = get_coupled_pseudo_cls(field, field, nmt_binning)

                    decoupled_pcls = decouple_pseudo_cls(pcls, inv_couplings[filter_flag])
                    
                    np.savez(f"{cl_dir}/pcls_{cl_type}_{id_sim:04d}_{filter_flag}.npz",
                            **decoupled_pcls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-Cl calculator")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true", 
                        help="Plot the generated power spectra if True.")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--tf_est", action="store_true")
    mode.add_argument("--tf_val", action="store_true")

    args = parser.parse_args()
    
    if (args.tf_est and args.plots) or (args.tf_val and args.plots):
        warnings.warn("Both --tf_[...] and --plots are set to True. "
                      "This is not implemented yet. Set --plot to False")
        args.plots = False

    pcler(args)