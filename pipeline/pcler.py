import argparse
import healpy as hp
import numpy as np
import os
from bbmaster import BBmeta
import pymaster as nmt
from itertools import product


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
        decoupled_pcl = decoupled_pcl.reshape((4, n_bins))

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
    POL ONLY !!!
    """
    n_bins = nmt_binning.get_n_bands()
    pcls_mat = np.zeros((4,4,n_bins))

    index = 0
    cases = ["pureE", "pureB"]
    for index, (pure_type1, pure_type2) in enumerate(product(cases, cases)):
        pcls = get_coupled_pseudo_cls(fields[pure_type1], fields[pure_type2], nmt_binning)
        pcls = pcls["spin2xspin2"]
        pcls_mat[index] = pcls

    return pcls_mat


def pcler(args):
    """
    """
    meta = BBmeta(args.globals)

    mask = meta.read_mask("analysis")
    nmt_binning = meta.read_nmt_binning()
    n_bins = nmt_binning.get_n_bands()

    if args.tf_est or args.tf_val:
        cl_dir = meta.cell_transfer_directory
    if args.data:
        cl_dir = meta.cell_data_directory
    if args.sims:
        cl_dir = meta.cell_sims_directory

    if args.data or args.sims:
        
        # Load beams for each map set
        beams = {}
        for map_set in meta.map_sets_list:
            l, bl = meta.read_beam(map_set)
            beams[map_set] = bl

        # Set the number of sims to loop over
        Nsims = meta.num_sims if args.sims else 1

        # Load the inverse coupling matrix
        inv_couplings = {}
        for map_set1, map_set2 in meta.get_ps_names_list(type="all", coadd=True):
            couplings = np.load(f"{meta.coupling_directory}/couplings_{map_set1}_{map_set2}.npz")
            coupling_dict = {
                "spin2xspin2": couplings['wcal_inv'].reshape([4*n_bins, 4*n_bins])
            }
            inv_couplings[map_set1, map_set2] = coupling_dict
            if map_set1 != map_set2:
                # the only map set dependance is on the beam
                inv_couplings[map_set2, map_set1] = coupling_dict
        
        if args.plots:
            ells_effective = nmt_binning.get_effective_ells()
            cells_plots = {}

        for id_sim in range(Nsims):
            fields = {}
            # Load namaster fields that will be used
            for map_name in meta.maps_list:
                map_set, id_split = map_name.split("__")

                # Load maps
                map_file = meta.get_map_filename(map_set, id_split, id_sim=id_sim if Nsims > 1 else None)
                map_file = map_file.replace(".fits", "_filtered.fits")
                map = hp.read_map(map_file, field=[0,1,2])

                # Include beam in namaster fields to deconvolve it
                field_spin0 = nmt.NmtField(mask, [map[0]], beam=beams[map_set])
                field_spin2 = nmt.NmtField(mask, [map[1], map[2]], beam=beams[map_set])

                fields[map_set, id_split] = {
                    #"spin0": field_spin0, 
                    "spin2": field_spin2
                }
            
            for map_name1, map_name2 in meta.get_ps_names_list():
                map_set1, id_split1 = map_name1.split("__")
                map_set2, id_split2 = map_name2.split("__")
                pcls = get_coupled_pseudo_cls(
                    fields[map_set1, id_split1], 
                    fields[map_set2, id_split2], 
                    nmt_binning
                )

                decoupled_pcls = decouple_pseudo_cls(pcls, wcal_inv)
                
                sim_label = f"_{id_sim:04d}" if Nsims > 1 else ""
                np.savez(f"{cl_dir}/decoupled_pcls_nobeam_{map_name1}_{map_name2}{sim_label}.npz", 
                         **decoupled_pcls, lb=nmt_binning.get_effective_ells())
                
                if args.plots:
                    cells_plots[(map_set1,map_set2)] = {'noisy':{}, 'noiseless':{}}
                    has_noise = map_set1==map_set2 and id_split1==id_split2
                    if has_noise:
                        cells_plots[(map_set1,map_set2)]['noisy'][
                            (id_split1,id_split2)] = decoupled_pcls
                    else:
                        cells_plots[(map_set1,map_set2)]['noiseless'][
                            (id_split1,id_split2)] = decoupled_pcls
     
        if args.plots:
            plot_dir = meta.plot_dir_from_output_dir(meta.map_directory_relative)
            
            for map_set1, map_set2 in cells_plots:
                for field_pair in ['EE','EB','BB']:
                    fig = plt.figure(figsize=(8,4))
                    cells_types = cells_plots[(map_set1, map_set2)]
                    for is_noisy in cells_types:
                        cells_splits = cells_types[is_noisy]
                        for id_split1, id_split2 in cells_splits:
                            cells = cells_splits[(id_split1, id_split2)][
                                field_pair
                            ]
                            label = f"{map_set1}__{id_split1}_x_"
                                    f"{map_set2}__{id_split2}___{field_pair}"
                            color = 'grey'
                            if is_noisy=='noiseless':
                                color = 'tab:red'
                            plt.plot(ells_effective, cells, lw=0.7, c=color,
                                     label=label)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.legend(bbox_to_anchor=(2,1))
                    plt.savefig(
                        os.path.join(
                            plot_dir, f"{map_set1}_{map_set2}_{field_pair}.png"
                        )
                    )

    if args.tf_est:
        for id_sim in range(meta.tf_est_num_sims):
            
            fields = {"filtered": {}, "unfiltered": {}}
            for pure_type in ["pureE", "pureB"]:
                map_file = meta.get_map_filename_transfer2(id_sim, "tf_est", pure_type=pure_type)
                map_file_filtered = map_file.replace(".fits", "_filtered.fits")

                map = hp.read_map(map_file, field=[1,2])
                map_filtered = hp.read_map(map_file_filtered, field=[1,2])

                field = {"spin2": nmt.NmtField(mask, map)}
                field_filtered = {"spin2": nmt.NmtField(mask, map_filtered)}

                fields["unfiltered"][pure_type] = field
                fields["filtered"][pure_type] = field_filtered

            pcls_mat_filtered = get_pcls_mat_transfer(fields["filtered"], nmt_binning)
            pcls_mat_unfiltered = get_pcls_mat_transfer(fields["unfiltered"], nmt_binning)

            np.savez(f"{cl_dir}/pcls_mat_tf_est_{id_sim:04d}.npz", pcls_mat_filtered=pcls_mat_filtered, pcls_mat_unfiltered=pcls_mat_unfiltered)

    if args.tf_val:
        for id_sim in range(meta.tf_est_num_sims):
            
            for cl_type in ["tf_val", "cosmo"]:
                map_file = meta.get_map_filename_transfer2(id_sim, cl_type=cl_type)
                map_file.replace(".fits", "_filtered.fits")

                map = hp.read_map(map_file, field=[1,2])

                field = {'spin2': nmt.NmtField(mask, map)}

                pcls = get_coupled_pseudo_cls(field, field, nmt_binning)["spin2xspin2"]

                np.savez(f"{cl_dir}/pcls_{cl_type}_{id_sim:04d}_filtered.npz",
                         EE=pcls[0], EB=pcls[1], BE=pcls[2], BB=pcls[3])

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
    
    if args.sims and args.plots:
        warnings.warn("Both --sims and --plot are set to True. "
                      "Too many plots will be generated. Set --plot to False")
        args.plots = False
        
    args = parser.parse_args()

    pcler(args)