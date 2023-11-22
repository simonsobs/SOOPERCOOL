import argparse
from bbmaster.utils import PipelineManager
import sacc
import numpy as np
import matplotlib.pyplot as plt
import os
from bbmaster import BBmeta

def read_transfer(transfer_file):
    """
    """
    transfer = np.load(transfer_file)
    
    tf_dict = {}
    spin_pairs = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }
    for spin_pair, fields in spin_pairs.items():
        for id1, f1 in enumerate(fields):
            for id2, f2 in enumerate(fields):
                tf_dict[f1, f2] = transfer[f"tf_{spin_pair}"][id1, id2]
                tf_dict[f1, f2, "std"] = transfer[f"tf_std_{spin_pair}"][id1, id2]
    
    return tf_dict

def transfer_validator(args):
    """
    Compare decoupled spectra (including deconvolved form the TF)
    to the unfiltered spectra (deconvolved from mode coupling only)
    Compare coupled spectra (with TF + MCM)
    """
    meta = BBmeta(args.globals)

    fields = ["TT", "TE", "TB", "EE", "EB", "BE", "BB"]

    tf_dict = read_transfer(f"{meta.coupling_directory}/transfer_function.npz")

    nmt_binning = meta.read_nmt_binning()
    lb = nmt_binning.get_effective_ells()

    plot_dir = meta.plot_dir_from_output_dir(meta.coupling_directory)
    # First plot the transfer functions
    plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(7, 7, hspace=0.3, wspace=0.3)
    
    for id1, f1 in enumerate(fields):
        for id2, f2 in enumerate(fields):
            ax = plt.subplot(grid[id1, id2])
    
            if f1 == "TT" and f2 != "TT": 
                ax.axis("off")
                continue
            if f1 in ["TE", "TB"] and f2 not in ["TE", "TB"]: 
                ax.axis("off")
                continue
            if f1 in ["EE", "EB", "BE", "BB"] and f2 not in ["EE", "EB", "BE", "BB"]: 
                ax.axis("off")
                continue

            ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)

            ax.errorbar(
                lb, tf_dict[f1, f2], tf_dict[f1, f2, "std"],
                marker=".", markerfacecolor="white",
                color="navy")

            if not([id1, id2] in [[0,0], [2,1], [2,2], [6,3], [6,4], [6,5], [6,6]]):
                ax.set_xticks([])
            else:
                ax.set_xlabel(r"$\ell$", fontsize=14)

            if f1 == f2:
                ax.axhline(1., color="k", ls="--")
            else:
                ax.axhline(0, color="k", ls="--")

            ax.set_xlim(meta.lmin, meta.lmax)
    
    plt.savefig(f"{plot_dir}/transfer.pdf", bbox_inches="tight")


    # Then we read the decoupled spectra
    # both for the filtered and unfiltered 
    # cases
    cl_dir = meta.cell_transfer_directory
    cross_fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    val_types = ["tf_val", "cosmo"]
    filter_flags = ["filtered", "unfiltered"]

    cls_dict = {
        (val_type, filter_flag, k): [] for val_type in val_types for k in cross_fields for filter_flag in filter_flags
    }

    for val_type in ["tf_val", "cosmo"]:
        for filter_flag in filter_flags:
            for id_sim in range(meta.tf_est_num_sims):

                cls = np.load(f"{cl_dir}/pcls_{val_type}_{id_sim:04d}_{filter_flag}.npz")
                for k in cross_fields:
                    cls_dict[val_type, filter_flag, k] += [cls[k]]
    
    # Compute mean and std
    cls_mean_dict = {
        (val_type, filter_flag, k): np.mean(cls_dict[val_type, filter_flag, k], axis=0)
        for val_type in val_types for filter_flag in filter_flags for k in cross_fields
    }
    cls_std_dict = {
        (val_type, filter_flag, k): np.std(cls_dict[val_type, filter_flag, k], axis=0)
        for val_type in val_types for filter_flag in filter_flags for k in cross_fields
    }
    
    cls_theory = {
        val_type: meta.load_fiducial_cl(val_type) for val_type in val_types
    }

    cls_theory_binned = {}
    bp_win = np.load(f"{meta.coupling_directory}/couplings_unfiltered.npz")
    for spin_comb in ["spin0xspin0", "spin0xspin2", "spin2xspin2"]:
        bpw_mat = bp_win[f"bp_win_{spin_comb}"]
        for val_type in val_types:

            if spin_comb == "spin0xspin0":
                cls_vec = np.array([cls_theory[val_type]["TT"][:meta.lmax+1]]).reshape(1, meta.lmax+1)
            elif spin_comb == "spin0xspin2":
                cls_vec = np.array([cls_theory[val_type]["TE"][:meta.lmax+1], cls_theory[val_type]["TB"][:meta.lmax+1]])
            elif spin_comb == "spin2xspin2":
                cls_vec = np.array([cls_theory[val_type]["EE"][:meta.lmax+1], cls_theory[val_type]["EB"][:meta.lmax+1], cls_theory[val_type]["EB"][:meta.lmax+1], cls_theory[val_type]["BB"][:meta.lmax+1]])
            
            cls_vec_binned = np.einsum("ijkl,kl", bpw_mat, cls_vec)
            if spin_comb == "spin0xspin0":
                cls_theory_binned[val_type, "TT"] = cls_vec_binned[0]
            elif spin_comb == "spin0xspin2":
                cls_theory_binned[val_type, "TE"] = cls_vec_binned[0]
                cls_theory_binned[val_type, "TB"] = cls_vec_binned[1]
            elif spin_comb == "spin2xspin2":
                cls_theory_binned[val_type, "EE"] = cls_vec_binned[0]
                cls_theory_binned[val_type, "EB"] = cls_vec_binned[1]
                cls_theory_binned[val_type, "BE"] = cls_vec_binned[2]
                cls_theory_binned[val_type, "BB"] = cls_vec_binned[3]

    for val_type in val_types:

        plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(9, 3, hspace=0.3, wspace=0.3)
        for id1, f1 in enumerate("TEB"):
            for id2, f2 in enumerate("TEB"):

                # Define subplots
                main = plt.subplot(grid[3*id1:3*(id1+1)-1, id2])
                sub = plt.subplot(grid[3*(id1+1)-1, id2])

                spec = f2 + f1 if id1 > id2 else f1 + f2

                # Plot theory
                l = cls_theory[val_type]["l"]
                rescaling = 1 if val_type == "tf_val" else l * (l + 1) / (2*np.pi)
                main.plot(l, rescaling*cls_theory[val_type][spec], color="k")

                offset = 0.5
                rescaling = 1 if val_type == "tf_val" else lb * (lb + 1) / (2*np.pi)
                # Plot filtered & unfiltered (decoupled)
                main.errorbar(
                    lb-offset, rescaling*cls_mean_dict[val_type, "unfiltered", spec],
                    rescaling*cls_std_dict[val_type, "unfiltered", spec],
                    color="navy", marker=".", markerfacecolor="white",
                    label=r"Unfiltered decoupled $C_\ell$", ls="None"
                )
                main.errorbar(
                    lb+offset, rescaling*cls_mean_dict[val_type, "filtered", spec],
                    rescaling*cls_std_dict[val_type, "filtered", spec],
                    color="darkorange", marker=".", markerfacecolor="white",
                    label=r"Filtered decoupled $C_\ell$", ls="None"
                )

                if f1 == f2:
                    main.set_yscale("log")

                # Plot residuals
                residual_unfiltered = (cls_mean_dict[val_type, "unfiltered", spec] - cls_theory_binned[val_type, spec]) / cls_std_dict[val_type, "unfiltered", spec]
                residual_filtered = (cls_mean_dict[val_type, "filtered", spec] - cls_theory_binned[val_type, spec]) / cls_std_dict[val_type, "unfiltered", spec]

                sub.axhspan(-2, 2, color="gray", alpha=0.2)
                sub.axhspan(-1, 1, color="gray", alpha=0.7)
                
                sub.axhline(0, color="k")
                sub.plot(lb-offset, residual_unfiltered * np.sqrt(meta.tf_est_num_sims), color="navy", marker=".", markerfacecolor="white", ls="None")
                sub.plot(lb+offset, residual_filtered * np.sqrt(meta.tf_est_num_sims), color="darkorange", marker=".", markerfacecolor="white", ls="None")

                
                
                # Multipole range
                main.set_xlim(2, meta.lmax)
                sub.set_xlim(*main.get_xlim())

                # Cosmetix
                main.set_title(f1+f2, fontsize=14)
                if spec == "TT":
                    main.legend(fontsize=13)
                main.set_xticklabels([])
                if id1 != 2:
                    sub.set_xticklabels([])
                else:
                    sub.set_xlabel(r"$\ell$", fontsize=13)

                if id2 == 0:
                    if isinstance(rescaling, float):
                        main.set_ylabel(r"$C_\ell$", fontsize=13)
                    else:
                        main.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$", fontsize=13)
                    sub.set_ylabel(r"$\Delta C_\ell / (\sigma/\sqrt{N_\mathrm{sims}})$", fontsize=13)

        plt.savefig(f"{plot_dir}/decoupled_{val_type}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer function')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')

    args = parser.parse_args()

    transfer_validator(args)