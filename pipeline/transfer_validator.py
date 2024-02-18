import argparse
import numpy as np
from soopercool import BBmeta
from soopercool.utils import (bin_validation_power_spectra,
                              plot_transfer_function,
                              plot_transfer_validation)


def read_transfer(transfer_file):
    """
    """
    tfd = np.load(transfer_file)

    tf_dict = {}
    spin_pairs = {
        "spin0xspin0": ["TT"],
        "spin0xspin2": ["TE", "TB"],
        "spin2xspin2": ["EE", "EB", "BE", "BB"]
    }
    for spin_pair, fields in spin_pairs.items():
        for id1, f1 in enumerate(fields):
            for id2, f2 in enumerate(fields):
                tf_dict[f1, f2] = tfd[f"tf_{spin_pair}"][id1, id2]
                tf_dict[f1, f2, "std"] = tfd[f"tf_std_{spin_pair}"][id1, id2]
    return tf_dict


def transfer_validator(args):
    """
    Compare decoupled spectra (including deconvolved form the TF)
    to the unfiltered spectra (deconvolved from mode coupling only)
    Compare coupled spectra (with TF + MCM)
    """
    meta = BBmeta(args.globals)

    nmt_binning = meta.read_nmt_binning()
    tf_dict = read_transfer(f"{meta.coupling_directory}/transfer_function.npz")

    # First plot the transfer function
    plot_transfer_function(meta, tf_dict)

    # Compute theory power spectra and bin them
    # TODO: implement frequency-dependent theory (foregrounds!)
    val_types = ["tf_val", "cosmo"]
    map_set_pairs = (meta.get_ps_names_list(type="all", coadd=True)
                     if meta.validate_beam else [(None, None)])

    cls_theory = {
        val_type: meta.load_fiducial_cl(val_type)
        for val_type in val_types
    }
    bp_win = np.load(f"{meta.coupling_directory}/couplings_unfiltered.npz")
    cls_theory_binned = bin_validation_power_spectra(cls_theory,
                                                     nmt_binning,
                                                     bp_win)

    # Then we read the decoupled spectra
    # both for the filtered and unfiltered cases
    cl_dir = meta.cell_transfer_directory
    cross_fields = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    filter_flags = (["filtered"] if meta.validate_beam
                    else ["filtered", "unfiltered"])

    for map_sets in map_set_pairs:
        cls_dict = {
            (val_type, filter_flag, k): []
            for val_type in val_types
            for k in cross_fields
            for filter_flag in filter_flags
        }

        for val_type in val_types:
            for filter_flag in filter_flags:
                for id_sim in range(meta.tf_est_num_sims):
                    cl_prefix = f"pcls_{val_type}_{id_sim:04d}"
                    cl_suffix = (f"_{map_sets[0]}_{map_sets[1]}"
                                 if meta.validate_beam else f"_{filter_flag}")
                    cl_name = cl_prefix + cl_suffix
                    cls = np.load(f"{cl_dir}/{cl_name}.npz")

                    for k in cross_fields:
                        cls_dict[val_type, filter_flag, k] += [cls[k]]

        # Compute mean and std
        cls_mean_dict = {
            (val_type, filter_flag, k):
            np.mean(cls_dict[val_type, filter_flag, k], axis=0)
                for val_type in val_types
                for filter_flag in filter_flags
                for k in cross_fields
        }
        cls_std_dict = {
            (val_type, filter_flag, k):
            np.std(cls_dict[val_type, filter_flag, k], axis=0)
                for val_type in val_types
                for filter_flag in filter_flags
                for k in cross_fields
        }

        plot_transfer_validation(meta, map_sets[0], map_sets[1],
                                 cls_theory, cls_theory_binned,
                                 cls_mean_dict, cls_std_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer function')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')

    args = parser.parse_args()

    transfer_validator(args)
