import argparse
import numpy as np
from soopercool import BBmeta
from soopercool import coupling_utils as cu


def transfer(args):
    """ """
    meta = BBmeta(args.globals)

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin2"]

    filtering_pairs = meta.get_independent_filtering_pairs()

    cl_dir = meta.cell_transfer_directory
    coupling_dir = meta.coupling_directory

    nmt_binning = meta.read_nmt_binning()

    meta.timer.start("Mean sims")
    pcls_mat_dict = cu.read_pcls_matrices(
        cl_dir, filtering_pairs, spin_pairs, meta.tf_est_num_sims
    )

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = cu.average_pcls_matrices(
        pcls_mat_dict, filtering_pairs, spin_pairs, filtered=True
    )
    pcls_mat_unfiltered_mean = cu.average_pcls_matrices(
        pcls_mat_dict, filtering_pairs, spin_pairs, filtered=False
    )
    meta.timer.stop("Mean sims")

    meta.timer.start("Load mode-coupling matrices")
    # Load the beam corrected mode coupling matrices
    # used for the data and cov sims
    ps_names = meta.get_ps_names_list("all", coadd=True)
    mcms_dict = cu.load_mcms(coupling_dir, spin_pairs, ps_names=ps_names, full_mcm=True)

    # Load the un-beamed mode coupling matrix
    # for the TF validation steps, including
    # the beam according to the filtering tag
    mcms_dict_val = cu.load_mcms(
        coupling_dir, spin_pairs, ps_names=filtering_pairs, full_mcm=True
    )

    meta.timer.stop("Load mode-coupling matrices", verbose=True)

    # Compute and save the transfer functions
    trans = cu.get_transfer_dict(
        pcls_mat_filtered_mean,
        pcls_mat_unfiltered_mean,
        pcls_mat_dict,
        filtering_pairs,
        spin_pairs,
    )
    full_tf = {}
    for ftag1, ftag2 in filtering_pairs:
        tf = trans[ftag1, ftag2]
        np.savez(f"{coupling_dir}/transfer_function_{ftag1}x{ftag2}.npz", **tf)
        full_tf[ftag1, ftag2] = tf["full_tf"]

    # Compute and save couplings for different cases
    meta.timer.start("Compute full coupling")
    # First for the data (i.e. including beams, tf, and mcm)
    ps_names_and_ftags = {
        (ms1, ms2): (
            meta.filtering_tag_from_map_set(ms1),
            meta.filtering_tag_from_map_set(ms2),
        )
        for ms1, ms2 in ps_names
    }
    couplings = cu.get_couplings_dict(
        mcms_dict,
        nmt_binning,
        spin_pairs,
        transfer_dict=full_tf,
        ps_names_and_ftags=ps_names_and_ftags,
    )

    for ms1, ms2 in ps_names:
        np.savez(f"{coupling_dir}/couplings_{ms1}_{ms2}.npz", **couplings[ms1, ms2])

    # Then for validation simulations
    # (i.e. with/without beams, with/without tf)
    couplings_val_with_tf = cu.get_couplings_dict(
        mcms_dict_val,
        nmt_binning,
        spin_pairs,
        transfer_dict=full_tf,
        filtering_pairs=filtering_pairs,
    )
    couplings_val_without_tf = cu.get_couplings_dict(
        mcms_dict_val, nmt_binning, spin_pairs, filtering_pairs=filtering_pairs
    )

    for ftag1, ftag2 in filtering_pairs:
        np.savez(
            f"{coupling_dir}/couplings_{ftag1}x{ftag2}_filtered.npz",
            **couplings_val_with_tf[ftag1, ftag2],
        )
        np.savez(
            f"{coupling_dir}/couplings_{ftag1}x{ftag2}_unfiltered.npz",
            **couplings_val_without_tf[ftag1, ftag2],
        )

    meta.timer.stop("Compute full coupling", verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer function stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    args = parser.parse_args()
    transfer(args)
