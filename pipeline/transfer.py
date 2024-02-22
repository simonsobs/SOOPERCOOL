import argparse
import numpy as np
from soopercool import BBmeta


# Move everything to `coupling_utils.py`
def get_transfer_with_error(mean_pcls_mat_filt,
                            mean_pcls_mat_unfilt,
                            pcls_mat_filt):
    """
    """
    cct_inv = np.transpose(
        np.linalg.inv(
            np.transpose(
                np.einsum('jil,jkl->ikl',
                          mean_pcls_mat_unfilt,
                          mean_pcls_mat_unfilt),
                axes=[2, 0, 1]
            )
        ), axes=[1, 2, 0]
    )

    tf = np.einsum(
        'ijl,jkl->kil', cct_inv,
        np.einsum(
            'jil,jkl->ikl',
            mean_pcls_mat_unfilt,
            mean_pcls_mat_filt
        )
    )

    tferr = np.std(
        np.array(
            [np.einsum(
                'ijl,jkl->kil', cct_inv,
                np.einsum(
                    'jil,jkl->ikl',
                    mean_pcls_mat_unfilt,
                    clf))
                for clf in pcls_mat_filt]
        ), axis=0
    )

    return tf, tferr


def get_transfer_dict(mean_pcls_mat_filt_dict,
                      mean_pcls_mat_unfilt_dict,
                      pcls_mat_dict,
                      filtering_pairs,
                      spin_pairs):
    """
    """
    tf_dict = {(ftag1, ftag2): {} for ftag1, ftag2 in filtering_pairs}
    tf_std_dict = {(ftag1, ftag2): {} for ftag1, ftag2 in filtering_pairs}
    for ftag1, ftag2 in filtering_pairs:
        for spin_pair in spin_pairs:

            mean_pcls_mat_filt = \
                mean_pcls_mat_filt_dict[ftag1, ftag2][spin_pair]
            mean_pcls_mat_unfilt = \
                mean_pcls_mat_unfilt_dict[ftag1, ftag2][spin_pair]
            pcls_mat_filt = \
                pcls_mat_dict[ftag1, ftag2][spin_pair]["filtered"]

            tf, tferr = get_transfer_with_error(mean_pcls_mat_filt,
                                                mean_pcls_mat_unfilt,
                                                pcls_mat_filt)

            tf_dict[ftag1, ftag2][spin_pair] = tf
            tf_std_dict[ftag1, ftag2][spin_pair] = tferr
    return tf_dict, tf_std_dict


def read_pcls_matrices(pcls_mat_dir, filtering_pairs, spin_pairs, Nsims):
    """
    """
    pcls_mat_dict = {
        (ftag1, ftag2): {
            spin_pair: {
                "filtered": [],
                "unfiltered": []
            } for spin_pair in spin_pairs
        } for ftag1, ftag2 in filtering_pairs
    }

    # Load the pseudo-cl matrices for each simulation
    # Should be (n_comb_pure, n_comb_mode, n_bins)
    for id_sim in range(Nsims):
        for label in ["filtered", "unfiltered"]:
            for ftag1, ftag2 in filtering_pairs:
                suffix = f"{ftag1}x{ftag2}_{label}_{id_sim:04d}"
                pcls_mat = np.load(
                    f"{pcls_mat_dir}/pcls_mat_tf_est_{suffix}.npz")
                for spin_pair in spin_pairs:
                    pcls_mat_dict[ftag1, ftag2][spin_pair][label] += \
                        [pcls_mat[spin_pair]]

    return pcls_mat_dict


def load_mcms(coupling_dir, spin_pairs, ps_names=None, B_pure=False):
    """
    """

    file_root = "mcm" if not B_pure else "mcm_pure"
    mcms_dict = {}

    if ps_names is None:
        mcm_file = f"{coupling_dir}/{file_root}.npz"
        mcm = np.load(mcm_file)
        for spin_pair in spin_pairs:
            mcms_dict[spin_pair] = mcm[f"{spin_pair}_binned"]
    else:
        for ms1, ms2 in ps_names:
            mcm_file = f"{coupling_dir}/{file_root}_{ms1}_{ms2}.npz"
            mcm = np.load(mcm_file)
            mcms_dict[ms1, ms2] = {}
            for spin_pair in spin_pairs:
                mcms_dict[ms1, ms2][spin_pair] = mcm[f"{spin_pair}_binned"]
    return mcms_dict


def average_pcls_matrices(pcls_mat_dict, filtering_pairs,
                          spin_pairs, filtered):
    """
    """
    label = "filtered" if filtered else "unfiltered"
    pcls_mat_mean = {
        (ftag1, ftag2): {
            spin_pair: np.mean(
                pcls_mat_dict[ftag1, ftag2][spin_pair][label],
                axis=0
            )
            for spin_pair in spin_pairs
        } for ftag1, ftag2 in filtering_pairs
    }

    return pcls_mat_mean


def compute_couplings(mcm, nmt_binning, transfer=None):
    """
    """

    size, n_bins, _, nl = mcm.shape
    if transfer is not None:
        tmcm = np.einsum('ijk,jklm->iklm',
                         transfer,
                         mcm)
    else:
        tmcm = mcm

    btmcm = np.transpose(
        np.array([
            np.sum(tmcm[:, :, :, nmt_binning.get_ell_list(i)],
                   axis=-1)
            for i in range(n_bins)
        ]), axes=[1, 2, 3, 0]
    )

    inv_btmcm = np.linalg.inv(
        btmcm.reshape([size*n_bins, size*n_bins])
    )
    winflat = np.dot(inv_btmcm, tmcm.reshape([size*n_bins, size*nl]))
    inv_coupling = inv_btmcm.reshape([size, n_bins, size, n_bins])
    bpw_windows = winflat.reshape([size, n_bins, size, nl])

    return bpw_windows, inv_coupling


def get_couplings_dict(mcm_dict, nmt_binning, spin_pairs,
                       transfer_dict=None,
                       ps_names_and_ftags=None,
                       filtering_pairs=None):
    """
    """
    couplings = {}

    if ps_names_and_ftags is not None:
        for (ms1, ms2), (ftag1, ftag2) in ps_names_and_ftags.items():

            couplings[ms1, ms2] = {}

            for spin_pair in spin_pairs:
                mcm = mcm_dict[ms1, ms2][spin_pair]
                if transfer_dict is not None:
                    transfer = transfer_dict[ftag1, ftag2][spin_pair]
                else:
                    transfer = None

                bpw_win, inv_coupling = compute_couplings(
                    mcm, nmt_binning, transfer
                )
                couplings[ms1, ms2][f"bp_win_{spin_pair}"] = bpw_win
                couplings[ms1, ms2][f"inv_coupling_{spin_pair}"] = inv_coupling

    else:
        for ftag1, ftag2 in filtering_pairs:
            couplings[ftag1, ftag2] = {}
            for spin_pair in spin_pairs:
                mcm = mcm_dict[ftag1, ftag2][spin_pair]
                if transfer_dict is not None:
                    transfer = transfer_dict[ftag1, ftag2][spin_pair]
                else:
                    transfer = None

                bpw_win, inv_coupling = compute_couplings(
                    mcm, nmt_binning, transfer
                )
                couplings[ftag1, ftag2][f"bp_win_{spin_pair}"] = bpw_win
                couplings[ftag1, ftag2][
                    f"inv_coupling_{spin_pair}"] = inv_coupling

    return couplings


def transfer(args):
    """
    """
    meta = BBmeta(args.globals)

    spin_pairs = [
        "spin0xspin0",
        "spin0xspin2",
        "spin2xspin2"
    ]

    filtering_pairs = meta.get_independent_filtering_pairs()

    cl_dir = meta.cell_transfer_directory
    coupling_dir = meta.coupling_directory

    nmt_binning = meta.read_nmt_binning()

    meta.timer.start("Mean sims")
    pcls_mat_dict = read_pcls_matrices(cl_dir, filtering_pairs,
                                       spin_pairs, meta.tf_est_num_sims)

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = average_pcls_matrices(
        pcls_mat_dict,
        filtering_pairs,
        spin_pairs,
        filtered=True
    )
    pcls_mat_unfiltered_mean = average_pcls_matrices(
        pcls_mat_dict,
        filtering_pairs,
        spin_pairs,
        filtered=False
    )
    meta.timer.stop("Mean sims")

    meta.timer.start("Load mode-coupling matrices")
    # Load the beam corrected mode coupling matrices
    ps_names = meta.get_ps_names_list("all", coadd=True)
    mcms_dict = load_mcms(coupling_dir, spin_pairs, ps_names=ps_names)

    # Load the un-beamed mode coupling matrix
    mcms_dict_val = load_mcms(coupling_dir,
                              spin_pairs,
                              ps_names=filtering_pairs)

    # If we B-purify for transfer function estimation, load the un-beamed
    # purified mode coupling matrix
    if meta.tf_est_pure_B:
        mcms_dict_val_pure = load_mcms(coupling_dir,
                                       spin_pairs,
                                       ps_names=filtering_pairs,
                                       B_pure=True)
    meta.timer.stop("Load mode-coupling matrices", verbose=True)

    # Compute and save the transfer functions
    trans, etrans = get_transfer_dict(
        pcls_mat_filtered_mean,
        pcls_mat_unfiltered_mean,
        pcls_mat_dict,
        filtering_pairs,
        spin_pairs
    )
    for ftag1, ftag2 in filtering_pairs:
        tf = trans[ftag1, ftag2]
        tferr = etrans[ftag1, ftag2]
        np.savez(
            f"{coupling_dir}/transfer_function_{ftag1}x{ftag2}.npz",
            tf_spin0xspin0=tf["spin0xspin0"],
            tf_spin0xspin2=tf["spin0xspin2"],
            tf_spin2xspin2=tf["spin2xspin2"],
            tf_std_spin0xspin0=tferr["spin0xspin0"],
            tf_std_spin0xspin2=tferr["spin0xspin2"],
            tf_std_spin2xspin2=tferr["spin2xspin2"]
        )

    # Compute and save couplings for different cases
    meta.timer.start("Compute full coupling")
    # First for the data (i.e. including beams, tf, and mcm)
    ps_names_and_ftags = {
        (ms1, ms2): (meta.filtering_tag_from_map_set(ms1),
                     meta.filtering_tag_from_map_set(ms2))
        for ms1, ms2 in ps_names
    }
    couplings = get_couplings_dict(
        mcms_dict, nmt_binning, spin_pairs,
        transfer_dict=trans,
        ps_names_and_ftags=ps_names_and_ftags
    )

    for ms1, ms2 in ps_names:
        np.savez(
            f"{coupling_dir}/couplings_{ms1}_{ms2}.npz",
            **couplings[ms1, ms2]
        )

    # Then for validation simulations (i.e. no beams, with/without tf)

    couplings_val_with_tf = get_couplings_dict(
        mcms_dict_val, nmt_binning, spin_pairs,
        transfer_dict=trans,
        filtering_pairs=filtering_pairs
    )
    couplings_val_without_tf = get_couplings_dict(
        mcms_dict_val, nmt_binning, spin_pairs,
        filtering_pairs=filtering_pairs
    )

    for ftag1, ftag2 in filtering_pairs:
        np.savez(
            f"{coupling_dir}/couplings_{ftag1}x{ftag2}_filtered.npz",
            **couplings_val_with_tf[ftag1, ftag2]
        )
        np.savez(
            f"{coupling_dir}/couplings_{ftag1}x{ftag2}_unfiltered.npz",
            **couplings_val_without_tf[ftag1, ftag2]
        )

    # Finally an optional case for pure B tf estimation
    if meta.tf_est_pure_B:
        couplings_val_with_tf_pure = get_couplings_dict(
            mcms_dict_val_pure, nmt_binning, spin_pairs,
            transfer_dict=trans,
            filtering_pairs=filtering_pairs
        )
        couplings_val_without_tf_pure = get_couplings_dict(
            mcms_dict_val_pure, nmt_binning, spin_pairs,
            filtering_pairs=filtering_pairs
        )

        for ftag1, ftag2 in filtering_pairs:
            ftag_label = f"{ftag1}x{ftag2}"
            np.savez(
                f"{coupling_dir}/couplings_{ftag_label}_filtered_pure.npz",
                **couplings_val_with_tf_pure[ftag1, ftag2]
            )
            np.savez(
                f"{coupling_dir}/couplings_{ftag_label}_unfiltered_pure.npz",
                **couplings_val_without_tf_pure
            )
    meta.timer.stop("Compute full coupling", verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer function stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    args = parser.parse_args()
    transfer(args)
