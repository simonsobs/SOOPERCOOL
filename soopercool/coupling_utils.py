import numpy as np


def get_transfer_with_error(mean_pcls_mat_filt,
                            mean_pcls_mat_unfilt,
                            pcls_mat_filt):
    """
    Given two matrices filled with filtered
    and unfiltered pseudo-cls averaged
    over several realizations, compute the
    transfer function. The associated statistical
    error is computed from the scatter measured
    accross realizations.

    N_pure_pairs = len(["pureTxpureT", "pureTxpureE", ...])
    N_field_pairs = len(["TT", "TE", ...])
    N_bins = number of bandpower bins

    Parameters
    ----------
    mean_pcls_mat_filt : ndarray
        Matrix of shape (N_pure_pairs, N_field_pairs, N_bins)
        containing the mean pseudo-cls for the filtered simulations.
    mean_pcls_mat_unfilt : ndarray
        Matrix of shape (N_pure_pairs, N_field_pairs, N_bins)
        containing the mean pseudo-cls for the unfiltered simulations.
    pcls_mat_filt : ndarray
        Matrix of shape (N_sims, N_pure_pairs, N_field_pairs, N_bins)
        containing the pseudo-cls for the filtered simulations.

    Returns
    -------
    tf : ndarray
        Matrix of shape (N_field_pairs, N_field_pairs, N_bins)
        containing the transfer function for each field pair and bin.
    tferr : ndarray
        Matrix of shape (N_field_pairs, N_field_pairs, N_bins)
        containing statistical errors on the TF.
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
                      filtering_pairs):
    """
    This is just a wrapper to loop over the filtering_pairs
    provided and compute the transfer function for each pair
    using the `get_transfer_with_error` function.

    Parameters
    ----------
    mean_pcls_mat_filt_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as the mean
        pseudo-cl matrices for the filtered simulations.
    mean_pcls_mat_unfilt_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as the mean
        pseudo-cl matrices for the unfiltered simulations.
    pcls_mat_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as the pseudo-cl
        matrices for the filtered simulations for each realizations.
    filtering_pairs : list of tuples
        List of filtering tag pairs (ftag1, ftag2) for which to compute

    Returns
    -------
    tf_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as another dict
        containing the transfer function and its error.
    """
    tf_dict = {(ftag1, ftag2): {} for ftag1, ftag2 in filtering_pairs}
    for ftag1, ftag2 in filtering_pairs:

        mean_pcls_mat_filt = \
            mean_pcls_mat_filt_dict[ftag1, ftag2]
        mean_pcls_mat_unfilt = \
            mean_pcls_mat_unfilt_dict[ftag1, ftag2]
        pcls_mat_filt = \
            pcls_mat_dict[ftag1, ftag2]["filtered"]

        tf, tferr = get_transfer_with_error(mean_pcls_mat_filt,
                                            mean_pcls_mat_unfilt,
                                            pcls_mat_filt)
        field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

        for i, fp1 in enumerate(field_pairs):
            for j, fp2 in enumerate(field_pairs):
                tf_dict[ftag1, ftag2][f"{fp2}_to_{fp1}"] = tf[i, j]
                tf_dict[ftag1, ftag2][f"{fp2}_to_{fp1}_std"] = tferr[i, j]
        tf_dict[ftag1, ftag2]["full_tf"] = tf

    return tf_dict


def read_pcls_matrices(pcls_mat_dir, filtering_pairs, Nsims, sim_id_start=0):
    """
    Utility function to read pseudo cls matrices from disk and organize
    them in a dictionary for easy access.

    Parameters
    ----------
    pcls_mat_dir : str
        Directory where the pseudo-cl matrices are stored.
    filtering_pairs : list of tuples
        List of filtering tag pairs (ftag1, ftag2) for which to
        read the pseudo-cl matrices.
    Nsims : int
        Number of simulations for which to read the pseudo-cl matrices.
    sim_id_start : int, optional
        Starting index for the simulation IDs. Default is 0.

    Returns
    -------
    pcls_mat_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as another dict
        containing the pseudo-cl matrices for the filtered and unfiltered
        simulations.
    """
    pcls_mat_dict = {
        (ftag1, ftag2): {
                "filtered": [],
                "unfiltered": []
        } for ftag1, ftag2 in filtering_pairs
    }

    # Load the pseudo-cl matrices for each simulation
    # Should be (n_comb_pure, n_comb_mode, n_bins)
    for id_sim in range(sim_id_start, Nsims + sim_id_start):
        for label in ["filtered", "unfiltered"]:
            for ftag1, ftag2 in filtering_pairs:
                lab1 = f"{ftag1[0]}_{ftag1[1]}"
                lab2 = f"{ftag2[0]}_{ftag2[1]}"
                suffix = f"{lab1}_x_{lab2}_{label}_{id_sim:04d}"
                pcls_mat = np.load(
                    f"{pcls_mat_dir}/pcls_mat_tf_est_{suffix}.npz")
                pcls_mat_dict[ftag1, ftag2][label] += [pcls_mat["pcls_mat"]]

    return pcls_mat_dict


def read_mcm(mcm_file, full_mcm=False):
    """
    Utility function to read the mode-coupling matrix
    from disk and organize it in a dictionary for easy access.
    Alternatively, if `full_mcm` is True, returns the full
    MCM as a single array with shape (9, nl, 9, nl)

    Parameters
    ----------
    mcm_file : str
        Path to the .npz file containing the MCM.
    full_mcm : bool, optional
        If True, returns the full MCM as a single array of shape
        (9, nl, 9, nl). If False, returns a dictionary with keys
        "spin0xspin0", "spin0xspin2", "spin2xspin2" and values
        as the corresponding MCM blocks. Default is False.
    """
    mcm = np.load(mcm_file)
    _, nl, _, nl = mcm["spin0xspin0"].shape
    if full_mcm:
        full_mcm = np.zeros((9, nl, 9, nl))
        full_mcm[0, :, 0, :] = mcm["spin0xspin0"][0, :, 0, :]
        full_mcm[1:3, :, 1:3, :] = mcm["spin0xspin2"]
        full_mcm[3:5, :, 3:5, :] = mcm["spin0xspin2"]
        full_mcm[5:, :, 5:, :] = mcm["spin2xspin2"]
        return full_mcm
    else:
        return {
            "spin0xspin0": mcm["spin0xspin0"],
            "spin0xspin2": mcm["spin0xspin2"],
            "spin2xspin2": mcm["spin2xspin2"]
        }


def average_pcls_matrices(pcls_mat_dict, filtering_pairs,
                          filtered):
    """
    Utility function to average the pseudo-cl matrices over simulations
    for each pair of filtering tags.

    Parameters
    ----------
    pcls_mat_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as another dict
        containing the pseudo-cl matrices for the filtered and unfiltered
        simulations.
    filtering_pairs : list of tuples
        List of filtering tag pairs (ftag1, ftag2) for which to
        average the pseudo-cl matrices.
    filtered : bool
        If True, averages the filtered pseudo-cl matrices. If False, averages
        the unfiltered pseudo-cl matrices.

    Returns
    -------
    pcls_mat_mean : dict
        Dictionary with keys as (ftag1, ftag2) and values as the averaged
        pseudo-cl matrices.
    """
    label = "filtered" if filtered else "unfiltered"
    pcls_mat_mean = {
        (ftag1, ftag2): np.mean(
                pcls_mat_dict[ftag1, ftag2][label],
                axis=0
            )
        for ftag1, ftag2 in filtering_pairs}

    return pcls_mat_mean


def load_transfer_function(transfer_dir, ms1, ms2,
                           ftag_from_map_set,
                           ktag_from_map_set,
                           nmt_bins):
    """
    Load the transfer function for a given pair of map sets.

    Parameters
    ----------
    transfer_dir : str
        Directory where the transfer function files are stored.
    ms1, ms2 : str
        Map set names
    ftag_from_map_set : function
        Function that takes a map set name and returns the corresponding
        filtering tag.
    ktag_from_map_set : function
        Function that takes a map set name and returns the corresponding
        k-space filtering tag.
    nmt_bins : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    """
    ftag1 = ftag_from_map_set(ms1)
    ftag2 = ftag_from_map_set(ms2)
    ktag1 = ktag_from_map_set(ms1)
    ktag2 = ktag_from_map_set(ms2)

    # If no filtering, no need to complicate our
    # lives with transfer function-related steps.
    if ftag1 is None and ftag2 is None:
        if ktag1 is None or ktag2 is None:
            tf_unity = np.zeros((9, 9, nmt_bins.get_n_bands()))
            for i in range(9):
                tf_unity[i, i, :] = 1.0
            return tf_unity

    lab1 = f"{ftag1}_{ktag1}"
    lab2 = f"{ftag2}_{ktag2}"
    tf_fname = f"{transfer_dir}/transfer_function_{lab1}_x_{lab2}.npz"
    return np.load(tf_fname)["full_tf"]


def compute_couplings(mcm, nmt_binning,
                      transfer=None,
                      compute_Dl=False,
                      beam=None):
    """
    Compute couplings from pre-computed mode-coupling
    matrices `mcm` and optional transfer functions.

    Parameters
    ----------
    mcm : ndarray
        Mode-coupling matrix of shape (size, n_bins, size, nl), where
        `size` is the number of field combinations (e.g., 9 for TEBxTEB),
        `n_bins` is the number of bandpower bins,
        and `nl` is the maximum multipole.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    transfer : ndarray, optional
        Transfer function of shape (size, size, n_bins) to apply to the MCM.
    compute_Dl : bool, optional
        If True, applies the Dl conversion when computing the binned MCM.
        The code will then output power spectra in Dl units.
    beam: ndarray, optional
        Beam function (squared or two maps) of shape (nl, nl)
        to apply to the MCM.

    Returns
    -------
    bpw_windows : ndarray
        Binned power window functions of shape (size, n_bins, size, nl).
    inv_coupling : ndarray
        Inverse binned mode-coupling matrix
        of shape (size, n_bins, size, n_bins).
    """
    # Beam the MCM if a beam is provided.
    if beam is not None:
        mcm *= beam[np.newaxis, :, np.newaxis, :]

    # Bin the MCM on one side
    nl = mcm.shape[-1]
    binner = np.array([
        nmt_binning.bin_cell(np.array([cl]))[0]
        for cl in np.eye(nl)
    ]).T

    # Resulting MCM will be (size, n_bins, size, nl)
    mcm = np.einsum('ij,kjlm->kilm', binner, mcm)
    size, n_bins, _, nl = mcm.shape

    # If there is a transfer function,
    # apply it to the MCM on the left
    if transfer is not None:
        n_bins_nmt = nmt_binning.get_n_bands()
        nl_nmt = nmt_binning.lmax + 1
        size_tf, _, n_bins_tf = transfer.shape
        if size != size_tf:
            raise ValueError(
                "MCM and transfer function have incompatible field dimensions"
            )
        n_bins = min(n_bins, n_bins_tf, n_bins_nmt)
        nl = min(nl, nl_nmt, nmt_binning.get_ell_max(n_bins-1)+1)
        tmcm = np.einsum('ijk,jklm->iklm',
                         transfer[:, :, :n_bins],
                         mcm[:, :n_bins, :, :nl])
    else:
        tmcm = mcm

    # We then bin the TFxMCM on the right
    # side to get the binned MCM
    ells_per_bin = [
        nmt_binning.get_ell_list(i)
        for i in range(n_bins)
    ]
    if compute_Dl:
        cl2dl_per_bin = []
        for i in range(n_bins):
            cl2dl_per_bin.append(
                ells_per_bin[i] * (ells_per_bin[i] + 1) / 2 / np.pi
            )
            # Regularize to avoid division by zero for l=0
            cl2dl_per_bin[i][cl2dl_per_bin[i] == 0] = np.inf
    else:
        cl2dl_per_bin = [np.ones_like(ells_per_bin[i]) for i in range(n_bins)]

    btmcm = np.transpose(
        np.array([
            np.sum(
                tmcm[:, :, :, ells_per_bin[i]] /
                cl2dl_per_bin[i][None, None, None, :],
                axis=-1
            )
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


def get_couplings_dict(mcm_dict, nmt_binning,
                       transfer_dict=None,
                       ps_names_and_ftags=None,
                       compute_Dl=False):
    """
    Compute couplings from pre-computed mode-coupling
    matrices `mcm_dict` and optional transfer functions.
    This loops over either ps_names_and_ftags or filtering_pairs
    to compute the couplings for each pair.

    Parameters
    ----------
    mcm_dict : dict
        Dictionary of mode-coupling matrices of shape
        (size, n_bins, size, nl), where `size` is the
        number of field combinations (e.g., 9 for TEBxTEB),
        `n_bins` is the number of bandpower bins,
        and `nl` is the maximum multipole.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    transfer_dict : dict, optional
        Dictionary of transfer functions of shape
        (size, n_bins, n_bins) to apply to the MCM.
    ps_names_and_ftags : dict, optional
        Dictionary with keys as (map_set1, map_set2) and
        values as (ftag1, ftag2).
        If provided, couplings will be computed for these pairs.
    compute_Dl : bool, optional
        If True, applies the Dl conversion when computing the binned MCM.
        The code will then output power spectra in Dl units.
    """
    couplings = {}

    if ps_names_and_ftags is not None:
        for (ms1, ms2), (ftag1, ftag2) in ps_names_and_ftags.items():

            couplings[ms1, ms2] = {}

            mcm = mcm_dict[ms1, ms2]
            if transfer_dict is not None:
                transfer = transfer_dict[ftag1, ftag2]
            else:
                transfer = None

            bpw_win, inv_coupling = compute_couplings(
                mcm, nmt_binning, transfer, compute_Dl=compute_Dl
            )
            couplings[ms1, ms2]["bp_win"] = bpw_win
            couplings[ms1, ms2]["inv_coupling"] = inv_coupling

    else:
        raise ValueError(
            "Nothing computed. "
            "Should provide a ps_names_and_ftags argument."
        )

    return couplings
