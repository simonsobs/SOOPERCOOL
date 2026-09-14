import numpy as np


def get_transfer_with_error_TM(pcls_mat_filt,
                               cl,
                               mcm,
                               nmt_binning,
                               compute_Dl=False):
    """
    Compute the transfer function and its associated error
    assuming that the binned filtered pseudo-Cl is given by

    \\tilde{C}_{b} = T_{b} P_{b\\ell} M_{\\ell,\\ell'} C_{\\ell'}

    Parameters
    ----------
    pcls_mat_filt : ndarray
        Filtered and binned pseudo-cells for different pure pairs
        and simulations
        Shape should be (N_sims, N_pure_pairs, N_field_pairs, N_bins)
    cl : ndarray
        Input power spectrum used to generate the simulations.
        Should contain any beam smoothing operation used to generate
        input simulations.
        Shape should be (N_ells)
    mcm: ndarray
        This is the full TEB mask mode-coupling matrix.
        Shape should be (N_field_pairs, N_ells, N_field_pairs, N_ells)
        i.e. (9, N_ells, 9, N_ells) for TEB.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    compute_Dl : bool, optional
        This is a dummy argument to be consistent with
        the MT version of this function.

    Returns
    -------
    T_mean : ndarray
        Transfer function for each field pair and bin.
        Shape should be (N_field_pairs, N_field_pairs, N_bins)
        Example:
            T_mean[0, 0, :] is the transfer function for TT->TT
            T_mean[-4, -1, :] is the transfer function for BB->EE
            i.e. the contribution of BB to EE due to filtering.
    T_std : ndarray
        Standard deviation of the transfer function
        Shape should be (N_field_pairs, N_field_pairs, N_bins)
    """
    cb_coupled_matrix = np.einsum(
        "CkBj,j->CBk",
        bin_mcm_left(mcm, nmt_binning),
        cl
    )
    # Flip first two axes to put pure_pairs first
    cb_coupled_matrix = np.transpose(
        cb_coupled_matrix,
        [1, 0, 2]
    )

    cct_inv = np.transpose(
        np.linalg.inv(
            np.transpose(
                np.einsum(
                    "jil,jkl->ikl",
                    cb_coupled_matrix,
                    cb_coupled_matrix
                ),
                axes=[2, 0, 1]
            )
        ), axes=[1, 2, 0]
    )
    T_mean = np.einsum(
        "ijl,jkl->ikl",
        cct_inv,
        np.einsum(
            "jil,jkl->ikl",
            cb_coupled_matrix,
            np.mean(pcls_mat_filt, axis=0)
        )
    )
    T_std = np.std(
        np.array(
            [np.einsum(
                "ijl,jkl->kil",
                cct_inv,
                np.einsum(
                    "jil,jkl->ikl",
                    cb_coupled_matrix,
                    clf
                )
            ) for clf in pcls_mat_filt]
        ), axis=0
    )

    return T_mean, T_std


def get_transfer_with_error_MT(pcls_mat_filt,
                               cl,
                               mcm,
                               nmt_binning,
                               compute_Dl=False):
    """
    Compute the transfer function and its associated error
    assuming that the binned filtered pseudo-Cl is given by

    \\tilde{C}_{b} = P_{b\\ell} M_{\\ell,\\ell'} T_{\\ell'} C_{\\ell'}

    Parameters
    ----------
    pcls_mat_filt : ndarray
        Filtered and binned pseudo-cells for different pure pairs
        and simulations
        Shape should be (N_sims, N_pure_pairs, N_field_pairs, N_bins)
    cl : ndarray
        Input power spectrum used to generate the simulations.
        Should contain any beam smoothing operation used to generate
        input simulations.
        Shape should be (N_ells)
    mcm: ndarray
        This is the full TEB mask mode-coupling matrix.
        Shape should be (N_field_pairs, N_ells, N_field_pairs, N_ells)
        i.e. (9, N_ells, 9, N_ells) for TEB.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    compute_Dl : bool, optional
        Whether to compute Cl or Dl. This should already be accounted
        for in the nmt_binning object but is needed here when
        we apply the right binning operator on mode coupling matrices.

    Returns
    -------
    T_mean : ndarray
        Transfer function for each field pair and bin.
        Shape should be (N_field_pairs, N_field_pairs, N_bins)
        Example:
            T_mean[0, 0, :] is the transfer function for TT->TT
            T_mean[-4, -1, :] is the transfer function for BB->EE
            i.e. the contribution of BB to EE due to filtering.
    T_std : ndarray
        Standard deviation of the transfer function
        Shape should be (N_field_pairs, N_field_pairs, N_bins)
    """
    nl = nmt_binning.lmax + 1
    n_bins = nmt_binning.get_n_bands()

    phi_idx = [
        nmt_binning.get_ell_list(i)
        for i in range(n_bins)
    ]

    # phiC will have shape(p:n_bins, j:nl)
    phiC = np.zeros((n_bins, nl))
    for i in range(n_bins):
        phiC[i, phi_idx[i]] = cl[phi_idx[i]]

    # mcm will have shape (D:n_pairs, i:nl, C:n_pairs, j:nl)
    # MphiC will have shape (D:n_pairs, i:nl, C:n_pairs, p:n_bins)
    MphiC = np.einsum(
        "DiCj,pj->DiCp",
        mcm,
        phiC
    )
    bMphiC = bin_mcm_left(MphiC, nmt_binning)

    # Invert the binned mcm
    mcm_binned = bin_mcm_left(mcm, nmt_binning)
    mcm_binned = bin_mcm_right(
        mcm_binned,
        nmt_binning,
        compute_Dl=compute_Dl
    )
    size, n_bins, _, _ = mcm_binned.shape
    mcm_binned_inv = np.linalg.inv(
        mcm_binned.reshape([size*n_bins, size*n_bins])
    ).reshape([size, n_bins, size, n_bins])

    # Write the D matrix
    # which should have shape (A:n_pairs, k:n_bins, C:n_pairs, p:n_bins)
    D = np.einsum(
        "AkDm,DmCp->AkCp",
        mcm_binned_inv,
        bMphiC
    )

    # pcls_mat_filt has shape (N_sims, N_pure_pairs, N_field_pairs, N_bins)
    # moving the N_pure_pairs to the last axis as we fix this in this method
    cls = np.transpose(
        pcls_mat_filt,
        [0, 2, 3, 1]
    )
    # cls are pseudo-cls.
    # We decouple them below
    cls = np.einsum(
        "AkBp,NBpC->NAkC",
        mcm_binned_inv,
        cls
    )

    cls_mean = np.mean(cls, axis=0)
    D = D.reshape([size*n_bins, size*n_bins])
    cls_mean = cls_mean.reshape([size*n_bins, size])

    D_inv = np.linalg.inv(D.T @ D) @ D.T

    T_mean = D_inv @ cls_mean
    T_mean = T_mean.reshape([size, n_bins, size])
    # Move last axis to second position to match
    # the default TF axis ordering
    T_mean = np.transpose(T_mean, [0, 2, 1])

    # now let's try to compute the error on the TF
    T_std = np.std(
        np.einsum(
            "ab,Nbc->Nac",
            D_inv,
            cls.reshape([-1, size*n_bins, size])
        ),
        axis=0
    ).reshape([size, n_bins, size])
    T_std = np.transpose(T_std, [0, 2, 1])

    return T_mean, T_std


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

    This is a legacy function. Not sure we still want to use
    it. Maybe remove in future commits.

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


def get_transfer_dict(pcls_mat_dict,
                      filtering_pairs,
                      cl,
                      mcm,
                      nmt_binning,
                      compute_Dl=False,
                      tf_ordering="TM"):
    """
    This is just a wrapper to loop over the filtering_pairs
    provided and compute the transfer function for each pair
    using the `get_transfer_with_error` functions.

    Parameters
    ----------
    pcls_mat_filt : ndarray
        Filtered and binned pseudo-cells for different pure pairs
        and simulations
        Shape should be (N_sims, N_pure_pairs, N_field_pairs, N_bins)
    filtering_pairs : list of tuples
        List of filtering tag pairs (ftag1, ftag2) for which to compute
    cl : ndarray
        Input power spectrum used to generate the simulations.
        Should contain any beam smoothing operation used to generate
        input simulations.
        Shape should be (N_ells)
    mcm: ndarray
        This is the full TEB mask mode-coupling matrix.
        Shape should be (N_field_pairs, N_ells, N_field_pairs, N_ells)
        i.e. (9, N_ells, 9, N_ells) for TEB.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    compute_Dl : bool, optional
        Whether to compute Cl or Dl. This should already be accounted
        for in the nmt_binning object but is needed here when
        we apply the right binning operator on mode coupling matrices.
    tf_ordering : str, optional
        Whether to use the TM or MT ordering in the definition
        of the transfer function.
        Default is "TM" and corresponds to the the approach
        described in https://arxiv.org/abs/2502.00946.
    Returns
    -------
    tf_dict : dict
        Dictionary with keys as (ftag1, ftag2) and values as another dict
        containing the transfer function and its error.
    """
    tf_dict = {(ftag1, ftag2): {} for ftag1, ftag2 in filtering_pairs}
    for ftag1, ftag2 in filtering_pairs:

        pcls_mat_filt = pcls_mat_dict[ftag1, ftag2]["filtered"]
        tf_func = (
            get_transfer_with_error_TM
            if tf_ordering == "TM"
            else get_transfer_with_error_MT
        )
        tf, tferr = tf_func(
            pcls_mat_filt,
            cl,
            mcm,
            nmt_binning,
            compute_Dl
        )
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


def bin_mcm_left(mcm, nmt_binning):
    """
    Bin the mode-coupling matrix on the left side
    using the provided mcm and the NmtBin object.
    If input mcm has shape (size, nl, size, nl),
    the output will have shape (size, n_bins, size, nl).

    Parameters
    ----------
    mcm : ndarray
        Mode-coupling matrix of shape (size, nl, size, ...)
        The second dimension should always be nl.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    """
    if mcm.ndim != 4:
        raise ValueError("Input mcm must be a 4D array.")

    nl = nmt_binning.lmax + 1
    if mcm.shape[1] != nl:
        raise ValueError(
            f"Input mcm second dimension must be nl={nl}, "
            f"but got {mcm.shape[1]}."
        )

    binner = np.array([
        nmt_binning.bin_cell(np.array([cl]))[0]
        for cl in np.eye(nl)
    ]).T

    # Resulting MCM will be (size, n_bins, size, nl)
    bmcm = np.einsum('ij,kjlm->kilm', binner, mcm)

    return bmcm


def bin_mcm_right(mcm, nmt_binning, compute_Dl=False):
    """
    Bin the mode-coupling matrix on the right side
    using the provided mcm and the NmtBin object.
    If input mcm has shape (size, n_bins, size, nl),
    the output will have shape (size, n_bins, size, n_bins).

    Parameters
    ----------
    mcm : ndarray
        Mode-coupling matrix of shape (size, ..., size, nl)
        The last dimension should always be nl.
    nmt_binning : NmtBin object
        Namaster binning scheme used to define the bandpowers.
    compute_Dl : bool, optional
        If True, applies the Dl conversion when computing the binned MCM.
        The code will then output power spectra in Dl units.
    """
    if mcm.ndim != 4:
        raise ValueError("Input mcm must be a 4D array.")
    nl = nmt_binning.lmax + 1
    if mcm.shape[-1] != nl:
        raise ValueError(
            f"Input mcm last dimension must be nl={nl}, "
            f"but got {mcm.shape[-1]}."
        )

    n_bins = nmt_binning.get_n_bands()
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

    bmcm = np.transpose(
        np.array([
            np.sum(
                mcm[:, :, :, ells_per_bin[i]] /
                cl2dl_per_bin[i][None, None, None, :],
                axis=-1
            )
            for i in range(n_bins)
        ]), axes=[1, 2, 3, 0]
    )

    return bmcm


def compute_couplings(mcm, nmt_binning,
                      transfer=None,
                      compute_Dl=False,
                      beam=None,
                      tf_ordering="TM"):
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
    tf_ordering: str, optional
        Ordering of the transfer function application.
        "TM" means transfer function is applied on the left side of the MCM.
        "MT" means transfer function is applied on the right side of the MCM
        which requires a more careful treatment. Default is "TM".

    Returns
    -------
    bpw_windows : ndarray
        Binned power window functions of shape (size, n_bins, size, nl).
    inv_coupling : ndarray
        Inverse binned mode-coupling matrix
        of shape (size, n_bins, size, n_bins).
    """
    mcm = mcm.copy()  # Avoid modifying the input array in place

    n_bins = nmt_binning.get_n_bands()
    size, nl, _, _ = mcm.shape

    # Beam the MCM if a beam is provided.
    if beam is not None:
        mcm *= beam[np.newaxis, :, np.newaxis, :]

    if tf_ordering == "TM":
        # Bin the MCM on the left
        mcm = bin_mcm_left(mcm, nmt_binning)

        # If there is a transfer function,
        # apply it to the MCM on the left
        if transfer is not None:
            size_tf, _, n_bins_tf = transfer.shape
            if size != size_tf:
                raise ValueError(
                    "MCM and transfer function have "
                    "incompatible field dimensions"
                )
            tmcm = np.einsum(
                'ijk,jklm->iklm',
                transfer,
                mcm
            )
        else:
            tmcm = mcm

        couplings = bin_mcm_right(
            tmcm,
            nmt_binning,
            compute_Dl=compute_Dl
        )
        inv_couplings = np.linalg.inv(
            couplings.reshape([
                size*n_bins,
                size*n_bins
            ])
        )

        bpw_windows = np.dot(
            inv_couplings,
            tmcm.reshape([size*n_bins, size*nl])
        )

        inv_couplings = inv_couplings.reshape([
            size, n_bins, size, n_bins
        ])
        bpw_windows = bpw_windows.reshape([
            size, n_bins, size, nl
        ])

    elif tf_ordering == "MT":
        # Now we assume that the transfer function is
        # applied first in our model of the filtered
        # pseudo-cls.
        # TODO: generalize this, as we assume below
        # that the transfer function T_\ell is defined
        # as a step function in bandpowers. If we want to
        # generalize this, we need to change the definition
        # of phi below.
        phiT = np.zeros([size, size, nl])

        for i in range(n_bins):
            idx = nmt_binning.get_ell_list(i)

            if transfer is not None:
                # this is to play with
                # numpy broadcasting
                phiT[:, :, idx] = transfer[:, :, i][:, :, None]
        if transfer is None:
            phiT[:, :, :] = np.eye(size, size)[:, :, None]
        mcm_phiT = np.einsum(
            "AiCj,CBj->AiBj",
            mcm,
            phiT
        )
        couplings = bin_mcm_left(
            mcm_phiT,
            nmt_binning
        )
        couplings = bin_mcm_right(
            couplings,
            nmt_binning,
            compute_Dl=compute_Dl
        )
        inv_couplings = np.linalg.inv(
            couplings.reshape([size*n_bins, size*n_bins])
        ).reshape([size, n_bins, size, n_bins])

        mcm_phiT_bin_left = bin_mcm_left(
            mcm_phiT,
            nmt_binning
        )
        bpw_windows = np.einsum(
            "AkCp,CpBi->AkBi",
            inv_couplings,
            mcm_phiT_bin_left
        )

    return bpw_windows, inv_couplings


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
