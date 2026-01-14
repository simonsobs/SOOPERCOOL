import numpy as np
import os


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
                      filtering_pairs):
    """
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
                suffix = f"{ftag1}_x_{ftag2}_{label}_{id_sim:04d}"
                pcls_mat = np.load(
                    f"{pcls_mat_dir}/pcls_mat_tf_est_{suffix}.npz")
                pcls_mat_dict[ftag1, ftag2][label] += [pcls_mat["pcls_mat"]]

    return pcls_mat_dict


def load_mcms(coupling_dir, ps_names=None, full_mcm=False):
    """
    """
    file_root = "mcm"
    mcms_dict = {}

    if ps_names is None:
        mcm_file = f"{coupling_dir}/{file_root}.npz"
        mcm = read_mcm(mcm_file, binned=True, full_mcm=full_mcm)
        return mcm
    else:
        for ms1, ms2 in ps_names:
            mcm_file = f"{coupling_dir}/{file_root}_{ms1}_{ms2}.npz"
            mcm_file_swap = f"{coupling_dir}/{file_root}_{ms2}_{ms1}.npz"
            if not os.path.isfile(mcm_file):
                if os.path.isfile(mcm_file_swap):
                    raise FileNotFoundError(
                        f"It seems the order of map sets ({ms1}, {ms2}) was "
                        "for the MCMs was swapped. Please recompute the "
                        "correct couplings.")
                else:
                    raise FileNotFoundError(
                        f"Mode coupling doesn't exist: {mcm_file}"
                    )
            mcm = read_mcm(mcm_file, binned=True, full_mcm=full_mcm)
            mcms_dict[ms1, ms2] = mcm
        return mcms_dict


def read_mcm(mcm_file, binned=False, full_mcm=False):
    """
    """
    mcm = np.load(mcm_file)
    suffix = "_binned" if binned else ""
    _, n_bins, _, nl = mcm[f"spin0xspin0{suffix}"].shape
    if full_mcm:
        full_mcm = np.zeros((9, n_bins, 9, nl))
        full_mcm[0, :, 0, :] = mcm[f"spin0xspin0{suffix}"][0, :, 0, :]
        full_mcm[1:3, :, 1:3, :] = mcm[f"spin0xspin2{suffix}"]
        full_mcm[3:5, :, 3:5, :] = mcm[f"spin0xspin2{suffix}"]
        full_mcm[5:, :, 5:, :] = mcm[f"spin2xspin2{suffix}"]
        return full_mcm
    else:
        return {
            "spin0xspin0": mcm[f"spin0xspin0{suffix}"],
            "spin0xspin2": mcm[f"spin0xspin2{suffix}"],
            "spin2xspin2": mcm[f"spin2xspin2{suffix}"]
        }


def average_pcls_matrices(pcls_mat_dict, filtering_pairs,
                          filtered):
    """
    """
    label = "filtered" if filtered else "unfiltered"
    pcls_mat_mean = {
        (ftag1, ftag2): np.mean(
                pcls_mat_dict[ftag1, ftag2][label],
                axis=0
            )
        for ftag1, ftag2 in filtering_pairs}

    return pcls_mat_mean


def compute_couplings(mcm, nmt_binning, transfer=None, compute_Dl=False):
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
        Transfer function of shape (size, n_bins, n_bins) to apply to the MCM.
    compute_Dl : bool, optional
        If True, applies the Dl conversion when computing the binned MCM.
        The code will then output power spectra in Dl units.

    Returns
    -------
    bpw_windows : ndarray
        Binned power window functions of shape (size, n_bins, size, nl).
    inv_coupling : ndarray
        Inverse binned mode-coupling matrix
        of shape (size, n_bins, size, n_bins).
    """
    size, n_bins, _, nl = mcm.shape
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
                       filtering_pairs=None,
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
    filtering_pairs : list of tuples, optional
        List of (ftag1, ftag2) tuples. If provided, couplings
        will be computed for these pairs.
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
        for ftag1, ftag2 in filtering_pairs:
            couplings[ftag1, ftag2] = {}
            mcm = mcm_dict[ftag1, ftag2]
            if transfer_dict is not None:
                transfer = transfer_dict[ftag1, ftag2]
            else:
                transfer = None

            bpw_win, inv_coupling = compute_couplings(
                mcm, nmt_binning, transfer, compute_Dl=compute_Dl
            )
            couplings[ftag1, ftag2]["bp_win"] = bpw_win
            couplings[ftag1, ftag2]["inv_coupling"] = inv_coupling

    return couplings
