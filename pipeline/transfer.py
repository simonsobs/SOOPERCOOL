import argparse
import numpy as np
from soopercool import BBmeta


def transfer(args):
    """
    """
    meta = BBmeta(args.globals)

    spin_pairs = [
        "spin0xspin0",
        "spin0xspin2",
        "spin2xspin2"
    ]

    pcls_mat_dict = {
        spin_pair: {
            "filtered": [],
            "unfiltered": []
        } for spin_pair in spin_pairs
    }

    cl_dir = meta.cell_transfer_directory
    coupling_dir = meta.coupling_directory

    nmt_binning = meta.read_nmt_binning()
    n_bins = nmt_binning.get_n_bands()
    nl = 3*meta.nside

    # Load the pseudo-cl matrices for each simulation
    # Should be (n_comb_pure, n_comb_mode, n_bins)
    meta.timer.start("Mean sims")
    for id_sim in range(meta.tf_est_num_sims):
        for spin_pair in spin_pairs:
            for label in ["filtered", "unfiltered"]:
                pcls_mat = np.load(f"{cl_dir}/pcls_mat_tf_est_{label}_{id_sim:04d}.npz")  # noqa
                pcls_mat_dict[spin_pair][label] += [pcls_mat[spin_pair]]

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = {
        spin_pair: np.mean(pcls_mat_dict[spin_pair]["filtered"], axis=0)
        for spin_pair in spin_pairs}
    pcls_mat_unfiltered_mean = {
        spin_pair: np.mean(pcls_mat_dict[spin_pair]["unfiltered"], axis=0)
        for spin_pair in spin_pairs}
    meta.timer.stop("Mean sims")

    meta.timer.start("Load mode-coupling matrices")
    # Load the beam corrected mode coupling matrices
    mcms_dict = {}
    for map_set1, map_set2 in meta.get_ps_names_list("all", coadd=True):
        mcms = np.load(f"{coupling_dir}/mcm_{map_set1}_{map_set2}.npz")
        mcms_dict[map_set1, map_set2] = {
            spin_pair: mcms[f"{spin_pair}_binned"] for spin_pair in spin_pairs
        }

    # Load the un-beamed mode coupling matrix
    mcm = np.load(f"{coupling_dir}/mcm.npz")
    mcms_dict_nobeam = {
        spin_pair: mcm[f"{spin_pair}_binned"] for spin_pair in spin_pairs
    }

    # If we B-purify for transfer function estimation, load the un-beamed
    # purified mode coupling matrix
    if meta.tf_est_pure_B:
        mcm_pure = np.load(f"{coupling_dir}/mcm_pure.npz")
        mcms_dict_nobeam_pure = {
            spin_pair: mcm_pure[f"{spin_pair}_binned"] for spin_pair in spin_pairs  # noqa
        }

    meta.timer.stop("Load mode-coupling matrices", verbose=True)

    meta.timer.start("Inverse the unfiltered pcls matrix")
    # Is there a better way to formulate this ?
    cct_invs = {
        spin_pair: np.transpose(
            np.linalg.inv(
                np.transpose(
                    np.einsum('jil,jkl->ikl',
                              pcls_mat_unfiltered_mean[spin_pair],
                              pcls_mat_unfiltered_mean[spin_pair]),
                    axes=[2, 0, 1]
                )
            ), axes=[1, 2, 0]
        ) for spin_pair in spin_pairs
    }
    meta.timer.stop("Inverse the unfiltered pcls matrix", verbose=True)

    meta.timer.start("Compute transfer function")
    # Same comment as above
    trans = {
        spin_pair: np.einsum('ijl,jkl->kil', cct_invs[spin_pair],
                             np.einsum('jil,jkl->ikl',
                                       pcls_mat_unfiltered_mean[spin_pair],
                                       pcls_mat_filtered_mean[spin_pair]))
        for spin_pair in spin_pairs
    }
    meta.timer.stop("Compute transfer function", verbose=True)

    # Same comment as above
    etrans = {
        spin_pair: np.std(
            np.array([np.einsum('ijl,jkl->kil', cct_invs[spin_pair],
                                np.einsum('jil,jkl->ikl',
                                          pcls_mat_unfiltered_mean[spin_pair],
                                          clf))
                      for clf in pcls_mat_dict[spin_pair]["filtered"]]),
            axis=0)
        for spin_pair in spin_pairs}

    np.savez(
        f"{coupling_dir}/transfer_function.npz",
        tf_spin0xspin0=trans["spin0xspin0"],
        tf_spin0xspin2=trans["spin0xspin2"],
        tf_spin2xspin2=trans["spin2xspin2"],
        tf_std_spin0xspin0=etrans["spin0xspin0"],
        tf_std_spin0xspin2=etrans["spin0xspin2"],
        tf_std_spin2xspin2=etrans["spin2xspin2"]
    )

    meta.timer.start("Compute full coupling")
    for (map_set1, map_set2), mcm in mcms_dict.items():

        couplings = {}
        for spin_pair in spin_pairs:
            # Binned mask MCM times transfer function
            tbmcm = np.einsum('ijk,jklm->iklm', trans[spin_pair],
                              mcm[spin_pair])

            # Fully binned coupling matrix (including filtering)
            btbmcm = np.transpose(
                np.array([np.sum(tbmcm[:, :, :, nmt_binning.get_ell_list(i)],
                                 axis=-1)
                          for i in range(n_bins)]),
                axes=[1, 2, 3, 0])

            # Invert and multiply by tbmcm to get final bandpower
            # window functions.
            if spin_pair == "spin0xspin0":
                size = 1
            elif spin_pair == "spin0xspin2":
                size = 2
            elif spin_pair == "spin2xspin2":
                size = 4
            ibtbmcm = np.linalg.inv(btbmcm.reshape([size*n_bins, size*n_bins]))
            winflat = np.dot(ibtbmcm, tbmcm.reshape([size*n_bins, size*nl]))
            wcal_inv = ibtbmcm.reshape([size, n_bins, size, n_bins])
            bpw_windows = winflat.reshape([size, n_bins, size, nl])

            couplings[f"bp_win_{spin_pair}"] = bpw_windows
            couplings[f"inv_coupling_{spin_pair}"] = wcal_inv

        np.savez(
            f"{coupling_dir}/couplings_{map_set1}_{map_set2}.npz",
            **couplings
        )

    for filter_tag in ["filtered", "unfiltered"]:
        couplings_nobeam = {}
        couplings_list = [couplings_nobeam]
        mcms_list = [mcms_dict_nobeam]

        if meta.tf_est_pure_B:
            couplings_nobeam_pure = {}
            couplings_list.append(couplings_nobeam_pure)
            mcms_list.append(mcms_dict_nobeam_pure)

        for couplings, mcms_dict in zip(couplings_list, mcms_list):
            for spin_pair in spin_pairs:
                if filter_tag == "filtered":
                    tbmcm = np.einsum('ijk,jklm->iklm', trans[spin_pair],
                                      mcms_dict[spin_pair])
                else:
                    tbmcm = mcms_dict[spin_pair]
                btbmcm = np.transpose(
                    np.array([np.sum(
                        tbmcm[:, :, :, nmt_binning.get_ell_list(i)], axis=-1)
                            for i in range(n_bins)]),
                    axes=[1, 2, 3, 0])
                # Invert and multiply by tbmcm to get final bandpower
                # window functions.
                if spin_pair == "spin0xspin0":
                    size = 1
                elif spin_pair == "spin0xspin2":
                    size = 2
                elif spin_pair == "spin2xspin2":
                    size = 4
                ibtbmcm = np.linalg.inv(btbmcm.reshape([size*n_bins,
                                                        size*n_bins]))
                winflat = np.dot(ibtbmcm, tbmcm.reshape([size*n_bins,
                                                         size*nl]))
                wcal_inv = ibtbmcm.reshape([size, n_bins, size, n_bins])
                bpw_windows = winflat.reshape([size, n_bins, size, nl])

                couplings[f"bp_win_{spin_pair}"] = bpw_windows
                couplings[f"inv_coupling_{spin_pair}"] = wcal_inv

        np.savez(
            f"{coupling_dir}/couplings_{filter_tag}.npz",
            **couplings_nobeam
        )
        if meta.tf_est_pure_B:
            np.savez(
                f"{coupling_dir}/couplings_{filter_tag}_pure.npz",
                **couplings_nobeam_pure
            )

    meta.timer.stop("Compute full coupling", verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer function stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    args = parser.parse_args()
    transfer(args)
