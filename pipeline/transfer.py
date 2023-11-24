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
    nl = meta.lmax + 1

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
        for spin_pair in spin_pairs:
            if filter_tag == "filtered":
                tbmcm = np.einsum('ijk,jklm->iklm', trans[spin_pair],
                                  mcms_dict_nobeam[spin_pair])
            else:
                tbmcm = mcms_dict_nobeam[spin_pair]
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
            ibtbmcm = np.linalg.inv(btbmcm.reshape([size*n_bins, size*n_bins]))
            winflat = np.dot(ibtbmcm, tbmcm.reshape([size*n_bins, size*nl]))
            wcal_inv = ibtbmcm.reshape([size, n_bins, size, n_bins])
            bpw_windows = winflat.reshape([size, n_bins, size, nl])

            couplings_nobeam[f"bp_win_{spin_pair}"] = bpw_windows
            couplings_nobeam[f"inv_coupling_{spin_pair}"] = wcal_inv

        np.savez(
            f"{coupling_dir}/couplings_{filter_tag}.npz",
            **couplings_nobeam
        )

    meta.timer.stop("Compute full coupling", verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer function stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    args = parser.parse_args()
    transfer(args)

_ = """
    # Save to file
    fname = man.get_filename('transfer_function', o.output_dir)
    np.savez(fname,
             mcm=mcm,  # We save the original mcm for completeness
             bmcm=bmcm,  # We save the original mcm for completeness
             transfer_function=trans,
             transfer_function_error=etrans,
             bpw_windows=bpw_windows,
             wcal_inv=wcal_inv)

    if o.plot:
        # Reliable ells
        goodl = leff < 2*man.nside

        # Now recover Cl_filt from Cl_in
        cl_filt_r = np.einsum('ijl,kjl->kil', trans, cl_in)
        combs = ['EE', 'EB', 'BE', 'BB']
        for i, comb in enumerate(combs):
            plt.figure()
            plt.title(comb)
            plt.plot(leff[goodl], cl_filt[i, i][goodl], 'k-')
            plt.plot(leff[goodl], cl_filt_r[i, i][goodl], 'r:')
        plt.show()
        exit(1)

        for i1, comb1 in enumerate(combs):
            for i2, comb2 in enumerate(combs):
                plt.figure()
                plt.title(f'{comb2}->{comb1}')
                plt.plot(leff[goodl], trans[i1, i2][goodl], 'k-')
        plt.show()

        for i1, c1 in enumerate(['EE', 'EB', 'BE', 'BB']):
            plt.figure()
            plt.title(f'{c1}->XY')
            for i2, (c2, col) in enumerate(zip(['EE', 'EB', 'BE', 'BB'],
                                               ['r', 'b', 'y', 'c'])):
                p = cl_th[i1, i2]

                plt.plot(leff[goodl], p[goodl], col+'-', label=f'XY={c2}')
                plt.plot(leff[goodl], -p[goodl], col+':')
                plt.errorbar(leff[goodl], np.fabs(cl_in[i1, i2][goodl]),
                             yerr=ecl_in[i1, i2][goodl]/np.sqrt(nsims),
                             fmt=col+'.')
            plt.loglog()
            plt.legend()
        plt.show()
"""
