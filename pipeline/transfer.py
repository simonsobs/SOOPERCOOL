import argparse
from bbmaster.utils import PipelineManager
import sacc
import numpy as np
import matplotlib.pyplot as plt
from bbmaster import BBmeta

def transfer(args):
    """
    """
    meta = BBmeta(args.globals)

    pcls_mat_dict = {
        "filtered": [],
        "unfiltered": []
    }

    cl_dir = meta.cell_transfer_directory
    coupling_dir = meta.coupling_directory

    # Load the pseudo-cl matrices for each simulation
    # Should be (n_comb_pure, n_comb_mode, n_bins)
    for id_sim in range(meta.tf_est_num_sims):
        pcls_mat = np.load(f"{cl_dir}/pcls_mat_tf_est_{id_sim:04d}.npz")
        pcls_mat_dict["filtered"] += [pcls_mat["pcls_mat_filtered"]]
        pcls_mat_dict["unfiltered"] += [pcls_mat["pcls_mat_unfiltered"]]

    # Average the pseudo-cl matrices
    pcls_mat_filtered_mean = np.mean(pcls_mat_dict["filtered"], axis=0)
    pcls_mat_unfiltered_mean = np.mean(pcls_mat_dict["unfiltered"], axis=0)

    # Load the binning and create
    # the `binner`. Maybe this should move
    # to the mcm step.
    nmt_binning = meta.read_nmt_binning()
    nl = meta.lmax
    n_bins = nmt_binning.get_n_bands()
    binner = np.array([nmt_binning.bin_cell(np.array([cl]))[0] for cl in np.eye(nl)]).T

    # Load the MCM and compute the binned
    # MCM. Same comment as above : should
    # move this to the mcm step.
    mcm = np.load(f"{coupling_dir}/mcm.npz")["spin2xspin2"]

    # Load beams
    beams = {}
    for map_set in meta.map_sets_list:
        l, bl = meta.read_beam(map_set)
        beams[map_set] = bl[:meta.lmax]

    # Apply the beam to the mcm
    beamed_mcm = {}
    for map_set1, map_set2 in meta.get_ps_names_list("all", coadd=True):
        beam_mcm = mcm * np.outer(beams[map_set1], beams[map_set2])[np.newaxis,:,np.newaxis,:]
        beam_bmcm = np.einsum('ij,kjlm->kilm', binner, beam_mcm)
        beamed_mcm[map_set1, map_set2] = beam_bmcm

    # Is there a better way to formulate this ?
    cct_inv = np.transpose(
        np.linalg.inv(
            np.transpose(
                np.einsum('jil,jkl->ikl', pcls_mat_unfiltered_mean, pcls_mat_unfiltered_mean), axes=[2,0,1]
            )
        ), axes=[1,2,0]
    )

    # Same comment as above
    trans = np.einsum('ijl,jkl->kil', cct_inv,
                      np.einsum('jil,jkl->ikl', pcls_mat_unfiltered_mean, pcls_mat_filtered_mean))
    
    # Same comment as above
    etrans = np.std(np.array([np.einsum('ijl,jkl->kil', cct_inv,
                                        np.einsum('jil,jkl->ikl', pcls_mat_unfiltered_mean, clf))
                              for clf in pcls_mat_dict["filtered"]]), axis=0)
    
    np.savez(
        f"{coupling_dir}/transfer_function.npz",
        transfer_function=trans,
        transfer_function_error=etrans
    )

    for (map_set1, map_set2), beam_bmcm in beamed_mcm.items():
    # Binned mask MCM times transfer function
        tbmcm = np.einsum('ijk,jklm->iklm', trans, beam_bmcm)

        # Fully binned coupling matrix (including filtering)
        btbmcm = np.transpose(
            np.array([np.sum(tbmcm[:, :, :, nmt_binning.get_ell_list(i)],
                            axis=-1)
                    for i in range(nmt_binning.get_n_bands())]),
            axes=[1, 2, 3, 0])
        
        # Invert and multiply by tbmcm to get final bandpower
        # window functions.
        ibtbmcm = np.linalg.inv(btbmcm.reshape([4*n_bins, 4*n_bins]))
        winflat = np.dot(ibtbmcm, tbmcm.reshape([4*n_bins, 4*nl]))
        wcal_inv = ibtbmcm.reshape([4, n_bins, 4, n_bins])
        bpw_windows = winflat.reshape([4, n_bins, 4, nl])

        np.savez(
            f"{coupling_dir}/couplings_{map_set1}_{map_set2}.npz",
            mcm=mcm,  # We save the original mcm for completeness
            bmcm=beam_bmcm,  # We save the original mcm for completeness
            bpw_windows=bpw_windows,
            wcal_inv=wcal_inv
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer function stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    
    args = parser.parse_args()

    transfer(args)

a="""



    
    
    

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