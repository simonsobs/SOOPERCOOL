import argparse
from .utils import PipelineManager
import sacc
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer function')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--use-theory", action='store_true',
                        help='Pass to use theoretical power spectrum '
                        'convolved with MCM to estimate the transfer function')
    parser.add_argument("--plot", action='store_true',
                        help='Pass to generate a plot of the MCM.')
    o = parser.parse_args()

    man = PipelineManager(o.globals)

    # Get all simulation names and the pCl files
    sorter = man.pl_sim_names_EandB
    sim_names = sorter(0, -1, o.output_dir, which='names')
    pcl_in_names = sorter(0, -1, o.output_dir, which='input_Cl')
    pcl_filt_names = sorter(0, -1, o.output_dir, which='filtered_Cl')

    # Loop through each file and read all power spectra
    nsims = len(sim_names)
    cls_in = []
    cls_filt = []
    for n, pin, pfilt in zip(sim_names, pcl_in_names, pcl_filt_names):
        nE, nB = n
        print(nE, nB, pin, pfilt)
        order = [(nE, nE, 'cl_ee'),
                 (nE, nE, 'cl_eb'),
                 (nE, nE, 'cl_eb'),
                 (nE, nE, 'cl_bb'),
                 (nE, nB, 'cl_ee'),
                 (nE, nB, 'cl_eb'),
                 (nE, nB, 'cl_be'),
                 (nE, nB, 'cl_bb'),
                 (nE, nB, 'cl_ee'),
                 (nE, nB, 'cl_be'),
                 (nE, nB, 'cl_eb'),
                 (nE, nB, 'cl_bb'),
                 (nB, nB, 'cl_ee'),
                 (nB, nB, 'cl_eb'),
                 (nB, nB, 'cl_eb'),
                 (nB, nB, 'cl_bb')]
        s = sacc.Sacc.load_fits(pin)
        cls = np.array([s.get_ell_cl(kind, t1, t2)[1]
                        for t1, t2, kind in order]).reshape([4, 4, -1])
        cls_in.append(cls)
        s = sacc.Sacc.load_fits(pfilt)
        cls = np.array([s.get_ell_cl(kind, t1, t2)[1]
                        for t1, t2, kind in order]).reshape([4, 4, -1])
        cls_filt.append(cls)
    s = sacc.Sacc.load_fits(pfilt)
    leff = s.get_ell_cl('cl_ee', nE, nE)[0]
    # Shape is [Nsims, N_pure_pairs, N_pol_pairs, N_ells]
    cls_in = np.array(cls_in)
    cls_filt = np.array(cls_filt)

    # Compute average over sims
    # Shape is [N_pure_pairs, N_pol_pairs, N_ells]
    cl_in = np.mean(cls_in, axis=0)
    cl_filt = np.mean(cls_filt, axis=0)
    ecl_in = np.std(cls_in, axis=0)
    ecl_filt = np.std(cls_filt, axis=0)

    # Construct binning matrix
    b = man.get_nmt_bins()
    nl = 3*man.nside
    nbpw = b.get_n_bands()
    binner = np.array([b.bin_cell(np.array([cl]))[0]
                       for cl in np.eye(nl)]).T

    # Compute theoretical input pCl
    cl0 = np.zeros(3*man.nside)
    # Full mask MCM
    mcm = np.load(os.path.join(o.output_dir, 'mcm.npz'))['mcm']
    # Binned mask MCM
    bmcm = np.einsum('ij,kjlm->kilm', binner, mcm)
    cl_th = np.array([
        np.einsum('ijkl,kl->ij', bmcm,
                  np.array([man.cls_PL[0], cl0, cl0, cl0])),
        np.einsum('ijkl,kl->ij', bmcm,
                  np.array([cl0, man.cls_PL[1], cl0, cl0])),
        np.einsum('ijkl,kl->ij', bmcm,
                  np.array([cl0, cl0, man.cls_PL[2], cl0])),
        np.einsum('ijkl,kl->ij', bmcm,
                  np.array([cl0, cl0, cl0, man.cls_PL[3]]))])

    # Transfer function via least-squares fitting.
    if o.use_theory:
        pcl = cl_th
    else:
        pcl = cl_in
    cct_inv = np.transpose(
        np.linalg.inv(
            np.transpose(np.einsum('jil,jkl->ikl', pcl, pcl),
                         axes=[2, 0, 1])),
        axes=[1, 2, 0])
    trans = np.einsum('ijl,jkl->kil', cct_inv,
                      np.einsum('jil,jkl->ikl', pcl, cl_filt))
    # Standard deviation from MC simulations
    etrans = np.std(np.array([np.einsum('ijl,jkl->kil', cct_inv,
                                        np.einsum('jil,jkl->ikl', pcl, clf))
                              for clf in cls_filt]), axis=0)

    # Binned mask MCM times transfer function
    tbmcm = np.einsum('ijk,jklm->iklm', trans, bmcm)
    # Fully binned coupling matrix (including filtering)
    btbmcm = np.transpose(
        np.array([np.sum(tbmcm[:, :, :, b.get_ell_list(i)],
                         axis=-1)
                  for i in range(b.get_n_bands())]),
        axes=[1, 2, 3, 0])
    # Invert and multiply by tbmcm to get final bandpower
    # window functions.
    ibtbmcm = np.linalg.inv(btbmcm.reshape([4*nbpw, 4*nbpw]))
    winflat = np.dot(ibtbmcm, tbmcm.reshape([4*nbpw, 4*nl]))
    wcal_inv = ibtbmcm.reshape([4, nbpw, 4, nbpw])
    bpw_windows = winflat.reshape([4, nbpw, 4, nl])

    # Save to file
    np.savez(os.path.join(o.output_dir, 'transfer.npz'),
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
