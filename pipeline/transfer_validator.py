import argparse
from bbmaster.utils import PipelineManager
import sacc
import numpy as np
import matplotlib.pyplot as plt
import os


def get_cl_sim(fname):
    s = sacc.Sacc.load_fits(fname)
    tname = list(s.tracers.keys())[0]
    ls, clee = s.get_ell_cl('cl_ee', tname, tname)
    _, cleb = s.get_ell_cl('cl_eb', tname, tname)
    _, clbb = s.get_ell_cl('cl_bb', tname, tname)
    return ls, np.array([clee, cleb, cleb, clbb])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer function')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--transfer-threshold", type=float, default=0.05,
                        help='Minimum value of the transfer function '
                        'to validate')
    o = parser.parse_args()

    man = PipelineManager(o.globals)

    leff = man.get_nmt_bins().get_effective_ells()

    # Get all simulation names and the pCl files
    sorter = man.val_sim_names
    sim_names = sorter(0, -1, o.output_dir, which='names')
    pcl_in_names = sorter(0, -1, o.output_dir, which='input_Cl')
    pcl_filt_names = sorter(0, -1, o.output_dir, which='filtered_Cl')
    pcl_dec_names = sorter(0, -1, o.output_dir, which='decoupled_Cl')

    # Read all C_ells and compute mean and std
    nsims = len(sim_names)
    cls_in = np.array([get_cl_sim(f)[1] for f in pcl_in_names])
    cls_filt = np.array([get_cl_sim(f)[1] for f in pcl_filt_names])
    cls_dec = np.array([get_cl_sim(f)[1] for f in pcl_dec_names])
    cl_in = np.mean(cls_in, axis=0)
    ecl_in = np.std(cls_in, axis=0)
    cl_filt = np.mean(cls_filt, axis=0)
    ecl_filt = np.std(cls_filt, axis=0)
    cl_dec = np.mean(cls_dec, axis=0)
    ecl_dec = np.std(cls_dec, axis=0)

    # Read transfer function and compute theoretical predictions
    fname_transfer = man.get_filename('transfer_function', o.output_dir)
    transf = np.load(fname_transfer)
    cl_in_th = np.einsum('ijkl,kl', transf['bmcm'], man.cls_val)
    cl_filt_th = np.einsum('ijk,jk->ik', transf['transfer_function'], cl_in_th)
    cl_dec_th = np.einsum('ijkl,kl', transf['bpw_windows'], man.cls_val)

    # Determine ells to use in comparison
    tf = transf['transfer_function'][0, 0]
    goodl = (tf > o.transfer_threshold) & (leff < 2*man.nside)

    titles = ['EE', 'EB', 'BE', 'BB']
    # Transfer function
    fig, axes = plt.subplots(4, 4, figsize=(10, 10),
                             sharex=True)
    fig.suptitle("Transfer function")
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.plot(leff, transf['transfer_function'][i, j], 'r-')
            ax.errorbar(leff[goodl], transf['transfer_function'][i, j][goodl],
                        yerr=transf['transfer_function_error'][i, j][goodl],
                        fmt='b.')
            ax.text(0.6, 0.9, f'{titles[j]}->{titles[i]}',
                    transform=ax.transAxes)
    plt.savefig(os.path.join(o.output_dir, 'transfer_val.pdf'),
                bbox_inches='tight')

    # Bandpower window functions
    bpw = transf['bpw_windows']
    fig, axes = plt.subplots(4, 4, figsize=(10, 10),
                             sharex=True)
    fig.suptitle("Bandpower window functions")
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            for k in range(len(leff)):
                ax.plot(bpw[i, k, j, :], 'r-')
            ax.text(0.6, 0.9, f'{titles[j]}->{titles[i]}',
                    transform=ax.transAxes)
    plt.savefig(os.path.join(o.output_dir, 'bandpower_windows.pdf'),
                bbox_inches='tight')

    # Input spectra
    fig.suptitle("Masked PCLs")
    fig, axes = plt.subplots(2, 2, figsize=(8, 8),
                             sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(leff, cl_in[i], yerr=ecl_in[i], fmt='k.')
        ax.plot(leff, cl_in_th[i], 'r-')
        if i in [0, 3]:  # Log-scale for EE and BB
            ax.set_yscale('log')
        ax.text(0.85, 0.9, titles[i], transform=ax.transAxes)
    plt.savefig(os.path.join(o.output_dir, 'pcl_input_val.pdf'),
                bbox_inches='tight')

    # Filtered spectra
    fig.suptitle("Filtered PCLs")
    fig, axes = plt.subplots(2, 2, figsize=(8, 8),
                             sharex=True)
    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(leff, cl_filt[i], yerr=ecl_filt[i], fmt='k.')
        ax.plot(leff, cl_filt_th[i], 'r-')
        if i in [0, 3]:  # Log-scale for EE and BB
            ax.set_yscale('log')
        ax.text(0.85, 0.9, titles[i], transform=ax.transAxes)
    plt.savefig(os.path.join(o.output_dir, 'pcl_filtered_val.pdf'),
                bbox_inches='tight')

    # Decoupled spectra
    lth = np.arange(3*man.nside)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8),
                             sharex=True)
    fig.suptitle("Decoupled C_ells")
    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(leff[goodl], cl_dec[i][goodl],
                    yerr=ecl_dec[i][goodl], fmt='k.')
        ax.plot(leff[goodl], cl_dec_th[i][goodl], 'r-')
        ax.plot(lth, man.cls_val[i], 'b--')
        if i in [0, 3]:  # Log-scale for EE and BB
            ax.set_yscale('log')
        ax.text(0.85, 0.9, titles[i], transform=ax.transAxes)
    plt.savefig(os.path.join(o.output_dir, 'cl_decoupled_val.pdf'),
                bbox_inches='tight')
    plt.show()
