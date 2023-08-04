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
    parser.add_argument("--sim-sorter", type=str,
                        help='Name of sorting routine')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--sim-type", type=str, help='filtered or input')
    o = parser.parse_args()

    man = PipelineManager(o.globals)
    sorter = getattr(man, o.sim_sorter)
    sim_names = sorter(0, -1, o.output_dir, which='names')
    pcl_in_names = sorter(0, -1, o.output_dir, which='input_Cl')
    pcl_filt_names = sorter(0, -1, o.output_dir, which='filtered_Cl')

    # Loop through each file and read all power spectra
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
    cls_in = np.array(cls_in)
    cls_filt = np.array(cls_filt)
    cl_in = np.mean(cls_in, axis=0)
    cl_filt = np.mean(cls_filt, axis=0)
    ecl_in = np.std(cls_in, axis=0)
    ecl_filt = np.std(cls_filt, axis=0)

    # Naive transfer function
    trans0 = np.array([[cl_filt[j, i]/cl_in[i, i] for i in range(4)]
                       for j in range(4)])
    # Potentially more accurate transfer function
    trans1 = np.einsum('ijl,jkl->ikl',
                       np.einsum('jil,jkl->ikl', cl_in, cl_filt),
                       np.transpose(np.linalg.inv(
                           np.transpose(np.einsum('jil,jkl->ikl',
                                                  cl_in, cl_in),
                                        axes=[2, 0, 1])),
                                    axes=[1, 2, 0]))

    # Save to file
    np.savez(os.path.join(o.output_dir, 'transfer.npz'),
             TF=trans1)
    combs = ['EE', 'EB', 'BE', 'BB']
    for i1, comb1 in enumerate(combs):
        for i2, comb2 in enumerate(combs):
            plt.figure()
            plt.title(f'{comb2}->{comb1}')
            plt.plot(leff, trans0[i1, i2], 'k-')
            plt.plot(leff, trans1[i1, i2], 'r--')
    plt.show()
    print(trans0[0, 0])
    print(trans1[0, 0])
