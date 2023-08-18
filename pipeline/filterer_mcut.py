import argparse
import healpy as hp
from bbmaster.utils import PipelineManager
import numpy as np


def filter_map(man, mpQ, mpU, msk, m_cut):
    # Mask the maps (binary)
    mpQ *= msk
    mpU *= msk

    # Compute alms
    almE, almB = hp.map2alm_spin([mpQ, mpU], spin=2)

    # Filter them by setting to zero all alms with m <= m_cut
    # Note that the alms are ordered by m in HEALPix, so all
    # the alms with m <= m_cut are the first (m_cut+1)*((lmax+1)-m_cut/2)
    lmax = 3*man.nside-1
    nalm_filter = (m_cut+1)*(lmax+1)-((m_cut+1)*m_cut)//2
    almE[:nalm_filter] = 0
    almB[:nalm_filter] = 0

    # Transform back to real space
    mpQf, mpUf = hp.alm2map_spin([almE, almB], man.nside, spin=2, lmax=lmax)
    return mpQf, mpUf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple m-cut filter')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--first-sim", type=int, help='Index of first sim')
    parser.add_argument("--num-sims", type=int, help='Number of sims')
    parser.add_argument("--sim-sorter", type=str,
                        help='Name of sorting routine')
    parser.add_argument("--m-cut", type=int, help='m-mode cut')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--plot", action='store_true',
                        help='Show last filtered and unfiltered maps?')
    o = parser.parse_args()
    man = PipelineManager(o.globals)

    sorter = getattr(man, o.sim_sorter)
    file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir,
                             which='input')
    file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir,
                              which='filtered')

    msk = hp.read_map(man.fname_binary_mask)
    for fin, fout in zip(file_input_list, file_output_list):
        # Read maps
        mpQ, mpU = hp.read_map(fin, field=[0, 1])
        # Filter
        mpQf, mpUf = filter_map(man, mpQ, mpU, msk, o.m_cut)
        # Write output
        hp.write_map(fout, [mpQf, mpUf], overwrite=True, dtype=np.float32)

    if o.plot:
        import matplotlib.pyplot as plt

        hp.mollview(mpQ, title='Unfiltered Q')
        hp.mollview(mpQf, title='Filtered Q')
        hp.mollview(mpU, title='Unfiltered U')
        hp.mollview(mpUf, title='Filtered U')
        plt.show()
