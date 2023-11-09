import argparse, glob, os
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
    # This is just a random toy model filter, it doesn't have any physical meaning.
    lmax = 3*man.nside-1
    nalm_filter = (m_cut+1)*(lmax+1)-((m_cut+1)*m_cut)//2
    almE[:nalm_filter] = 0
    almB[:nalm_filter] = 0

    # Transform back to real space
    mpQf, mpUf = hp.alm2map_spin([almE, almB], man.nside, spin=2, lmax=lmax)
    return mpQf, mpUf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple m-cut filter, set all alm with m<=m_cut to zero.')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')

    parser.add_argument("--sim-dir", type=str, help='Directory contains all simulations (fits format).')
    parser.add_argument("--filtered-dir", type=str, help='Directory to put all filtered maps.')
    parser.add_argument("--binary-mask-file", type=str, help='Path to binary mask file.')
    parser.add_argument("--m-cut", type=int, help='m-mode cut. This will set all alm with m<=m_cut to zero.')
    parser.add_argument("--plot", action='store_true',
                        help='Show last filtered and unfiltered maps?')
    o = parser.parse_args()
    man = PipelineManager(o.globals)

    sim_dir = o.sim_dir
    if os.path.isabs(o.filtered_dir):
        filtered_dir = o.filtered_dir
    else:
        filtered_dir = os.path.join(o.output_dir, '..', o.filtered_dir)
    # Create filtered_dir if not exists
    os.makedirs(filtered_dir, exist_ok=True)
    if os.path.samefile(o.sim_dir, filtered_dir):
        raise InputValueError('--sim-dir and --filtered-dir can not be the same directory.')

    file_input_list = glob.glob(os.path.join(o.sim_dir, '*.fits'))
    msk = hp.read_map(o.binary_mask_file)
    
    for fin in file_input_list:
        # get basename
        basename = os.path.basename(fin)
        fout = os.path.join(filtered_dir, basename)
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
