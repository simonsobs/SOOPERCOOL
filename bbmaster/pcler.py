import argparse
import pymaster as nmt
import healpy as hp
from .utils import PipelineManager
import sacc


def get_pcls(man, fnames, names, fname_out, mask, binning):
    # Read maps
    fields = []
    for fname in fnames:
        mpQ, mpU = hp.read_map(fname, field=[0, 1])
        f = nmt.NmtField(mask, [mpQ, mpU])
        fields.append(f)
    nmaps = len(fields)

    # Compute pseudo-C_\ell
    cls = []
    for icl, i, j in man.cl_pair_iter(nmaps):
        f1 = fields[i]
        f2 = fields[j]
        pcl = binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
        cls.append(pcl)

    # Save to sacc
    leff = binning.get_effective_ells()
    s = sacc.Sacc()
    for n in names:
        s.add_tracer('Misc', n)
    for icl, i, j in man.cl_pair_iter(nmaps):
        s.add_ell_cl('cl_ee', names[i], names[j], leff, cls[icl][0])
        s.add_ell_cl('cl_eb', names[i], names[j], leff, cls[icl][1])
        if i != j:
            s.add_ell_cl('cl_be', names[i], names[j], leff, cls[icl][2])
        s.add_ell_cl('cl_bb', names[i], names[j], leff, cls[icl][3])
    s.save_fits(fname_out, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pseudo-C_ell calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--first-sim", type=int, help='Index of first sim')
    parser.add_argument("--num-sims", type=int, help='Number of sims')
    parser.add_argument("--sim-sorter", type=str,
                        help='Name of sorting routine')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--sim-type", type=str, help='filtered or input')
    o = parser.parse_args()

    man = PipelineManager(o.globals)
    sorter = getattr(man, o.sim_sorter)
    bpw_edges = man.get_bpw_edges()
    b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])

    sim_names = sorter(o.first_sim, o.num_sims, o.output_dir, which='names')
    file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir,
                             which=o.sim_type)
    file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir,
                              which=o.sim_type+'_Cl')
    mask = hp.ud_grade(hp.read_map(man.fname_mask),
                       nside_out=man.nside)
    for fin, nam, fout in zip(file_input_list, sim_names, file_output_list):
        if isinstance(fin, str):
            fin = [fin]
            nam = [nam]
        get_pcls(man, fin, nam, fout, mask, b)
