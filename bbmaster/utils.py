import yaml
import numpy as np
import os


class PipelineManager(object):
    def __init__(self, fname_config):
        with open(fname_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.pl_names = np.loadtxt(self.config['pl_sims'], dtype=str)
        self.pl_input_dir = self.config['pl_sims_dir']
        self.nside = self.config['nside']
        self.fname_mask = self.config['mask']
        self.fname_binary_mask = self.config['binary_mask']
        self.bpw_edges = None

    def get_bpw_edges(self):
        if self.bpw_edges is None:
            self.bpw_edges = np.load(self.config['bpw_edges'])['bpw_edges']
        return self.bpw_edges

    def cl_pair_iter(self, nmaps):
        icl = 0
        for i in range(nmaps):
            for j in range(i, nmaps):
                yield icl, i, j
                icl += 1

    def pl_sim_names_EandB(self, sim0, nsims, output_dir, which):
        return self.pl_sim_names(sim0, nsims, output_dir,
                                 which=which, EandB=True)

    def pl_sim_names(self, sim0, nsims, output_dir, which='input',
                     EandB=False):
        if nsims == -1:
            names = self.pl_names[sim0:]
        else:
            names = self.pl_names[sim0:sim0+nsims]
        fnames = []
        for n in names:
            if which == 'names':
                fE = n+'_E'
                fB = n+'_B'
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'input':
                fE = os.path.join(self.pl_input_dir, n+'_E.fits')
                fB = os.path.join(self.pl_input_dir, n+'_B.fits')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'filtered':
                fE = os.path.join(output_dir, n+'_filtered_E.fits')
                fB = os.path.join(output_dir, n+'_filtered_B.fits')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'input_Cl':
                fnames.append(os.path.join(output_dir, n+'_cl_in.fits'))
            elif which == 'filtered_Cl':
                fnames.append(os.path.join(output_dir, n+'_cl_filtered.fits'))
            else:
                raise ValueError(f"Unknown kind {which}")
        return fnames
