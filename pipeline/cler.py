import argparse
import healpy as hp
import numpy as np
from bbmaster.utils import PipelineManager, get_pcls
import yaml
import os

############################
# Metadata related to maps #
############################
def get_map_label(map_type, experiment, frequency, split):
    # E.g. 'map_science_SAT1_f093_split2.fits'
    return 'map_%s_%s_%s_split%d.fits' % (map_type, experiment, frequency, split+1)

def get_mask_label(experiment, mask_type, split):
    # E.g. 'mask_SAT1_galactic_split1.fits'
    return 'mask_%s_%s_split%d.fits' % (experiment, mask_type, split+1)

def get_beam_label(experiment, frequency):
    # E.g. 'beam_SAT1_f093.txt'
    return 'beam_%s_%f.txt' % (experiment, frequency)

def get_transfer_label(experiment, split):
    # E.g. 'transfer_SAT1_split1.npz'
    return 'transfer_%s_split%d.npz' % (experiment, split+1)

def read_map(map_fname):
    return hp.read_map(map_fname, field=range(3), verbose=False) # Assume IQU

def read_mask(mask_fname):
    return hp.read_map(mask_fname, field=0, verbose=False) # Assume one field
    
def read_beam(beam_fname):
    from scipy.interpolate import interp1d
    larr_all = np.arange(3*nside)
    l, b = np.loadtxt(beam_fname, unpack=True)
    beam = interp1d(l, b, fill_value=0, bounds_error=False)(larr_all)
    if l[0] != 0:
        bb[:int(l[0])] = b[0]
    return beam

def read_transfer(transfer_fname, load_mcm=False, load_bmcm=False, 
                  load_transfer_function=False):
    with np.load(transfer_fname) as npz:
        ########################################################################
        # * wcal_inv:    inverse binned coupling matrix
        #                (shape [4,nbpw,4,nbpw]; needed to compute PS estimator)
        # * bpw_windows: bandpower window function of PS estimator
        #                (shape [4,nbpw,4,nl]; needed to bin theory PS)
        ########################################################################
        wcal_inv = npz['wcal_inv']
        bpw_windows = npz['bpw_windows']
        # These other objects are needed only for validation. Don't return them. 
        if load_mcm:
            mcm = npz['mcm']
        if load_bmcm:
            bmcm = npz['bmcm']
        if load_transfer_function:
            transfer_function = npz['transfer_function']
            transfer_function_error = npz['transfer_function_error']
        return wcal_inv, bpw_windows
    
def compute_pcl(field1, field2):
    ############################################################################
    # TODO
    #
    #
    # cls = []
    # for icl, i, j in man.cl_pair_iter(nmaps):
    #     f1 = fields[i]
    #     f2 = fields[j]
    #     pcl = binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
    #     if winv is not None:
    #         pcl = np.dot(winv, pcl.flatten()).reshape([4, nbpw])
    #     cls.append(pcl)
    ############################################################################

def compute_cells_from_splits(self, splits_list):
    ############################################################################
    # TODO: adapt
    # print(" Generating fields")
    # fields = {}
    # for b in range(self.n_bpss):
    #     for s in range(self.nsplits):
    #         name = self.get_map_label(b, s)
    #         print("  "+name)
    #         fname = splits_list[s]
    #         if not os.path.isfile(fname):  # See if it's gzipped
    #             fname = fname + '.gz'
    #         if not os.path.isfile(fname):
    #             raise ValueError("Can't find file ", splits_list[s])
    #         mp_q, mp_u = hp.read_map(fname, field=[2*b, 2*b+1],
    #                                  verbose=False)
    #         fields[name] = self.get_field(b, [mp_q, mp_u])


#     # Iterate over field pairs
#     print(" Computing cross-spectra")
#     cells = {}
#     for b1, b2, s1, s2, l1, l2 in self.get_cell_iterator():
#         wsp = self.workspaces[self.get_workspace_label(b1, b2)]
#         # Create sub-dictionary if it doesn't exist
#         if cells.get(l1) is None:
#             cells[l1] = {}
#         f1 = fields[l1]
#         f2 = fields[l2]
#         # Compute power spectrum
#         print("  "+l1+" "+l2)
#         cells[l1][l2] = wsp.decouple_cell(nmt.compute_coupled_cell(f1, f2))

#     return cells
    ############################################################################



def get_cell_iterator(self):
    ############################################################################
    # TODO: This is a later task
    # for b1 in range(self.n_bpss):
    #     for b2 in range(b1, self.n_bpss):
    #         for s1 in range(self.nsplits):
    #             l1 = self.get_map_label(b1, s1)
    #             if b1 == b2:
    #                 splits_range = range(s1, self.nsplits)
    #             else:
    #                 splits_range = range(self.nsplits)
    #             for s2 in splits_range:
    #                 l2 = self.get_map_label(b2, s2)
    #                 yield(b1, b2, s1, s2, l1, l2)
    ############################################################################
        

if __name__ == '__main__':
    # Get PS estimates from data splits, mask, transfer function (Tb), 
    # binned bandpower window function (Wbl), beams
    
    # Init params
    parser.add_argument("--config", type=str,
                        help='Path to yaml with all parameters')
    o = parser.parse_args()
    config = yaml.safe_load(open(o.config))
    
    nside = config.nside
    npix = hp.nside2npix(nside)
    outdir = config.outdir
    exps = config.experiments
    freqs = config.frequencies
    num_splits = config.num_splits
    mask_type = config.mask_type
    
    # Read data split maps
    data_split_maps = {}
    for e in exps:
        data_split_maps[e] = np.zeros((len(freqs[e]), num_splits[e]))
        for f_idx, f in enumerate(freqs[e]):
            for s_idx in range(num_splits[e]):
                data_split_maps[e][f_idx, s_idx] = read_map(
                    config.map_directory + \
                    get_map_label(e, f, s_idx)
                )
                
    # Read masks
    ############################################################################
    # TODO: allow for mask to vary with frequency 
    ############################################################################
    masks = {}
    for e in exps:
        masks[e] = {}
        for t in mask_type[e]:
            masks[e][t] = np.zeros(num_splits[e])
            for s_idx in range(num_splits[e]):
                masks[e][t][s_idx] = read_mask(
                    config.mask_directory + get_mask_label(e, t, s_idx)
                )
    
    # Read beams
    ############################################################################
    # TODO: allow for beam to vary with split ?
    ############################################################################
    beams = {}
    for e in exps:
        beams[e] = np.zeros(len(freqs[e]))
        for f_idx, f in enumerate(freqs[e]):
            beams[e][f_idx] = read_beam(
                config.beam_directory + get_beam_label(e, f)
            )
            
    # Read bandpasses
    ############################################################################
    # This is probably only needed at the "Saccer" or "Compsep" stage. Ignore.
    ############################################################################

    # Read transfer function and inverse binned bandpower window function
    ############################################################################
    # TODO: * allow for transfer to vary with frequency
    #       * allow for nonzero cross-split transfers: T_k^{(s1s1)(s2s2)} 
    #         or even T_k^{(s1s2)(s3s4)} ?
    #       * allow for nonzero cross-experiment transfers: 
    #         T_k^{(e1e1)(e2e2)} or even T_k^{(e1e2)(e3e4)} ?
    #       * allow for nonzero cross-frequency transfers: 
    #         T_k^{(f1f1)(f2f2)} or even T_k^{(f1f2)(f3f4)} ?
    ############################################################################
    transfers = {}
    for e in exps:
        transfers[e] = {}
        for s_idx in range(num_splits[e]):
            transfers[e][s_idx] = read_transfer(
                config.transfer_directory + get_transfer_label(e, s_idx)
            )
            
    
    ############################################################################
    # The following is the pcler stage without the loop over simulations and 
    # without the sacc stage.
    ############################################################################
    
    # Compute NaMaster fields
    ############################################################################
    # TODO: 
    # * Compute ell-bins (bandpower windows)
    # * Check if MCMs can be loaded from disk (-> nmt workspaces)
    # * Add the map_type spec as a key to the nmt_field dictionary.
    ############################################################################
    nmt_fields = {}
    for e in exps:
        nmt_fields[e] = np.zeros((len(freqs[e]), num_splits[e]))
        for f_idx, f in enumerate(freqs[e]):
            for s_idx in range(num_splits[e]):
                mask = masks[e][]
                mpQ = data_split_maps[e][f_idx,s_idx][1]
                mpU = data_split_maps[e][f_idx,s_idx][2]
                nmt_fields[e][f_idx, s_idx] = nmt.NmtField(mask, [mpQ, mpU])
        
    pcls = {}            
    for e1 in exps:
        pcls[e1] = {}
        for e2 in exps:
            pcls[e1][e2] = np.zeros((len(freqs[e1]), num_splits[e1],
                                     len(freqs[e2]), num_splits[e2])
                                   )
            for f_idx1, f1 in enumerate(freqs[e1]):
                for s_idx1 in range(num_splits[e1]):
                    for f_idx2, f2 in enumerate(freqs[e2]):
                        for s_idx2 in range(num_splits[e2]):
                            ####################################################
                            # TODO
                            #compute_pcls(...)
                            ####################################################
    
    # Compute final C_ell estimators
    # Beam-deconvolve C_ells
    ############################################################################
    # TODO
    ############################################################################
