import argparse
import healpy as hp
import numpy as np
from bbmaster.utils import PipelineManager, get_pcls

def read_beam(fdir, tag):
    ############################################################################
    # TODO
    ############################################################################

def get_filename(mode, fdir, tag):
    ############################################################################
    # TODO
    ############################################################################
    
def read_map(fdir, tag, isim=None, filt=None, pol=None):
    ############################################################################
    # TODO
    ############################################################################
    
def 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pseudo-C_ell calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--mode", type=str, 
                        help="Four modes available: 'data', 'covsim', "
                            "'transsim', 'transval'.")
    o = parser.parse_args()
    man = PipelineManager(o.globals)
    
    # Global pars
    nside = man.nside
    npix = hp.nside2npix(nside)
    # Load mask, bandpower windows, map tags
    mask = hp.ud_grade(hp.read_map(man.fname_mask),
                       nside_out=nside)
    nmt_bins = get_nmt_bins() #### get_nmt_bins()
    nbpw = 
    tags = man.tags
    maps = {}
    pcls = {}
    
    if o.mode=='data':
        beams = {}
        # Load inverse bandpower coupling matrix
        fname = get_filename('transfer_function', o.output_dir)
        winv = np.load(fname)['wcal_inv']
        for t1 in tags:
            pcls[t1] = {}
            # Load data maps
            maps[t1] = read_map(o.map_directory, t1) #### read_map(
                                                   ####   fdir, tag, 
                                                   ####   isim=None,
                                                   ####   filt=None,
                                                   ####   pol=None
                                                   #### )
            # Load beams
            beams[t1] = read_beam(man.beam_directory, t)
        # Calculate power spectra
        ########################################################################
        # TODO: Include splits
        ########################################################################
            for t2 in tags:
                coupled_pcl = get_data_pcl(man.output_dir, maps[t1], maps[t2])
                decoupled_pcl = np.dot(winv, coupled_pcl)
                pcls[t1][t2] = decoupled_pcl
    elif o.mode=='covsim':
        # Load inverse bandpower coupling matrix
        fname = get_filename('transfer_function', o.output_dir)
        winv = np.load(fname)['wcal_inv']
        # Load list of covariance simulations
        num_sims = man.sim_pars['num_sims']
        for t1 in tags:
            pcls[t1] = {}
            beams[t1] = read_beam(man.beam_directory, t1)
            maps[t1] = np.zeros((num_sims,2,npix))
            for isim in range(num_sims):
                maps[t1][isim] = read_map(man.covsim_dir, t, isim=isim)
                ################################################################
                # TODO: Implement pcls with covsims
                ################################################################
            
        
    elif o.mode=='transsim':
        # Load list of pure E and pure B power-law simulations
        num_sims = man.sim_pars['num_sims']
        for t in tags:
            for f in ['filtered', 'unfiltered']:
                ################################################################
                # TODO: Compute pCls (EE, EB, BE, BB)
                #       for input (EE, EB, BE, BB), for filt, unfilt
                #       Include splits
                ################################################################
                maps[t][f] = np.zeros((num_sims,2,2,npix))
                pcls[t][f] = np.zeros((num_sims,4,4,nbpws))                    
                for isim in range(num_sims):
                    for ip, p in enumerate(['E','B']):
                        maps[t][f][isim,ip] = read_map(man.transsim_dir, t,
                                                     isim=isim,
                                                     filt=f, pol=p)
                    pcls[t][f][isim] = get_transsim_pcl(man.output_dir, 
                                                        maps[t][f][isim])
                        
    elif o.mode=='transval':
        # Load list of transfer validation simulations
        num_sims = man.sim_pars['num_sims']
        for t in tags:
            maps[t] = {}
            pcls[t] = {}
            for f in ['filtered', 'unfiltered']:
                ################################################################
                # TODO: Compute pCls (EE, EB, BE, BB) for filt, unfilt
                #       Include splits ?
                ################################################################
                maps[t][f] = np.zeros((num_sims,2,npix))
                pcls[t][f] = np.zeros((num_sims,4,nbpw))
                for isim in range(num_sims):
                    maps[t][f][isim] = read_map(man.valsim_dir, t,
                                                isim=isim,
                                                filt=f)
                    pcls[t][f][isim] = get_transval_pcl(man.output_dir, 
                                                        maps[t][f][isim])       
    else:
        raise ValueError("Unknown pipeline mode. Choose between "
                            "'data', 'covsim', 'transsim', 'transval'.")
        
        
    
#     sorter = getattr(man, o.sim_sorter)
    

#     sim_names = sorter(o.first_sim, o.num_sims, o.output_dir, which='names')
#     file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir,
#                              which=o.sim_type)
#     file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir,
#                               which=o.sim_type+'_Cl')
#     mask = hp.ud_grade(hp.read_map(man.fname_mask),
#                        nside_out=man.nside)
#     for fin, nam, fout in zip(file_input_list, sim_names, file_output_list):
#         if isinstance(fin, str):
#             fin = [fin]
#             nam = [nam]
#         get_pcls(man, fin, nam, fout, mask, b, winv=winv)
