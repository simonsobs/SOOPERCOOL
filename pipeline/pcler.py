import argparse
import healpy as hp
import numpy as np
import os
from bbmaster import BBmeta

def get_pcls(meta, map1, map2):
    """
    Compute the coupled angular power spectrum from two maps map1 and map2.
        
    Parameters
    ----------
    meta : BBmeta object
        BBmeta object of the metadata manager.
    map1 : array-like, shape [3,npix]
        TQU map to correlate.
    map2 : array-like, shape [3,npix]
        TQU map to correlate.
    """
    f1 = nmt.NmtField(meta.mask, [map1[1], map1[2]])
    f2 = nmt.NmtField(meta.mask, [map2[1], map2[2]])
    binning = meta.read_nmt_binning()
    pcl = binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
        
    return pcl

def get_pcls_transsim(meta, mapset):
    """
    Compute the coupled angular power spectra from a 2x2 matrix of QU maps with
    pure E and pure B modes.
        
    Parameters
    ----------
    meta : BBmeta object
        BBmeta object of the metadata manager.
    mapset : array-like, shape [2,2,npix]
        Pure-E and pure-B QU maps, corresponding to [pure_EB_in, QU, ipix]
    """
    binning = meta.read_nmt_binning()
    nbpw = meta.get_n_bandpowers()
    pcls = np.zeros((4, 4, nbpw))
    icl = 0
    for ip1, p1 in zip([0,1],['E','B']):
        for ip2, p2 in zip([0,1],['E','B']):
            f1 = nmt.NmtField(meta.mask, [mapset[ip1,0], mapset[ip1,1]])
            f2 = nmt.NmtField(meta.mask, [mapset[ip2,0], mapset[ip2,1]])
            pcls[icl] = binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
            icl += 1
    return pcls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pseudo-C_ell calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--plots", action="store_true", 
                        help="Plot the computed C_ells if True.")
    parser.add_argument("--mode", type=str, 
                        help="Four modes available: 'data', 'covsim', "
                            "'transsim', 'transval'.")
    o = parser.parse_args()
    meta = BBmeta(o.globals)
    
    # Global pars
    nside = meta.nside
    npix = hp.nside2npix(nside)
    # Load mask, bandpower windows, map sets
    mask = hp.ud_grade(hp.read_map(meta.get_fname_mask()),
                       nside_out=meta.nside)
    binning = meta.read_nmt_binning()
    nbpw = meta.get_n_bandpowers()
    ells_eff = meta.get_effective_ells()
    map_sets = meta.map_sets
    maps = {}
    pcls = {}
    
    if o.mode=='data':
        # Load inverse bandpower coupling matrix
        # For now, assume that all map sets are described by a single 
        # transfer function.
        # fname = meta.get_transfer_filename(meta.output_dir)
        # winv = np.load(fname)['wcal_inv']
        # if winv.shape != (4, nbpw, 4, nbpw):
        #     raise ValueError("Incompatible binning scheme and "
        #                      "binned MCM.")
        # winv = winv.reshape([4*nbpw, 4*nbpw])
        
        # Load maps, beams
        beams = {}
        for m in map_sets:
            n = meta.n_splits_from_map_set(m)
            maps[m] = np.zeros((n, 3, npix))
            beam_fname = meta.beam_file_from_map_set(m)
            beams[m] = meta.read_beam(beam_fname)
            for s in range(n):
                #read_map(self, map_set, id_split, id_sim=None, pol_only=False)
                maps[m][s] = meta.read_map(m, s)

        # Calculate power spectra
        for m1 in map_sets:
            pcls[m1] = {}
            n1 = meta.n_splits_from_map_set(m1)
            for s1 in range(n1):
                for m2 in map_sets:
                    n2 = meta.n_splits_from_map_set(m2)
                    pcls[m1][m2] = np.zeros((n1, n2, nbpw))
                    if m2==m1:
                        for s2 in range(s1): # Same tag: split pairs 
                                             # can be swapped
                            decoupled_pcl = meta.get_pcls(maps[m1][s1], 
                                                          maps[m1][s2])
                            # Calculate mask+filtering-corrected estimator
                            #decoupled_pcl = np.dot(
                            #    winv, coupled_pcl.flatten()).reshape([4, nbpw])
                            # Deconvolve the beam
                            beam_deconv = 1./beams[m1]**2
                            pcls[m1][m1][s1,s2,:] = decoupled_pcl*beam_deconv
                            pcls[m1][m1][s2,s1,:] = decoupled_pcl*beam_deconv
                    else:
                        for s2 in range(s1): # Different map sets: split pairs
                                             # cannot be swapped
                            decoupled_pcl_a = meta.get_pcls(maps[m1][s1], 
                                                          maps[m2][s2])
                            decoupled_pcl_b = meta.get_pcls(maps[m1][s2], 
                                                          maps[m2][s1])
                            # Calculate the mask+filtering-corrected estimator
                            #decoupled_pcl_a = np.dot(
                            #    winv, coupled_pcl_a.flatten()).reshape([4, 
                            #                                            nbpw])
                            #decoupled_pcl_b = np.dot(
                            #    winv, coupled_pcl_b.flatten()).reshape([4, 
                            #                                            nbpw])
                            # Deconvolve the beam
                            beam_deconv = 1./beams[m1]/beams[m2]
                            pcls[m1][m2][s1,s2,:] = decoupled_pcl_a*beam_deconv
                            pcls[m1][m2][s2,s1,:] = decoupled_pcl_b*beam_deconv
                
    elif o.mode=='covsim':
        # Load inverse bandpower coupling matrix
        # For now, assume that all map sets are described by a single 
        # transfer function.
        fname = meta.get_filename_transfer(meta.output_dir)
        winv = np.load(fname)['wcal_inv']
        if winv.shape != (4, nbpw, 4, nbpw):
            raise ValueError("Incompatible binning scheme and "
                             "binned MCM.")
        winv = winv.reshape([4*nbpw, 4*nbpw])
        
        # Load simulation maps, beams
        beams = {}
        nsims = meta.sim_pars['num_sims']
        for m in map_sets:
            n = meta.n_splits_from_map_set(m)
            maps[m] = np.zeros((nsims, n, 3, npix))
            beams[m] = meta.get_beam(m)
            for isim in range(nsims):
                for isplit in range(n):
                    maps[m][isim,isplit] = meta.read_map(m, isplit, isim=isim)
        
        # Calculate power spectra
        for m1 in map_sets:
            n1 = meta.n_splits_from_map_set(m1)
            for s1 in range(n1):
                for m2 in map_sets:
                    n2 = meta.n_splits_from_map_set(m2)
                    pcls[m1][m2] = np.zeros((nsims, n1, n2, nbpw))
                    for isim in range(nsims):
                        if m2==m1:
                            for s2 in range(s1): # Same tag: split pairs 
                                                 # can be swapped
                                coupled_pcl = meta.get_pcls_data(
                                    maps[m1][s1,isim], 
                                    maps[m1][s2,isim]
                                )
                                # Calculate the mask+filtering-corrected 
                                # estimator
                                decoupled_pcl = np.dot(
                                    winv, coupled_pcl.flatten()
                                ).reshape([4, nbpw])
                                # Deconvolve the beam
                                decoupled_pcl *= 1./beams[m1]**2
                                pcls[m1][m1][isim,s1,s2,:] = decoupled_pcl
                                pcls[m1][m1][isim,s2,s1,:] = decoupled_pcl
                        else:
                            for s2 in range(s1): # Different map sets: split 
                                                 # pairs cannot be swapped
                                coupled_pcl_a = meta.get_pcls(maps[m1][s1,isim],
                                                              maps[m2][s2,isim])
                                coupled_pcl_b = meta.get_pcls(maps[m1][s2,isim],                
                                                              maps[m2][s1,isim])
                                # Calculate the mask+filtering-corrected 
                                # estimator
                                decoupled_pcl_a = np.dot(
                                    winv, coupled_pcl_a.flatten()).reshape([4, 
                                                                          nbpw])
                                decoupled_pcl_b = np.dot(
                                    winv, coupled_pcl_b.flatten()).reshape([4, 
                                                                          nbpw])
                                # Deconvolve the beam
                                decoupled_pcl_a *= 1./(beams[m1]*beams[t2])
                                decoupled_pcl_b *= 1./(beams[m1]*beams[t2])
                                pcls[m1][m2][isim,s1,s2,:] = decoupled_pcl_a
                                pcls[m1][m2][isim,s2,s1,:] = decoupled_pcl_b
        
    elif o.mode=='transsim':
        # Load list of pure-E and pure-B power-law simulated maps, and compute
        # pseudo-C_ells (EE, EB, BE, BB) for all 4 input map combinations 
        # (EE, EB, BE, BB). Repeat for filtered and unfiltered maps, and loop
        # over simulations.
        # Output: pcls['(un-)filtered'] is array of shape [nsims,4,4,nbpw]
        nsims = meta.sim_pars['num_sims']
        for f in ['filtered', 'unfiltered']:
            maps[f] = np.zeros((nsims,2,2,npix))
            pcls[f] = np.zeros((nsims,4,4,nbpw))                    
            for isim in range(num_sims):
                maps[f][isim] = meta.get_maps_transsim(f, isim)
                pcls[f][isim] = meta.get_pcls_transsim(maps[f][isim])
                        
    elif o.mode=='transval':
        # Load list of transfer validation simulated maps, and compute
        # pseudo-C_ell autospectrum for each one of them. Repeat for filtered 
        # and unfiltered maps, and loop over simulations.
        # Output: pcls['(un-)filtered'] is array of shape [nsims,nbpw]
        nsims = meta.sim_pars['num_sims']
        for f in ['filtered', 'unfiltered']:
            maps[f] = np.zeros((num_sims,2,npix))
            pcls[f] = np.zeros((num_sims,4,nbpw))
            for isim in range(num_sims):
                maps[f][isim] = meta.get_maps_transval(f, isim)
                pcls[f][isim] = meta.get_pcls_data(maps[f][isim],
                                                   maps[f][isim])       
    else:
        raise ValueError("Unknown pipeline mode. Choose between "
                            "'data', 'covsim', 'transsim', 'transval'.")
    
    # Save beams, maps
    np.savez(os.path.join(meta.output_dir, f'maps_beams_{o.mode}.npz'),
             maps=maps, beams=beams)
    
    # Save power spectra
    for cln in meta.get_ps_names_list():
        # TODO: find out how to correctly load tags and splits.
        #       One possibility is to modify meta.get_ps_names_list(...)
        np.savez(os.path.join(meta.output_dir, f'cells_{o.mode}_{cln}.npz'), 
                 ells=ells_eff, cells=pcls[...])
    
    if o.plots:
        pass
        # TODO: plot stuff