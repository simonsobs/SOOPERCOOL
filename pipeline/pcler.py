import argparse
import healpy as hp
import numpy as np
from bbmaster import BBmeta


# TODO: put in metadata manager    
def get_map(tag, isplit, isim=None):
    ############################################################################
    # TODO
    # Only for data maps and (if isim!=None) covariance simulations.
    ############################################################################

# TODO: put in metadata manager  
def get_nmt_bins():
    ############################################################################
    # TODO
    ############################################################################
    return

def get_beam(tag):
    ############################################################################
    # TODO
    # output shape: [nbpw]
    ############################################################################
    return
    
# TODO: put in metadata manager  
def get_filename_transfer(fdir):
    ############################################################################
    # TODO
    ############################################################################
    return

# TODO: put in metadata manager 
def get_pcls(map1, map2):
    f1 = nmt.NmtField(meta.mask, [map1[1], map1[2]])
    f2 = nmt.NmtField(meta.mask, [map2[1], map2[2]])
    pcl = meta.binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
        
    return pcl

# TODO: put in metadata manager 
def get_map_transval(signal='CMB', filt, isim):
    # output QU maps (shape [2, npix]).
    maps = np.zeros((2,meta.npix))
    fname = meta.map_directory+\
        f'/map_transval_{signal}_{filt}_{str(isim).zfill(4)}.fits'
    try:
        maps[ip] = hp.read_map(fname, field=(1,2))
    except:
        maps[ip] = hp.read_map(fname, field=(0,1))
    return maps

# TODO: put in metadata manager 
def get_maps_transsim(filt, isim)
    # output pure-E and pure-B QU maps (shape [2, 2, npix]), corresponding to 
    # [pure_EB_in, QU, ipix].
    maps = np.zeros((2,2,meta.npix))
    for ip, p in zip([0,1],['E','B']):
        fname = meta.map_directory+\
            f'/map_transsim_{filt}_seed{str(isim).zfill(4)}_{p}only.fits'
        try:
            maps[ip] = hp.read_map(fname, field=(1,2))
        except:
            maps[ip] = hp.read_map(fname, field=(0,1))
    return maps

# TODO: put in metadata manager
def get_pcls_transsim(mapset)
    # input pure-E and pure-B QU maps (shape [2, 2, npix]), corresponding to 
    # [pure_EB_in, QU, ipix], and output 4x4 matrix of cross-pol C_ells 
    # (shape [4,4,nbpws]), corresponding to [(EE,EB,BE,EE)_in, 
    # (EE,EB,BE,EE)_out, i_ell]
    pcls = np.zeros((4,4,meta.nbpws))
    icl = 0
    for ip1, p1 in zip([0,1],['E','B']):
        for ip2, p2 in zip([0,1],['E','B']):
            f1 = nmt.NmtField(meta.mask, [mapset[ip1,0], mapset[ip1,1]])
            f2 = nmt.NmtField(meta.mask, [mapset[ip2,0], mapset[ip2,1]])
            pcls[icl] = meta.binning.bin_cell(nmt.compute_coupled_cell(f1, f2))
            icl += 1
    return pcls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pseudo-C_ell calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--mode", type=str, 
                        help="Four modes available: 'data', 'covsim', "
                            "'transsim', 'transval'.")
    o = parser.parse_args()
    meta = BBmeta(o.globals)
    
    # Global pars
    nside = meta.nside
    npix = hp.nside2npix(nside)
    # Load mask, bandpower windows, map tags
    mask = hp.ud_grade(hp.read_map(meta.fname_mask),
                       nside_out=meta.nside)
    nmt_bins = meta.get_nmt_bins() #### get_nmt_bins()
    nbpw = len(nmt_bins)
    tags = meta.tags
    maps = {}
    pcls = {}
    
    if o.mode=='data':
        # Load inverse bandpower coupling matrix
        # For now, assume that all map sets (tags) are described by a single 
        # transfer function.
        fname = meta.get_transfer_filename(o.output_dir)
        winv = np.load(fname)['wcal_inv']
        if winv.shape != (4, nbpw, 4, nbpw):
            raise ValueError("Incompatible binning scheme and "
                             "binned MCM.")
        winv = winv.reshape([4*nbpw, 4*nbpw])
        
        # Load maps, beams
        beams = {}
        for t in tags:
            n = range(t['num_splits'])
            maps[t] = np.zeros(n1)
            beams[t] = meta.get_beam(t)
            for s in range(n):
                maps[t][s] = meta.get_map(t, s)

        # Calculate power spectra
        for t1 in tags:
            n1 = range(t1['num_splits'])
            for s1 in range(n1):
                for t2 in tags:
                    n2 = range(t2['num_splits'])
                    pcls[t1][t2] = np.zeros((n1, n2, nbpws))
                    if t2==t1:
                        for s2 in range(s1): # Same tag: split pairs 
                                             # can be swapped
                            coupled_pcl = meta.get_pcls(maps[t1][s1], 
                                                        maps[t1][s2])
                            # Calculate mask+filtering-corrected estimator
                            decoupled_pcl = np.dot(
                                winv, coupled_pcl.flatten()).reshape([4, nbpw])
                            # Deconvolve the beam
                            beam_deconv = 1./beams[t1]**2
                            pcls[t1][t1][s1,s2,:] = decoupled_pcl*beam_deconv
                            pcls[t1][t1][s2,s1,:] = decoupled_pcl*beam_deconv
                    else:
                        for s2 in range(s1): # Different tags: split pairs
                                             # cannot be swapped
                            coupled_pcl_a = meta.get_pcls(maps[t1][s1], 
                                                          maps[t2][s2])
                            coupled_pcl_b = meta.get_pcls(maps[t1][s2], 
                                                          maps[t2][s1])
                            # Calculate the mask+filtering-corrected estimator
                            decoupled_pcl_a = np.dot(
                                winv, coupled_pcl_a.flatten()).reshape([4, 
                                                                        nbpw])
                            decoupled_pcl_b = np.dot(
                                winv, coupled_pcl_b.flatten()).reshape([4, 
                                                                        nbpw])
                            # Deconvolve the beam
                            beam_deconv = 1./beams[t1]/beams[t2]
                            pcls[t1][t2][s1,s2,:] = decoupled_pcl_a*beam_deconv
                            pcls[t1][t2][s2,s1,:] = decoupled_pcl_b*beam_deconv
                
    elif o.mode=='covsim':
        # Load inverse bandpower coupling matrix
        # For now, assume that all map sets (tags) are described by a single 
        # transfer function.
        fname = meta.get_filename_transfer(o.output_dir)
        winv = np.load(fname)['wcal_inv']
        nbpw = binning.get_n_bands()
        if winv.shape != (4, nbpw, 4, nbpw):
            raise ValueError("Incompatible binning scheme and "
                             "binned MCM.")
        winv = winv.reshape([4*nbpw, 4*nbpw])
        
        # Load simulation maps, beams
        beams = {}
        nsims = meta.sim_pars['num_sims']
        for t in tags:
            nsplits = range(t['num_splits'])
            maps[t] = np.zeros(nsplits,nsims)
            beams[t] = meta.get_beam(t)
            for isplit in range(n):
                for isim in range(nsims):
                    maps[t][isplit,isim] = meta.get_map(t, isplit=isplit, 
                                                        isim = isim)
        
        # Calculate power spectra
        for t1 in tags:
            n1 = range(t1['num_splits'])
            for s1 in range(n1):
                for t2 in tags:
                    nsp2 = range(t2['num_splits'])
                    pcls[t1][t2] = np.zeros((nsims, n1, n2, nbpws))
                    for isim in range(nsims):
                        if t2==t1:
                            for s2 in range(s1): # Same tag: split pairs 
                                                 # can be swapped
                                coupled_pcl[] = meta.get_pcls_data(
                                    maps[t1][s1,isim], 
                                    maps[t1][s2,isim]
                                )
                                # Calculate the mask+filtering-corrected estimator
                                decoupled_pcl = np.dot(
                                    winv, coupled_pcl.flatten()
                                ).reshape([4, nbpw])
                                # Deconvolve the beam
                                decoupled_pcl *= 1./beams[t1]**2
                                pcls[t1][t1][isim,s1,s2,:] = decoupled_pcl
                                pcls[t1][t1][isim,s2,s1,:] = decoupled_pcl
                        else:
                            for s2 in range(s1): # Different tags: split pairs
                                                 # cannot be swapped
                                coupled_pcl_a = meta.get_pcls(maps[t1][s1,isim],
                                                              maps[t2][s2,isim])
                                coupled_pcl_b = meta.get_pcls(maps[t1][s2,isim],                
                                                              maps[t2][s1,isim])
                                # Calculate the mask+filtering-corrected estimator
                                decoupled_pcl_a = np.dot(
                                    winv, coupled_pcl_a.flatten()).reshape([4, 
                                                                          nbpw])
                                decoupled_pcl_b = np.dot(
                                    winv, coupled_pcl_b.flatten()).reshape([4, 
                                                                          nbpw])
                                # Deconvolve the beam
                                decoupled_pcl_a *= 1./(beams[t1]*beams[t2])
                                decoupled_pcl_b *= 1./(beams[t1]*beams[t2])
                                pcls[t1][t2][isim,s1,s2,:] = decoupled_pcl_a
                                pcls[t1][t2][isim,s2,s1,:] = decoupled_pcl_b
        
    elif o.mode=='transsim':
        # Load list of pure-E and pure-B power-law simulated maps, and compute
        # pseudo-C_ells (EE, EB, BE, BB) for all 4 input map combinations 
        # (EE, EB, BE, BB). Repeat for filtered and unfiltered maps, and loop
        # over simulations.
        # Output: pcls['(un-)filtered'] is array of shape [nsims,4,4,nbpws]
        nsims = meta.sim_pars['num_sims']
        for f in ['filtered', 'unfiltered']:
            maps[f] = np.zeros((nsims,2,2,npix))
            pcls[f] = np.zeros((nsims,4,4,nbpws))                    
            for isim in range(num_sims):
                maps[f][isim] = meta.get_maps_transsim(f, isim)
                pcls[f][isim] = meta.get_pcls_transsim(maps[f][isim])
                        
    elif o.mode=='transval':
        # Load list of transfer validation simulated maps, and compute
        # pseudo-C_ell autospectrum for each one of them. Repeat for filtered 
        # and unfiltered maps, and loop over simulations.
        # Output: pcls['(un-)filtered'] is array of shape [nsims,nbpws]
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
    np.savez(o.outdir+f'maps_beams_{o.mode}.npz', maps=maps, beams=beams)
    
    # Save power spectra
    np.savez(o.outdir+f'cells_{o.mode}.npz', cells=pcls)