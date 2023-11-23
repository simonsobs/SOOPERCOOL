import yaml
import numpy as np
import os
from scipy.interpolate import interp1d
from so_models_v3 import SO_Noise_Calculator_Public_v3_1_2 as noise_calc
import pymaster as nmt
import healpy as hp
import sacc
import camb


def get_pcls(man, fnames, names, fname_out, mask, binning, winv=None):
    """
    man -> pipeline manager
    fnames -> files with input maps
    names -> map names
    fname_out -> output file name
    mask -> mask
    binning -> binning scheme to use
    winv -> inverse binned MCM (optional)
    """

    if winv is not None:
        nbpw = binning.get_n_bands()
        if winv.shape != (4, nbpw, 4, nbpw):
            raise ValueError("Incompatible binning scheme and "
                             "binned MCM.")
        winv = winv.reshape([4*nbpw, 4*nbpw])

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
        if winv is not None:
            pcl = np.dot(winv, pcl.flatten()).reshape([4, nbpw])
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


def theory_cls(cosmo_params, lmax, lmin=0):
    """
    """
    params = camb.set_params(**cosmo_params)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
    ell = np.arange(lmin, lmax+1)
    out_ps = {
        "TT": powers["total"][:, 0][lmin:lmax+1],
        "EE": powers["total"][:, 1][lmin:lmax+1],
        "TE": powers["total"][:, 3][lmin:lmax+1],
        "BB": powers["total"][:, 2][lmin:lmax+1]
    }
    for spec in ["EB", "TB"]:
        out_ps[spec] = np.zeros_like(ell)
    return ell, out_ps


def generate_noise_map_white(nside, noise_rms_muKarcmin, ncomp=3):
    """
    """
    size = 12 * nside**2

    pixel_area_deg = hp.nside2pixarea(nside, degrees=True)
    pixel_area_arcmin = 60**2 * pixel_area_deg

    noise_rms_muK_T = noise_rms_muKarcmin / np.sqrt(pixel_area_arcmin)
    
    out_map = np.zeros((ncomp, size))
    out_map[0, :] = np.random.randn(size) * noise_rms_muK_T

    if ncomp == 3:
        noise_rms_muK_P = np.sqrt(2) * noise_rms_muK_T
        out_map[1, :] = np.random.randn(size) * noise_rms_muK_P
        out_map[2, :] = np.random.randn(size) * noise_rms_muK_P
        return out_map
    return out_map


def get_noise_cls(fsky, lmax, lmin=0, sensitivity_mode='baseline', 
                          oof_mode='optimistic', is_beam_deconvolved=True):
    """
    """
    # Load noise curves
    noise_model = noise_calc.SOSatV3point1(sensitivity_mode=sensitivity_mode)
    lth, _, nlth_P = noise_model.get_noise_curves(
        fsky,
        lmax+1,
        delta_ell=1,
        deconv_beam=is_beam_deconvolved
    )
    lth = np.concatenate(([0, 1], lth))[lmin:]
    nlth_P = np.array(
        [np.concatenate(([0, 0], nl))[lmin:] for nl in nlth_P]
        
    )
    # Only support polarization noise at the moment
    nlth_dict = {
        "T": {freq_band: nlth_P[i]/2 for i, freq_band in enumerate(noise_model.get_bands())},
        "P": {freq_band: nlth_P[i] for i, freq_band in enumerate(noise_model.get_bands())}
    }
    
    return lth, nlth_dict


def generate_noise_map(nl_T, nl_P, hitmap, n_splits):
    """
    """    
    # healpix ordering ["TT", "EE", "BB", "TE"]
    noise_mat = np.array([nl_T, nl_P, nl_P, np.zeros_like(nl_P)])
    # Normalize the noise
    noise_mat *= n_splits

    noise_map = hp.synfast(noise_mat, hp.get_nside(hitmap), pol=True, new=True)

    # Weight with hitmap
    noise_map[:, hitmap != 0] /= np.sqrt(hitmap[hitmap != 0] / np.max(hitmap))

    return noise_map

    
def random_src_mask(mask, nsrcs, mask_radius_arcmin):
    """
    pspy.so_map
    """
    ps_mask = mask.copy()
    src_ids = np.random.choice(np.where(mask == 1)[0], nsrcs)
    for src_id in src_ids:
        vec = hp.pix2vec(hp.get_nside(mask), src_id)
        disc = hp.query_disc(hp.get_nside(mask), vec, np.deg2rad(mask_radius_arcmin / 60))
        ps_mask[disc] = 0
    return ps_mask


def beam_gaussian(ll, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.
    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.
    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ll * (ll + 1) * sigma_rad**2)


def beam_hpix(ll, nside):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.
    Args:
        l (float or array): multipoles.
        nside (int): HEALPix resolution parameter.
    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / nside
    return beam_gaussian(ll, fwhm_hp_amin)

def create_binning(lmin, lmax, delta_ell):
    """
    """
    bin_low = np.arange(lmin, lmax, delta_ell)
    bin_high = bin_low + delta_ell - 1

    idx = bin_high <= lmax
    bin_low, bin_high = bin_low[idx], bin_high[idx]

    if not lmax in bin_high:
        bin_low = np.concatenate([bin_low, [bin_high[-1]+1]])
        bin_high = np.concatenate([bin_high, [lmax]])
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def power_law_cl(ell, amp, delta_ell, power_law_index):
    """
    """
    pl_ps = {}
    for spec in ["TT", "TE", "TB", "EE", "EB", "BB"]:
        if isinstance(amp, dict):
            A = amp[spec]
        else:
            A = amp
        pl_ps[spec] = A / (ell + delta_ell) ** power_law_index

    return pl_ps


def m_filter_map(map, mask, m_cut):
    """
    """

    map_masked = map * mask
    nside = hp.get_nside(map)
    lmax = 3 * nside - 1

    alms_list = []
    alms = hp.map2alm(map_masked, lmax=lmax)

    n_modes_to_filter = (m_cut + 1) * (lmax + 1) - ((m_cut + 1) * m_cut) // 2
    alms[:, :n_modes_to_filter] = 0.

    return hp.alm2map(alms, nside=nside, lmax=lmax)

def toast_filter_map(map, mask):
    """
    """
    raise NotImplementedError("TOAST filtering not implemented yet.")


def get_split_pairs_from_coadd_ps_name(map_set1, map_set2,
                                       all_splits_ps_names,
                                       cross_splits_ps_names,
                                       auto_splits_ps_names):
    """
    """
    split_pairs_list = {
        "auto": [],
        "cross": []
    }
    for split_ms1, split_ms2 in all_splits_ps_names:
        if not (split_ms1.startswith(map_set1) and split_ms2.startswith(map_set2)):
            continue

        if (split_ms1, split_ms2) in cross_splits_ps_names:
            split_pairs_list["cross"].append((split_ms1, split_ms2))
        
        elif (split_ms1, split_ms2) in auto_splits_ps_names:
            split_pairs_list["auto"].append((split_ms1, split_ms2))

    return split_pairs_list



class PipelineManager(object):
    def __init__(self, fname_config):
        with open(fname_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.nside = self.config['nside']
        self.fname_mask = self.config['mask']
        self.fname_binary_mask = self.config['binary_mask']
        self.bpw_edges = None
        self.pl_names = np.loadtxt(self.config['pl_sims'], dtype=str)
        self.val_names = np.loadtxt(self.config['val_sims'], dtype=str)
        self.pl_input_dir = self.config['pl_sims_dir']
        self.val_input_dir = self.config['val_sims_dir']
        self._get_cls_PL()
        self._get_cls_val()

        self.stname_mcm = 'mcm'
        self.stname_filtpl = 'filter_PL'
        self.stname_pclpl_in = 'pcl_PL_in'
        self.stname_pclpl_filt = 'pcl_PL_filt'
        self.stname_filtval = 'filter_val'
        self.stname_pclval_in = 'pcl_val_in'
        self.stname_pclval_filt = 'pcl_val_filt'
        self.stname_clval = 'cl_val'
        self.stname_transfer = 'transfer'

    def get_filename(self, product, out_base_dir, simname=None):
        if product == 'mcm':  # NaMaster's MCM
            fname = os.path.join(out_base_dir, '..',
                                 self.stname_mcm, 'mcm.npz')
        if product == 'mcm_plots':  # NaMaster's MCM plots dir
            fname = os.path.join(out_base_dir, '..',
                                 self.stname_mcm)
        if product == 'pl_sim_input':  # Input PL sims
            fname = os.path.join(self.pl_input_dir,
                                 simname+'.fits')
        if product == 'pl_sim_filtered':  # Filtered PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_filtpl,
                simname+'.fits')
        if product == 'pcl_pl_sim_input':  # PCL of input PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclpl_in,
                simname+'_pcl_in.fits')
        if product == 'pcl_pl_sim_filtered':  # PCL of filtered PL sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclpl_filt,
                simname+'_pcl_filt.fits')
        if product == 'val_sim_input':  # Input validation sims
            fname = os.path.join(self.val_input_dir,
                                 simname+'.fits')
        if product == 'val_sim_filtered':  # Filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_filtval,
                simname+'.fits')
        if product == 'pcl_val_sim_input':  # PCL of input validations sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclval_in,
                simname+'_pcl_in.fits')
        if product == 'pcl_val_sim_filtered':  # PCL of filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_pclval_filt,
                simname+'_pcl_filt.fits')
        if product == 'cl_val_sim':  # CL of filtered validation sims
            fname = os.path.join(
                out_base_dir, '..', self.stname_clval,
                simname+'_cl.fits')
        if product == 'transfer_function':
            fname = os.path.join(
                out_base_dir, '..', self.stname_transfer, 'transfer.npz')
        return fname

    def _get_cls_PL(self):
        d = np.load(self.config['cl_PL'])
        lin = d['ls']
        ls = np.arange(3*self.nside)
        self.cls_PL = []
        for kind in ['EE', 'EB', 'BE', 'BB']:
            cli = interp1d(lin, d[f'cl{kind}'], bounds_error=False,
                           fill_value=0)
            self.cls_PL.append(cli(ls))
        self.cls_PL = np.array(self.cls_PL)

    def _get_cls_val(self):
        d = np.load(self.config['cl_val'])
        lin = d['ls']
        ls = np.arange(3*self.nside)
        self.cls_val = []
        for kind in ['EE', 'EB', 'BE', 'BB']:
            cli = interp1d(lin, d[f'cl{kind}'], bounds_error=False,
                           fill_value=0)
            self.cls_val.append(cli(ls))
        self.cls_val = np.array(self.cls_val)

    def get_bpw_edges(self):
        if self.bpw_edges is None:
            self.bpw_edges = np.load(self.config['bpw_edges'])['bpw_edges']
        return self.bpw_edges

    def get_nmt_bins(self):
        bpw_edges = self.get_bpw_edges()
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b

    def cl_pair_iter(self, nmaps):
        icl = 0
        for i in range(nmaps):
            for j in range(i, nmaps):
                yield icl, i, j
                icl += 1

    def val_sim_names(self, sim0, nsims, output_dir, which='input'):
        if nsims == -1:
            names = self.val_names[sim0:]
        else:
            names = self.val_names[sim0:sim0+nsims]
        fnames = []
        for n in names:
            if which == 'names':
                fn = n
            elif which == 'input':
                fn = self.get_filename('val_sim_input',
                                       output_dir, n)
            elif which in ['filtered', 'decoupled']:
                fn = self.get_filename('val_sim_filtered',
                                       output_dir, n)
            elif which == 'input_Cl':
                fn = self.get_filename("pcl_val_sim_input",
                                       output_dir, n)
            elif which == 'filtered_Cl':
                fn = self.get_filename("pcl_val_sim_filtered",
                                       output_dir, n)
            elif which == 'decoupled_Cl':
                fn = self.get_filename("cl_val_sim",
                                       output_dir, n)
            else:
                raise ValueError(f"Unknown kind {which}")
            fnames.append(fn)
        return fnames

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
                fE = self.get_filename('pl_sim_input',
                                       output_dir, n+'_E')
                fB = self.get_filename('pl_sim_input',
                                       output_dir, n+'_B')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'filtered':
                fE = self.get_filename('pl_sim_filtered',
                                       output_dir, n+'_E')
                fB = self.get_filename('pl_sim_filtered',
                                       output_dir, n+'_B')
                if EandB:
                    fnames.append([fE, fB])
                else:
                    fnames.append(fE)
                    fnames.append(fB)
            elif which == 'input_Cl':
                fn = self.get_filename('pcl_pl_sim_input', output_dir, n)
                fnames.append(fn)
            elif which == 'filtered_Cl':
                fn = self.get_filename('pcl_pl_sim_filtered', output_dir, n)
                fnames.append(fn)
            else:
                raise ValueError(f"Unknown kind {which}")
        return fnames
