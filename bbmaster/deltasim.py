import numpy as np
import healpy as hp
import pymaster as nmt
import copy


class DeltaBbl(object):
    def __init__(self, nside, dsim, filt, bins, lmin=2, lmax=None, nsim_per_ell=10, seed0=1000, n_iter=0):
        if not isinstance(dsim, dict):
            raise TypeError("For now delta simulators can only be "
                            "specified through a dictionary.")

        if not isinstance(filt, dict):
            raise TypeError("For now filtering operations can only be "
                            "specified through a dictionary.")

        if not isinstance(bins, nmt.NmtBin):
            raise TypeError("`bins` must be a NaMaster NmtBin object.")
        self.dsim_d = copy.deepcopy(dsim)
        self.dsim = self._dsim_default
        self.filt_d = copy.deepcopy(filt)
        self.filt = self._filt_default
        self.lmin = lmin
        self.nside = nside
        if lmax is None:
            lmax = 3*self.nside-1
        self.lmax = lmax
        self.bins = bins
        self.n_ells = lmax-lmin+1
        self.nsim_per_ell = nsim_per_ell
        self.seed0 = seed0
        self.n_iter = n_iter
        self._prepare_filtering()

    def _prepare_filtering(self):
        # Match pixel resolution
        self.filt_d['mask'] = hp.ud_grade(self.filt_d['mask'], nside_out=self.nside)

    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        cl = np.zeros(3*self.nside)
        cl[ell] = 1
        if self.dsim_d['stats'] == 'Gaussian':
            return hp.synfast(cl, self.nside)
        else:
            raise ValueError("Only Gaussian sims implemented")

    def _filt_default(self, mp_true):
        return self.filt_d['mask']*mp_true

    def gen_deltasim(self, seed, ell):
        dsim_true = self.dsim(seed, ell)
        dsim_filt = self.filt(dsim_true)
        return dsim_filt

    def gen_deltasim_bpw(self, seed, ell):
        dsim = self.gen_deltasim(seed, ell)
        cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        return cb

    def gen_Bbl_at_ell(self, ell):
        Bbl = np.zeros(self.bins.get_n_bands())
        for i in range(self.nsim_per_ell):
            seed = ell*self.nsim_per_ell + i
            cb = self.gen_deltasim_bpw(seed, ell)
            Bbl += cb
        Bbl /= self.nsim_per_ell
        return Bbl

    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)

    def gen_Bbl_all(self):
        return np.array([self.gen_Bbl_at_ell(l)
                         for l in self.get_ells()]).T
