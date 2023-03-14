import numpy as np
import healpy as hp
import pymaster as nmt
import copy
import os
import warnings


class DeltaBbl(object):
    def __init__(self, nside, dsim, filt, bins, prefix_save=None,
                 lmin=2, lmax=None, nsim_per_ell=10, seed0=1000,
                 n_iter=0):
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
        self.alm_ord = hp.Alm()
        self._sqrt2 = np.sqrt(2.)
        self._oosqrt2 = 1/self._sqrt2
        self.prefix_save= prefix_save

    def cleanup(self):
        os.system(f'rm -rf {self.prefix_save}')

    def _prepare_filtering(self):
        # Match pixel resolution
        self.filt_d['mask'] = hp.ud_grade(self.filt_d['mask'], nside_out=self.nside)

    def _gen_gaussian_alm(self, ell):
        cl = np.zeros(3*self.nside)
        cl[ell] = 1
        # TODO: we can save time on the SHT massively, since in this case there is no
        # sum over ell!
        return hp.synfast(cl, self.nside)

    def _gen_Z2_alm(self, ell):
        idx = self.alm_ord.getidx(3*self.nside-1, ell,
                                  np.arange(ell+1))
        # Generate Z2 numbers (one per m)
        # TODO: Is it clear that it's best to excite all m's
        # rather than one (or some) at a time?
        rans = self._oosqrt2*(2*np.random.binomial(1, 0.5,
                                                   size=2*(ell+1))-1).reshape([2,
                                                                               ell+1])
        # Correct m=0 (it should be real and have twice as much variance
        rans[0, 0] *= self._sqrt2
        rans[1, 0] = 0
        # Populate alms and transform to map
        # TODO: we can save time on the SHT massively, since in this case there is no
        # sum over ell!
        alms = np.zeros(self.alm_ord.getsize(3*self.nside-1),
                        dtype='complex128')
        alms[idx] = rans[0]+1j*rans[1]
        return hp.alm2map(alms, self.nside)

    def _dsim_default(self, seed, ell):
        np.random.seed(seed)
        if self.dsim_d['stats'] == 'Gaussian':
            return self._gen_gaussian_alm(ell)
        elif self.dsim_d['stats'] == 'Z2':
            return self._gen_Z2_alm(ell)
        else:
            raise ValueError("Only Gaussian sims implemented")

    def _filt_default(self, mp_true):
        return self.filt_d['mask']*mp_true

    def gen_deltasim(self, seed, ell):
        dsim_true = self.dsim(seed, ell)
        dsim_filt = self.filt(dsim_true)
        return dsim_filt

    def _file_exists(self, isim, ell):
        if self.prefix_save is None:
            return False, None
        else:
            fname = os.path.join(self.prefix_save,
                                 f'cb_ell{ell}_sim{isim}.npz')
            isfile = os.path.isfile(fname)
            if isfile:
                warnings.warn("Reading from file. Run `cleanup` "
                              "if you'd like to recompute bandpowers.",
                              Warning)
            return isfile, fname

    def gen_deltasim_bpw(self, isim, ell):
        isfile, fname = self._file_exists(isim, ell)
        seed = ell*self.nsim_per_ell + isim
        if isfile:
            return np.load(fname)['cb']
        dsim = self.gen_deltasim(seed, ell)
        cb = self.bins.bin_cell(hp.anafast(dsim, iter=self.n_iter))
        if fname is not None:
            np.savez(fname, cb=cb)
        return cb

    def gen_Bbl_at_ell(self, ell):
        Bbl = np.zeros(self.bins.get_n_bands())
        for isim in range(self.nsim_per_ell):
            cb = self.gen_deltasim_bpw(isim, ell)
            Bbl += cb
        Bbl /= self.nsim_per_ell
        return Bbl

    def get_ells(self):
        return np.arange(self.lmin, self.lmax+1)

    def gen_Bbl_all(self, verbose=False, every=10):
        Bbls = []
        for l in self.get_ells():
            if verbose:
                if l % every == 0:
                    print(f" - ell = {l}")
            Bbl = self.gen_Bbl_at_ell(l)
            Bbls.append(Bbl)
        Bbls = np.array(Bbls).T
        return Bbls
