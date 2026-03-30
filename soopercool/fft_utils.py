import numpy as np
from pixell import enmap


def kspace_filter(m, dkx=0, dky=0, type="sharp"):
    """
    Filter the Fourier space of the map `m` by zeroing out modes with |kx| < dkx or |ky| < dky.

    Parameters
    ----------
    m : enmap
        The input map to be filtered.
    dkx : float, optional
        The cutoff in the kx direction. Modes with |kx| < dkx will be filtered out. Default is 0 (no filtering).
    dky : float, optional
        The cutoff in the ky direction. Modes with |ky| < dky will be filtered out. Default is 0 (no filtering).
    type : str, optional
        The type of filter to apply. Options are "sharp" (default) for a hard cutoff, and "cosine" for a smooth cosine taper.
    """
    ky, kx = m.lmap()
    mf = enmap.fft(m)

    if type == "sharp":
        msk_x = np.abs(kx) < dkx
        msk_y = np.abs(ky) < dky
        msk = msk_x | msk_y
        mf[:, msk] = 0.
    if type == "cosine":
        w_x = cosine_taper(np.abs(kx), 0, dkx)
        w_y = cosine_taper(np.abs(ky), 0, dky)
        mf *= w_x * w_y
    mf_back = enmap.ifft(mf)
    return mf_back


def cosine_taper(k, kmin, kmax):
    """
    """
    if kmax == kmin:
        return np.ones_like(k)
    w = np.zeros_like(k)
    mask = (k >= kmin) & (k <= kmax)
    w[mask] = 0.5 * (1 - np.cos(np.pi * (k[mask] - kmin) / (kmax - kmin)))
    w[k > kmax] = 1.0
    return w
