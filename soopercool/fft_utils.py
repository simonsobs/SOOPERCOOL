import numpy as np
from pixell import enmap


def kspace_filter(m, dkx=0, dky=0, type="sharp", pix_type="car"):
    """
    Filter the Fourier space of the map `m`

    Parameters
    ----------
    m : enmap
        The input map to be filtered.
    dkx : float, optional
        The cutoff in the kx direction.
        Modes with |kx| < dkx will be filtered out.
        Default is 0 (no filtering).
    dky : float, optional
        The cutoff in the ky direction.
        Modes with |ky| < dky will be filtered out.
        Default is 0 (no filtering).
    type : str, optional
        The type of filter to apply. 
        Options are "sharp" (default) for a hard cutoff
        and "cosine" for a smooth cosine taper.
    """
    if pix_type == "hp":
        raise NotImplementedError(
            "k-space filtering is currently only implemented for CAR maps."
        )

    elif pix_type == "car":
        ky, kx = m.lmap()
        mf = enmap.fft(m)

        if type == "sharp":
            msk_x = np.abs(kx) < dkx
            msk_y = np.abs(ky) < dky
            msk = msk_x | msk_y
            mf[:, msk] = 0.0
        if type == "cosine":
            w_x = cosine_taper(np.abs(kx), 0, dkx)
            w_y = cosine_taper(np.abs(ky), 0, dky)
            mf *= w_x * w_y
        mf_back = enmap.ifft(mf)
        return mf_back


def cosine_taper(k, kmin, kmax):
    """
    Utility function to create a cosine taper filter in Fourier space.
    Transition smoothly from 0 to 1 between kmin and kmax.

    Parameters
    ----------
    k : numpy.ndarray
        The wavenumber array.
    kmin : float
        The minimum wavenumber for the taper.
    kmax : float
        The maximum wavenumber for the taper.

    Returns
    -------
    numpy.ndarray
        The taper filter.
    """
    if kmax == kmin:
        return np.ones_like(k)
    w = np.zeros_like(k)
    mask = (k >= kmin) & (k <= kmax)
    w[mask] = 0.5 * (1 - np.cos(np.pi * (k[mask] - kmin) / (kmax - kmin)))
    w[k > kmax] = 1.0
    return w
