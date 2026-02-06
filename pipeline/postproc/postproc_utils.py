import numpy as np
import healpy as hp


def EBrot_mat(s, EBangles_dict):
    """
    Compute the matrix that rotates E and B modes by the angles
    specified in EBangles_dict. The angles should be in degrees

    Parameters
    ----------
    s : sacc.Sacc object
        The sacc object containing the data and covariance to be rotated
    EBangles_dict : dict
        A dictionary where the keys are tracer names and values in degrees

    Returns
    -------
    rot_mat : np.ndarray
        The rotation matrix that can be applied to the data vector
        and covariance matrix s.mean and s.covariance.covmat
    """
    rot_mat = np.eye(s.covariance.covmat.shape[0])
    for dtype in s.get_data_types():
        # if "0" in dtype:
        #     continue
        for t1, t2 in s.get_tracer_combinations():

            out_idx = s.indices(dtype, (t1, t2))
            try:
                alpha1 = EBangles_dict[t1]
            except KeyError:
                alpha1 = 0.0
            try:
                alpha2 = EBangles_dict[t2]
            except KeyError:
                alpha2 = 0.0

            c1 = np.cos(2 * np.deg2rad(alpha1))
            c2 = np.cos(2 * np.deg2rad(alpha2))
            s1 = np.sin(2 * np.deg2rad(alpha1))
            s2 = np.sin(2 * np.deg2rad(alpha2))

            if dtype == "cl_ee":
                idxEE = s.indices("cl_ee", (t1, t2))
                idxBB = s.indices("cl_bb", (t1, t2))
                idxEB = s.indices("cl_eb", (t1, t2))
                idxBE = s.indices("cl_be", (t1, t2))

                rot_mat[out_idx, idxEE] = c1 * c2
                rot_mat[out_idx, idxBB] = s1 * s2
                rot_mat[out_idx, idxEB] = c1 * s2
                rot_mat[out_idx, idxBE] = s1 * c2

            elif dtype == "cl_bb":
                idxEE = s.indices("cl_ee", (t1, t2))
                idxBB = s.indices("cl_bb", (t1, t2))
                idxEB = s.indices("cl_eb", (t1, t2))
                idxBE = s.indices("cl_be", (t1, t2))

                rot_mat[out_idx, idxEE] = s1 * s2
                rot_mat[out_idx, idxBB] = c1 * c2
                rot_mat[out_idx, idxEB] = -s1 * c2
                rot_mat[out_idx, idxBE] = -c1 * s2

            elif dtype == "cl_eb":
                idxEE = s.indices("cl_ee", (t1, t2))
                idxBB = s.indices("cl_bb", (t1, t2))
                idxEB = s.indices("cl_eb", (t1, t2))
                idxBE = s.indices("cl_be", (t1, t2))

                rot_mat[out_idx, idxEE] = -c1 * s2
                rot_mat[out_idx, idxBB] = s1 * c2
                rot_mat[out_idx, idxEB] = c1 * c2
                rot_mat[out_idx, idxBE] = -s1 * s2

            elif dtype == "cl_be":
                idxEE = s.indices("cl_ee", (t1, t2))
                idxBB = s.indices("cl_bb", (t1, t2))
                idxEB = s.indices("cl_eb", (t1, t2))
                idxBE = s.indices("cl_be", (t1, t2))

                rot_mat[out_idx, idxEE] = -s1 * c2
                rot_mat[out_idx, idxBB] = c1 * s2
                rot_mat[out_idx, idxEB] = -s1 * s2
                rot_mat[out_idx, idxBE] = c1 * c2

            elif dtype == "cl_0e":
                idxTE = s.indices("cl_0e", (t1, t2))
                idxTB = s.indices("cl_0b", (t1, t2))

                rot_mat[out_idx, idxTE] = c2
                rot_mat[out_idx, idxTB] = s2

            elif dtype == "cl_0b":
                idxTE = s.indices("cl_0e", (t1, t2))
                idxTB = s.indices("cl_0b", (t1, t2))

                rot_mat[out_idx, idxTE] = -s2
                rot_mat[out_idx, idxTB] = c2

            elif dtype == "cl_e0":
                idxET = s.indices("cl_e0", (t1, t2))
                idxBT = s.indices("cl_b0", (t1, t2))

                rot_mat[out_idx, idxET] = c1
                rot_mat[out_idx, idxBT] = s1

            elif dtype == "cl_b0":
                idxET = s.indices("cl_e0", (t1, t2))
                idxBT = s.indices("cl_b0", (t1, t2))

                rot_mat[out_idx, idxET] = -s1
                rot_mat[out_idx, idxBT] = c1

    return rot_mat


def get_ps_vec_model(s, lth, clth, beam_fwhms=None, cals=None):
    """
    Compute model power spectrum and bin it according
    to the bandpower window functions stored in s.

    Parameters
    ----------
    s : sacc.Sacc object
        The sacc object containing the data and covariance to be rotated
    lth : array-like
        The multipoles at which the theory power spectrum is evaluated
    clth : dict
        A dictionary where the keys are "TT", "EE", "BB", "TE", "EB", "BE" and
        the values are the corresponding theory power spectra evaluated at lth
    beam_fwhms : dict, optional
        A dictionary where the keys are tracer names and values are the beam
        FWHM in arcminutes. If None, no beam is applied.
        Default is None.
    cals : dict, optional
        A dictionary where the keys are tracer names and values are the
        calibration factors. If None, no calibration is applied.
        Default is None.
    """
    ps_vec_model = np.zeros_like(s.mean)
    for dtype in s.get_data_types():
        for t1, t2 in s.get_tracer_combinations():

            fp = dtype.replace("0", "t")
            fp = fp.split("_")[-1]
            fp = fp.upper()

            idx = s.indices(dtype, tracers=(t1, t2))
            bpw = s.get_bandpower_windows(idx).weight.T

            if beam_fwhms is not None:
                bl1 = hp.gauss_beam(
                    fwhm=np.deg2rad(beam_fwhms[t1] / 60), lmax=lth[-1]
                )[: bpw.shape[-1]]
                bl2 = hp.gauss_beam(
                    fwhm=np.deg2rad(beam_fwhms[t2] / 60), lmax=lth[-1]
                )[: bpw.shape[-1]]
            else:
                bl1 = 1.0
                bl2 = 1.0
            if cals is not None:
                c1 = cals[t1]
                c2 = cals[t2]
            else:
                c1 = 1.0
                c2 = 1.0

            theory = clth[fp][: bpw.shape[-1]] * (bl1 * bl2) * (c1 * c2)
            ps_vec_model[idx] = bpw @ theory

    return ps_vec_model
