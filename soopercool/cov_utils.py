"""
This module defines useful functions to compute covariance matrices
"""
from itertools import product
import pymaster as nmt
import numpy as np
import os


def compute_covariance_workspace(analysis_mask,
                                 nmt_bins,
                                 wcs=None,
                                 purify_b=False,
                                 save_dir=None):
    """
    Return dictionary of NmtWorkspace and NmtCovarianceWorkspace objects
    for the given analysis mask and binning.

    Parameters
    ----------
    analysis_mask : array
        The analysis mask to use for the covariance workspace. Can be either
        CAR or HEALPix
    nmt_bins : NmtBin object
        The binning to use for the covariance workspace.
    wcs : WCS object, optional
        Only required if the analysis mask is a CAR map.
    purify_b : bool, optional
        Whether to purify the B-mode map. Default is False.*

    Returns
    -------
    wsp: dict
        Dictionary of NmtWorkspace objects for the given analysis mask and
        binning.
    cwsp: dict
        Dictionary of NmtCovarianceWorkspace objects for the given analysis
        mask and binning.
    """
    fields = {}
    fields["spin0"] = nmt.NmtField(
        analysis_mask,
        maps=None,
        purify_b=True,
        wcs=wcs,
        spin=0
    )
    fields["spin2"] = nmt.NmtField(
        analysis_mask,
        maps=None,
        purify_b=purify_b,
        wcs=wcs,
        spin=2
    )
    spin_pairs = list(product(["spin0", "spin2"], repeat=2))

    wsp = {}
    cwsp = {}
    for i, (s0, s1) in enumerate(spin_pairs):
        if save_dir is not None:
            fname_wsp = f"{save_dir}/wsp_{s0}_{s1}"
        else:
            fname_wsp = None

        if not os.path.exists(fname_wsp):
            wsp[s0, s1] = nmt.NmtWorkspace(
                fields[s0],
                fields[s1],
                nmt_bins
            )
            if fname_wsp is not None:
                wsp[s0, s1].write_to(fname_wsp)
        else:
            wsp[s0, s1] = nmt.NmtWorkspace.from_file(fname_wsp)

        for j, (s2, s3) in enumerate(spin_pairs):
            if save_dir is not None:
                fname_cwsp = f"{save_dir}/cwsp_{s0}_{s1}_{s2}_{s3}"
            else:
                fname_cwsp = None
            # if i > j:
            #    continue

            if (not os.path.exists(fname_cwsp)):
                cwsp[s0, s1, s2, s3] = nmt.NmtCovarianceWorkspace(
                    fields[s0],
                    fields[s1],
                    fields[s2],
                    fields[s3]
                )
                if fname_cwsp is not None:
                    cwsp[s0, s1, s2, s3].write_to(fname_cwsp)
            else:
                cwsp[s0, s1, s2, s3] = nmt.NmtCovarianceWorkspace.from_file(
                    fname_cwsp
                )
    return wsp, cwsp


def compute_covariance_block(map_set1,
                             map_set2,
                             map_set3,
                             map_set4,
                             exp1,
                             exp2,
                             exp3,
                             exp4,
                             wsp,
                             cwsp,
                             signal,
                             noise,
                             nmt_bins,
                             n_bundles):
    """
    """
    n1 = n_bundles[map_set1]
    n2 = n_bundles[map_set2]
    n3 = n_bundles[map_set3]
    n4 = n_bundles[map_set4]

    Npairs_12 = n1 * (n2 - (exp1 == exp2))
    Npairs_34 = n3 * (n4 - (exp3 == exp4))

    lb = nmt_bins.get_effective_ells()

    field_pairs = {
        ("spin0", "spin0"): ["TT"],
        ("spin0", "spin2"): ["TE", "TB"],
        ("spin2", "spin0"): ["ET", "BT"],
        ("spin2", "spin2"): ["EE", "EB", "BE", "BB"]
    }

    covs = {}
    for s1, s2, s3, s4 in cwsp.keys():

        fps13 = field_pairs[s1, s3]
        fps14 = field_pairs[s1, s4]
        fps23 = field_pairs[s2, s3]
        fps24 = field_pairs[s2, s4]

        cl13 = np.array([signal[map_set1, map_set3, fp] for fp in fps13])
        nl13 = np.array([noise[map_set1, map_set3, fp] for fp in fps13])
        cl14 = np.array([signal[map_set1, map_set4, fp] for fp in fps14])
        nl14 = np.array([noise[map_set1, map_set4, fp] for fp in fps14])
        cl23 = np.array([signal[map_set2, map_set3, fp] for fp in fps23])
        nl23 = np.array([noise[map_set2, map_set3, fp] for fp in fps23])
        cl24 = np.array([signal[map_set2, map_set4, fp] for fp in fps24])
        nl24 = np.array([noise[map_set2, map_set4, fp] for fp in fps24])

        # cl11, cl14, cl23, cl24
        covar = nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            cl13,
            cl14,
            cl23,
            cl24,
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )
        factor = n1 * (exp1 == exp2 == exp3 == exp4)

        # cl11, cl14, nl23, nl24
        factor24 = factor + n1 * n2 * n3 * (exp2 == exp4)
        factor24 -= n1 * n3 * ((exp1 == exp2 == exp4) + (exp2 == exp3 == exp4))

        factor23 = factor + n1 * n2 * n4 * (exp2 == exp3)
        factor23 -= n1 * n4 * ((exp1 == exp2 == exp3) + (exp2 == exp3 == exp4))

        factor24 /= Npairs_12 * Npairs_34
        factor23 /= Npairs_12 * Npairs_34

        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            cl13,
            cl14,
            nl23 * factor23,
            nl24 * factor24,
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )

        # nl13, nl14, cl23, cl24
        factor13 = factor + n1 * n2 * n4 * (exp1 == exp3)
        factor13 -= n2 * n4 * ((exp1 == exp2 == exp3) + (exp1 == exp3 == exp4))

        factor14 = factor + n1 * n2 * n3 * (exp1 == exp4)
        factor14 -= n2 * n3 * ((exp1 == exp2 == exp4) + (exp1 == exp3 == exp4))

        factor13 /= Npairs_12 * Npairs_34
        factor14 /= Npairs_12 * Npairs_34

        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            nl13 * factor13,
            nl14 * factor14,
            cl23,
            cl24,
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )

        # nl13, nl14, nl23, nl24
        factor_1324 = n1 * n2 * ((exp1 == exp3) & (exp2 == exp4))
        factor_1324 -= n1 * (exp1 == exp2 == exp3 == exp4)

        factor_1423 = n1 * n2 * ((exp1 == exp4) & (exp2 == exp3))
        factor_1423 -= n1 * (exp1 == exp2 == exp3 == exp4)

        factor_1324 /= Npairs_12 * Npairs_34
        factor_1423 /= Npairs_12 * Npairs_34

        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            nl13 * np.sqrt(factor_1324),
            nl14 * np.sqrt(factor_1423),
            nl23 * np.sqrt(factor_1423),
            nl24 * np.sqrt(factor_1324),
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )

        fp12 = field_pairs[s1, s2]
        fp34 = field_pairs[s3, s4]

        covar = covar.reshape([
            len(lb),
            len(fp12),
            len(lb),
            len(fp34)
        ])

        for i, XY in enumerate(fp12):
            for j, UV in enumerate(fp34):
                covs[XY, UV] = covar[:, i, :, j]
    return covs
