"""
This module defines useful functions to compute covariance matrices
"""
from itertools import product
import pymaster as nmt
import numpy as np
import os
from soopercool import map_utils as mu


def load_covariance_workspace(cov_dir):
    """
    """
    spin_pairs = list(product(["spin0", "spin2"], repeat=2))
    spin_combos = [
        (s0, s1, s2, s3)
        for (s0, s1) in spin_pairs
        for (s2, s3) in spin_pairs
    ]
    cwsp = {}
    for s0, s1, s2, s3 in spin_combos:
        cwsp[s0, s1, s2, s3] = nmt.NmtCovarianceWorkspace.from_file(
            f"{cov_dir}/cwsp_{s0}_{s1}_{s2}_{s3}"
        )
        for kind in ["snsn", "snns", "nssn", "nsns", "nn"]:
            cwsp[s0, s1, s2, s3, kind] = \
                nmt.NmtCovarianceWorkspace.from_file(
                    f"{cov_dir}/cwsp_{kind}_{s0}_{s1}_{s2}_{s3}"
                )

    return cwsp


def load_workspace(cov_dir):
    """
    """
    spin_pairs = list(product(["spin0", "spin2"], repeat=2))
    wsp = {}
    for s0, s1 in spin_pairs:
        wsp[s0, s1] = nmt.NmtWorkspace.from_file(
            f"{cov_dir}/wsp_{s0}_{s1}"
        )
    return wsp


def compute_covariance_workspace(analysis_mask,
                                 nmt_bins,
                                 lmax,
                                 wcs=None,
                                 purify_b=False,
                                 save_dir=None,
                                 hits_map=None):
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
    hits_map: array, optional
        If provided, use hits map to also save noise
        covariance workspace.

    Returns
    -------
    wsp: dict
        Dictionary of NmtWorkspace objects for the given analysis mask and
        binning.
    cwsp: dict
        Dictionary of NmtCovarianceWorkspace objects for the given analysis
        mask and binning.
    """
    print("Getting fields")
    fields = {}
    fields["spin0"] = nmt.NmtField(
        analysis_mask,
        maps=None,
        wcs=wcs,
        spin=0,
        lmax=lmax
    )
    fields["spin2"] = nmt.NmtField(
        analysis_mask,
        maps=None,
        purify_b=purify_b,
        wcs=wcs,
        spin=2,
        lmax=lmax
    )

    if hits_map is not None:
        print("Using hits map to get noise inhom.")

        sigma = hits_map.copy()
        sigma[hits_map > 0] = 1 / np.sqrt(hits_map[hits_map > 0])
        mask_noise = analysis_mask * sigma
        pix_type = "car" if hasattr(analysis_mask, "geometry") else "hp"
        rescale_noise = mu.sky_average(
            (sigma * analysis_mask) ** 2,
            pix_type=pix_type
        ) / mu.sky_average(
            analysis_mask ** 2,
            pix_type=pix_type
        )

        fields["spin0", "noise"] = nmt.NmtField(
            mask_noise / np.sqrt(rescale_noise),
            maps=None,
            wcs=wcs,
            spin=0,
            lmax=lmax
        )
        fields["spin2", "noise"] = nmt.NmtField(
            mask_noise / np.sqrt(rescale_noise),
            maps=None,
            purify_b=purify_b,
            wcs=wcs,
            spin=2,
            lmax=lmax
        )

    spin_pairs = list(product(["spin0", "spin2"], repeat=2))

    wsp = {}
    cwsp = {}
    for i, (s0, s1) in enumerate(spin_pairs):
        if save_dir is not None:
            fname_wsp = f"{save_dir}/wsp_{s0}_{s1}"
        else:
            fname_wsp = None

        # Only needs to compute it for signal,
        # optionally with purification
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

            # Signal-Signal part
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
            # Signal-Noise part (if hits_map is provided)
            if hits_map is not None:

                # snsn
                if save_dir is not None:
                    fname_cwsp = f"{save_dir}/cwsp_snsn_{s0}_{s1}_{s2}_{s3}"
                else:
                    fname_cwsp = None
                if (not os.path.exists(fname_cwsp)):
                    cwsp[s0, s1, s2, s3, "snsn"] = nmt.NmtCovarianceWorkspace(
                        fields[s0],
                        fields[s1, "noise"],
                        fields[s2],
                        fields[s3, "noise"]
                    )
                    if fname_cwsp is not None:
                        cwsp[s0, s1, s2, s3, "snsn"].write_to(fname_cwsp)
                else:
                    cwsp[s0, s1, s2, s3, "snsn"] = \
                        nmt.NmtCovarianceWorkspace.from_file(
                            fname_cwsp
                        )
                # snns
                if save_dir is not None:
                    fname_cwsp = f"{save_dir}/cwsp_snns_{s0}_{s1}_{s2}_{s3}"
                else:
                    fname_cwsp = None
                if (not os.path.exists(fname_cwsp)):
                    cwsp[s0, s1, s2, s3, "snns"] = nmt.NmtCovarianceWorkspace(
                        fields[s0],
                        fields[s1, "noise"],
                        fields[s2, "noise"],
                        fields[s3]
                    )
                    if fname_cwsp is not None:
                        cwsp[s0, s1, s2, s3, "snns"].write_to(fname_cwsp)

                else:
                    cwsp[s0, s1, s2, s3, "snns"] = \
                        nmt.NmtCovarianceWorkspace.from_file(
                            fname_cwsp
                        )

                # nssn
                if save_dir is not None:
                    fname_cwsp = f"{save_dir}/cwsp_nssn_{s0}_{s1}_{s2}_{s3}"
                else:
                    fname_cwsp = None
                if (not os.path.exists(fname_cwsp)):
                    cwsp[s0, s1, s2, s3, "nssn"] = nmt.NmtCovarianceWorkspace(
                        fields[s0, "noise"],
                        fields[s1],
                        fields[s2],
                        fields[s3, "noise"]
                    )
                    if fname_cwsp is not None:
                        cwsp[s0, s1, s2, s3, "nssn"].write_to(fname_cwsp)
                else:
                    cwsp[s0, s1, s2, s3, "nssn"] = \
                        nmt.NmtCovarianceWorkspace.from_file(
                            fname_cwsp
                        )

                # nsns
                if save_dir is not None:
                    fname_cwsp = f"{save_dir}/cwsp_nsns_{s0}_{s1}_{s2}_{s3}"
                else:
                    fname_cwsp = None
                if (not os.path.exists(fname_cwsp)):
                    cwsp[s0, s1, s2, s3, "nsns"] = nmt.NmtCovarianceWorkspace(
                        fields[s0, "noise"],
                        fields[s1],
                        fields[s2, "noise"],
                        fields[s3]
                    )
                    if fname_cwsp is not None:
                        cwsp[s0, s1, s2, s3, "nsns"].write_to(fname_cwsp)
                else:
                    cwsp[s0, s1, s2, s3, "nsns"] = \
                        nmt.NmtCovarianceWorkspace.from_file(
                            fname_cwsp
                        )

                # Noise-Noise part
                if save_dir is not None:
                    fname_cwsp = f"{save_dir}/cwsp_nn_{s0}_{s1}_{s2}_{s3}"
                else:
                    fname_cwsp = None
                if (not os.path.exists(fname_cwsp)):
                    cwsp[s0, s1, s2, s3, "nn"] = nmt.NmtCovarianceWorkspace(
                        fields[s0, "noise"],
                        fields[s1, "noise"],
                        fields[s2, "noise"],
                        fields[s3, "noise"]
                    )
                    if fname_cwsp is not None:
                        cwsp[s0, s1, s2, s3, "nn"].write_to(fname_cwsp)
                else:
                    cwsp[s0, s1, s2, s3, "nn"] = \
                        nmt.NmtCovarianceWorkspace.from_file(
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
    Compute covariance block for the map_set1 x map_set2 and
    map_set3 x map_set4 cross-spectra.
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
    cwsp_keys = list(cwsp.keys())
    # filter out only the 4 spin values
    cwsp_keys = [k for k in cwsp_keys if len(k) == 4]
    for s1, s2, s3, s4 in cwsp_keys:

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

        # cl13, cl14, nl23, nl24
        factor24 = factor + n1 * n2 * n3 * (exp2 == exp4)
        factor24 -= n1 * n3 * ((exp1 == exp2 == exp4) + (exp2 == exp3 == exp4))

        factor23 = factor + n1 * n2 * n4 * (exp2 == exp3)
        factor23 -= n1 * n4 * ((exp1 == exp2 == exp3) + (exp2 == exp3 == exp4))

        factor24 /= Npairs_12 * Npairs_34
        factor23 /= Npairs_12 * Npairs_34

        # This is the old way of doing it.
        # With noise inhomogeneities, we need to split those contributions
        # covar += nmt.gaussian_covariance(
        #     cwsp[s1, s2, s3, s4],
        #     int(s1[-1]), int(s2[-1]),
        #     int(s3[-1]), int(s4[-1]),
        #     cl13,
        #     cl14,
        #     nl23 * factor23,
        #     nl24 * factor24,
        #     wa=wsp[s1, s2],
        #     wb=wsp[s3, s4]
        # )
        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4, "snsn"],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            cl13,
            cl14*0.,
            nl23*0.,
            nl24 * factor24,
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )
        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4, "snns"],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            cl13 * 0.,
            cl14,
            nl23 * factor23,
            nl24 * 0.,
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

        # covar += nmt.gaussian_covariance(
        #     cwsp[s1, s2, s3, s4],
        #     int(s1[-1]), int(s2[-1]),
        #     int(s3[-1]), int(s4[-1]),
        #     nl13 * factor13,
        #     nl14 * factor14,
        #     cl23,
        #     cl24,
        #     wa=wsp[s1, s2],
        #     wb=wsp[s3, s4]
        # )
        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4, "nssn"],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            nl13 * 0.,
            nl14 * factor14,
            cl23,
            cl24 * 0.,
            wa=wsp[s1, s2],
            wb=wsp[s3, s4]
        )
        covar += nmt.gaussian_covariance(
            cwsp[s1, s2, s3, s4, "nsns"],
            int(s1[-1]), int(s2[-1]),
            int(s3[-1]), int(s4[-1]),
            nl13 * factor13,
            nl14 * 0.,
            cl23 * 0.,
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
            cwsp[s1, s2, s3, s4, "nn"],
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
