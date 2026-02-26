from scipy.optimize import minimize
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import sacc
import os
import healpy as hp


def main(args):
    sacc_file = args.sacc_file
    lmin = args.lmin_fit
    lmax = args.lmax_fit

    refs = args.map_sets_ref
    refs = refs.split(",")
    to_fit = args.map_sets_to_fit
    to_fit = to_fit.split(",")

    out_dir = "EE_cals_and_beams"
    os.makedirs(out_dir, exist_ok=True)

    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices("cl_ee", ell__gt=lmin, ell__lt=lmax)
    s.keep_indices(idx)

    tracers = list(s.tracers.keys())
    for ref in refs:
        if ref not in tracers:
            raise ValueError(
                f"Reference map set {ref} not found in the SACC file"
            )
    if not all([t in tracers for t in to_fit]):
        raise ValueError(
            f"Some map sets to fit not found in the SACC file. "
            f"Found tracers are {tracers}"
        )

    ps_vec = s.mean
    cov = s.covariance.covmat

    cals = {}
    fwhms = {}
    for tr_to_fit, tr_ref in zip(to_fit, refs):
        if tr_to_fit == tr_ref:
            raise ValueError(
                f"Map set {tr_to_fit} is both in the reference and to-fit list"
            )

        idx_00 = s.indices("cl_ee", (tr_ref, tr_ref))
        idx_01 = s.indices("cl_ee", (tr_ref, tr_to_fit))
        if len(idx_01) == 0:
            idx_01 = s.indices("cl_ee", (tr_to_fit, tr_ref))
        idx_11 = s.indices("cl_ee", (tr_to_fit, tr_to_fit))

        idx = np.concatenate([idx_00, idx_01, idx_11])
        # bpw_00 = s.get_bandpower_windows(idx_00).weight.T
        bpw_01 = s.get_bandpower_windows(idx_01).weight.T
        bpw_11 = s.get_bandpower_windows(idx_11).weight.T

        nl = len(idx) // 3

        def tomin(p):
            cal = p[0]
            fwhm_amin = p[1]
            fwhm_deg = fwhm_amin / 60.0
            fwhm_rad = np.deg2rad(fwhm_deg)

            bl = hp.gauss_beam(fwhm=fwhm_rad, lmax=2000)
            ell = np.arange(len(bl))
            bl_rescale = bl / ell / (ell+1) * 2 * np.pi
            bl_rescale[0] = 0.
            # bl_00 = bpw_00 @ bl_rescale[: bpw_00.shape[-1]]
            bl_01 = bpw_01 @ bl_rescale[: bpw_01.shape[-1]]
            bl_11 = bpw_11 @ bl_rescale[: bpw_11.shape[-1]]

            P = np.zeros((2 * nl, len(idx)))

            P[:nl, :nl] = -np.eye(nl)
            P[:nl, nl: 2 * nl] = cal * np.eye(nl) / bl_01
            P[:nl, 2 * nl:] = 0.0

            P[nl:, :nl] = -np.eye(nl)
            P[nl:, nl:2 * nl] = 0.0
            P[nl:, 2 * nl:] = cal**2 * np.eye(nl) / bl_11 ** 2

            res = P @ ps_vec[idx]
            rescov = P @ cov[np.ix_(idx, idx)] @ P.T
            rescov = np.diag(rescov.diagonal())

            invcov = np.linalg.inv(rescov)

            return res.T @ invcov @ res

        res = minimize(
            tomin,
            x0=np.array([1.0, 20.0]),
            method="L-BFGS-B",
        )
        cal, fwhm = res.x[0], res.x[1]
        print(
            f"Calibration factor for {tr_to_fit} relative to {tr_ref}: {cal:.4f}" # noqa
        )
        print(f"Best-fit beam FWHM for {tr_to_fit}: {fwhm:.2f} arcmin")

        cals[tr_to_fit] = cal
        fwhms[tr_to_fit] = fwhm

    # Get full multipole range
    print("Reading full sacc file")
    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices(ell__lt=args.lmax_sacc)
    s.keep_indices(idx)

    ps_vec = s.mean
    cov = s.covariance.covmat
    # idx_ref = s.indices("cl_ee", (ref, ref))

    # Save calibs
    with open(f"{out_dir}/calibrations_lmin{lmin}_lmax{lmax}.pkl", "wb") as f:
        pickle.dump(cals, f)
    with open(f"{out_dir}/beam_fwhms_lmin{lmin}_lmax{lmax}.pkl", "wb") as f:
        pickle.dump(fwhms, f)

    # Save sacc file
    s_out = sacc.Sacc()
    for tracer_name, tracer in s.tracers.items():
        s_out.add_tracer(
            tracer.tracer_type,
            tracer_name,
            quantity=tracer.quantity,
            spin=tracer.spin,
            nu=tracer.nu,
            ell=tracer.ell,
            beam=tracer.beam,
            bandpass=tracer.bandpass,
        )
    tot_idx = []
    cal_vec = []
    bl_vec = []
    for dtype in s.get_data_types():
        for t1, t2 in s.get_tracer_combinations():
            idx = s.indices(dtype, tracers=(t1, t2))
            f1, f2 = dtype.split("_")[-1]
            if f1 in ["e", "b"]:
                c1 = cals.get(t1, 1.0)
            else:
                c1 = 1.0
            if f2 in ["e", "b"]:
                c2 = cals.get(t2, 1.0)
            else:
                c2 = 1.0

            fwhm1 = fwhms.get(t1, 0.0)
            fwhm2 = fwhms.get(t2, 0.0)
            if fwhm1 != 0.0:
                fwhm_deg = fwhm1 / 60.0
                fwhm_rad = np.deg2rad(fwhm_deg)
                bl1 = hp.gauss_beam(fwhm=fwhm_rad, lmax=2000)
                ell = np.arange(len(bl1))
                bl_rescale1 = bl1 / ell / (ell+1) * 2 * np.pi
                bl_rescale1[0] = 0.
                bpw = s.get_bandpower_windows(idx).weight.T
                bl_rescale1 = bpw @ bl_rescale1[: bpw.shape[-1]]
            else:
                bl_rescale1 = 1

            if fwhm2 != 0.0:
                fwhm_deg = fwhm2 / 60.0
                fwhm_rad = np.deg2rad(fwhm_deg)
                bl2 = hp.gauss_beam(fwhm=fwhm_rad, lmax=2000)
                ell = np.arange(len(bl2))
                bl_rescale2 = bl2 / ell / (ell+1) * 2 * np.pi
                bl_rescale2[0] = 0.
                bpw = s.get_bandpower_windows(idx).weight.T
                bl_rescale2 = bpw @ bl_rescale2[: bpw.shape[-1]]
            else:
                bl_rescale2 = 1
            lb, _ = s.get_ell_cl(data_type=dtype, tracer1=t1, tracer2=t2)
            s_out.add_ell_cl(
                **{
                    "data_type": dtype,
                    "tracer1": t1,
                    "tracer2": t2,
                    "ell": lb,
                    "x": ps_vec[idx] * c1 * c2 / bl_rescale1 / bl_rescale2,
                    "window": s.get_bandpower_windows(idx),
                }
            )
            tot_idx += list(idx)
            cal_vec.append(
                np.full_like(idx, c1 * c2)
            )
            bl = bl_rescale1 * bl_rescale2
            # test if bl is an array
            if isinstance(bl, np.ndarray):
                bl_vec.append(bl)
            else:
                bl_vec.append(np.full_like(idx, bl))
    cal_vec = np.concatenate(cal_vec)
    bl_vec = np.concatenate(bl_vec)
    cov_cal = (
        cov[np.ix_(tot_idx, tot_idx)]
        * np.outer(cal_vec, cal_vec)
        / np.outer(bl_vec, bl_vec)
    )
    s_out.add_covariance(cov_cal)
    fname = f"{out_dir}/cl_and_cov_sacc_calibrated_lmin{lmin}_lmax{lmax}.fits"
    s_out.save_fits(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate based on the E-mode map of a reference map set"
    )
    parser.add_argument(
        "--sacc-file", type=str, help="Path to the SACC fits file"
    )
    parser.add_argument("--lmin-fit", type=int, help="Minimum multipole")
    parser.add_argument("--lmax-fit", type=int, help="Maximum multipole")
    parser.add_argument(
        "--lmax-sacc",
        type=int,
        help="Maximum multipole to keep in the output SACC file",
    )
    parser.add_argument(
        "--map-sets-ref",
        type=str,
        help="Reference map sets for the EE calibration fit",
    )
    parser.add_argument(
        "--map-sets-to-fit",
        type=str,
        help="Map sets to fit for the EE calibration (default: all map sets)",
    )
    args = parser.parse_args()

    main(args)
