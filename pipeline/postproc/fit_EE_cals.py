from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle as p
import numpy as np
import argparse
import sacc
import os


def main(args):
    sacc_file = args.sacc_file
    lmin = args.lmin_fit
    lmax = args.lmax_fit

    ref = args.map_set_ref
    to_fit = args.map_sets_to_fit
    to_fit = to_fit.split(",")

    out_dir = "EE_cals"
    os.makedirs(out_dir, exist_ok=True)

    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices("cl_ee", ell__gt=lmin, ell__lt=lmax)
    s.keep_indices(idx)

    tracers = list(s.tracers.keys())
    if ref not in tracers:
        raise ValueError(f"Reference map set {ref} not found in the SACC file")
    if not all([t in tracers for t in to_fit]):
        raise ValueError(
            f"Some map sets to fit not found in the SACC file. "
            f"Found tracers are {tracers}"
        )

    ps_vec = s.mean
    cov = s.covariance.covmat

    cals = {}
    for tr_to_fit in to_fit:
        if tr_to_fit == ref:
            raise ValueError(
                f"Map set {tr_to_fit} is both in the reference and to-fit list"
            )

        idx_00 = s.indices("cl_ee", (ref, ref))
        idx_01 = s.indices("cl_ee", (ref, tr_to_fit))
        if len(idx_01) == 0:
            idx_01 = s.indices("cl_ee", (tr_to_fit, ref))
        idx_11 = s.indices("cl_ee", (tr_to_fit, tr_to_fit))

        idx = np.concatenate([idx_00, idx_01, idx_11])

        nl = len(idx) // 3

        def tomin(cal):
            cal = cal[0]

            P = np.zeros((2 * nl, len(idx)))

            P[:nl, :nl] = -np.eye(nl)
            P[:nl, nl: 2 * nl] = cal * np.eye(nl)
            P[:nl, 2 * nl:] = 0.0

            P[nl:, :nl] = -np.eye(nl)
            P[nl:, nl:2 * nl] = 0.0
            P[nl:, 2 * nl:] = cal**2 * np.eye(nl)

            res = P @ ps_vec[idx]
            rescov = P @ cov[np.ix_(idx, idx)] @ P.T
            rescov = np.diag(rescov.diagonal())

            invcov = np.linalg.inv(rescov)

            return res.T @ invcov @ res

        res = minimize(
            tomin,
            x0=np.array([1.0]),
            method="L-BFGS-B",
        )
        cal = res.x[0]
        print(
            f"Calibration factor for {tr_to_fit} relative to {ref}: {cal:.4f}"
        )

        cals[tr_to_fit] = cal

    # Get full multipole range
    print("Reading full sacc file")
    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices(ell__lt=args.lmax_sacc)
    s.keep_indices(idx)

    ps_vec = s.mean
    cov = s.covariance.covmat
    idx_ref = s.indices("cl_ee", (ref, ref))
    for tr1, tr2 in s.get_tracer_combinations():
        idxEE = s.indices("cl_ee", (tr1, tr2))

        lb, _ = s.get_ell_cl("cl_ee", tr1, tr2)
        plt.figure(figsize=(8, 6))
        plt.axvline(lmin, color="gray", ls="--", lw=1.0)
        plt.axvline(lmax, color="gray", ls="--", lw=1.0)
        plt.errorbar(
            lb,
            ps_vec[idx_ref],
            yerr=np.sqrt(cov[np.ix_(idx_ref, idx_ref)].diagonal()),
            marker="o",
            ls="None",
            markerfacecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.2,
            color="k",
            label="Reference",
        )
        plt.errorbar(
            lb,
            ps_vec[idxEE],
            yerr=np.sqrt(cov[np.ix_(idxEE, idxEE)].diagonal()),
            marker="o",
            ls="None",
            markerfacecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.2,
            color="navy",
            label="Measured",
        )
        plt.errorbar(
            lb,
            ps_vec[idxEE] * cals.get(tr1, 1.0) * cals.get(tr2, 1.0),
            yerr=np.sqrt(cov[np.ix_(idxEE, idxEE)].diagonal())
            * cals.get(tr1, 1.0)
            * cals.get(tr2, 1.0),
            marker="o",
            ls="None",
            markerfacecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.2,
            color="DodgerBlue",
            label="Calibrated",
        )
        plt.ylim(0.0, 7.0)
        plt.xlabel(r"$\ell$", fontsize=17)
        plt.ylabel(r"$D_\ell^{EE}$", fontsize=17)
        plt.title(f"{tr1} x {tr2}", fontsize=13)
        plt.legend(frameon=False, fontsize=17)
        plt.savefig(
            f"{out_dir}/EE_spectrum_{tr1}_{tr2}.pdf", bbox_inches="tight"
        )

    # Save calibs
    with open(f"{out_dir}/calibrations_lmin{lmin}_lmax{lmax}.pkl", "wb") as f:
        p.dump(cals, f)

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
            s_out.add_ell_cl(
                **{
                    "data_type": dtype,
                    "tracer1": t1,
                    "tracer2": t2,
                    "ell": lb,
                    "x": ps_vec[idx] * c1 * c2,
                    "window": s.get_bandpower_windows(idx),
                }
            )
            tot_idx += list(idx)
            cal_vec.append(
                np.full_like(idx, c1 * c2)
            )
    cal_vec = np.concatenate(cal_vec)
    cov_cal = cov[np.ix_(tot_idx, tot_idx)] * np.outer(cal_vec, cal_vec)
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
        "--map-set-ref",
        type=str,
        help="Reference map set for the EE calibration fit",
    )
    parser.add_argument(
        "--map-sets-to-fit",
        type=str,
        help="Map sets to fit for the EE calibration (default: all map sets)",
    )
    args = parser.parse_args()

    main(args)
