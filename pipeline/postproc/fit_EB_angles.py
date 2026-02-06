import postproc_utils as pputils
import sacc
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle as p
import os


def main(args):
    sacc_file = args.sacc_file
    lmin = args.lmin_fit
    lmax = args.lmax_fit

    out_dir = "sacc_rotation"
    os.makedirs(out_dir, exist_ok=True)

    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices(ell__gt=lmin, ell__lt=lmax)
    s.keep_indices(idx)

    tracers = list(s.tracers.keys())
    n_params = len(tracers)

    def tomin(p, s):
        """
        Objective function to minimize.
        """
        tracers = list(s.tracers.keys())
        tracer_pairs = s.get_tracer_combinations()
        ps_vec = s.mean
        cov = s.covariance.covmat
        alpha_EB = {tracer: p[i] for i, tracer in enumerate(tracers)}
        Mrot = pputils.EBrot_mat(s, alpha_EB)
        unrot_ps_vec = np.transpose(Mrot) @ ps_vec

        inds = []
        for tr1, tr2 in tracer_pairs:
            idx = s.indices("cl_eb", (tr1, tr2))
            inds.append(idx)
        inds = np.concatenate(inds)

        # After rotation, EB power should be close to zero
        res = unrot_ps_vec[inds]
        invcov = np.linalg.inv(cov[np.ix_(inds, inds)])

        return res.T @ invcov @ res

    # Run optimization
    res = minimize(
        tomin,
        x0=np.zeros(n_params),
        args=(s,),
        method="L-BFGS-B",
    )
    params = res.x
    print("Minimization stopped !")
    alpha_EB = {tracer: params[i] for i, tracer in enumerate(tracers)}

    # Get full multipole range
    print("Reading full sacc file")
    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices(ell__lt=args.lmax_sacc)
    s.keep_indices(idx)

    print("Computing rotation matrix")
    Mrot = pputils.EBrot_mat(s, alpha_EB)
    ps_vec = s.mean
    cov = s.covariance.covmat
    unrot_ps_vec = np.transpose(Mrot) @ ps_vec
    unrot_covariance = np.transpose(Mrot) @ cov @ Mrot

    for tr1, tr2 in s.get_tracer_combinations():
        idxEB = s.indices("cl_eb", (tr1, tr2))

        lb, _ = s.get_ell_cl("cl_eb", tr1, tr2)
        plt.figure(figsize=(8, 6))
        plt.axhline(0, color="k", ls="--", lw=1.0)
        plt.axvline(lmin, color="gray", ls="--", lw=1.0)
        plt.axvline(lmax, color="gray", ls="--", lw=1.0)
        plt.errorbar(
            lb,
            ps_vec[idxEB],
            yerr=np.sqrt(cov[np.ix_(idxEB, idxEB)].diagonal()),
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
            unrot_ps_vec[idxEB],
            yerr=np.sqrt(unrot_covariance[np.ix_(idxEB, idxEB)].diagonal()),
            marker="o",
            ls="None",
            markerfacecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.2,
            capsize=2.5,
            capthick=1.2,
            color="DodgerBlue",
            label="EB-nulled",
        )
        plt.ylim(-0.8, 0.8)
        plt.xlabel(r"$\ell$", fontsize=17)
        plt.ylabel(r"$D_\ell^{EB}$", fontsize=17)
        plt.title(f"{tr1} x {tr2}", fontsize=13)
        plt.legend(frameon=False, fontsize=17)
        plt.savefig(
            f"{out_dir}/EB_spectrum_{tr1}_{tr2}.pdf", bbox_inches="tight"
        )

    # Save EB angles
    with open(f"{out_dir}/EB_angles.pkl", "wb") as f:
        p.dump(alpha_EB, f)

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
    for dtype in s.get_data_types():
        for t1, t2 in s.get_tracer_combinations():
            idx = s.indices(dtype, tracers=(t1, t2))
            s_out.add_ell_cl(
                **{
                    "data_type": dtype,
                    "tracer1": t1,
                    "tracer2": t2,
                    "ell": lb,
                    "x": unrot_ps_vec[idx],
                    "window": s.get_bandpower_windows(idx),
                }
            )
            tot_idx += list(idx)
    s_out.add_covariance(unrot_covariance[np.ix_(tot_idx, tot_idx)])
    fname = f"{out_dir}/cl_and_cov_sacc_rotated_lmin{lmin}_lmax{lmax}.fits"
    s_out.save_fits(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit EB angles")
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
    args = parser.parse_args()

    main(args)
