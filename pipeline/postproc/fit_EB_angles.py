from scipy.optimize import minimize
import postproc_utils as pputils
import numpy as np
import argparse
import sacc
import time
import json


def main(args):
    sacc_file = args.sacc_file
    lmin = args.lmin_fit
    lmax = args.lmax_fit

    s = sacc.Sacc.load_fits(sacc_file)
    idx = s.indices(ell__gt=lmin, ell__lt=lmax)
    s.keep_indices(idx)

    tracers = list(s.tracers.keys())

    map_sets_to_fit = args.map_sets_to_fit.split(",")
    n_params = len(map_sets_to_fit)

    def tomin(p, s):
        """
        Objective function to minimize.
        """
        tracers = list(s.tracers.keys())
        tracer_pairs = s.get_tracer_combinations()
        ps_vec = s.mean
        cov = s.covariance.covmat
        alpha_EB = {tracer: p[i] for i, tracer in enumerate(map_sets_to_fit)}
        for tr in tracers:
            if tr not in map_sets_to_fit:
                alpha_EB[tr] = 0.0
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
    alpha_EB = {tracer: params[i] for i, tracer in enumerate(map_sets_to_fit)}
    for tr in tracers:
        if tr not in map_sets_to_fit:
            alpha_EB[tr] = 0.0

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
            lb, _ = s.get_ell_cl(dtype, t1, t2)
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

    s_out.metadata = s.metadata
    s_out.metadata[f"EB_rot_{int(time.time())}"] = json.dumps({
        "alpha_EB": alpha_EB,
        "lmin_fit": lmin,
        "lmax_fit": lmax
    })

    fname = sacc_file.replace(
        ".fits",
        "_rotated.fits"
    )
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
    parser.add_argument(
        "--map-sets-to-fit",
        type=str,
        help="Map set to fit for the EB angles",
    )
    args = parser.parse_args()

    main(args)
