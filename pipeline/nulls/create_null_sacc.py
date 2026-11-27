from soopercool import BBmeta
import numpy as np
import argparse
import sacc


def main(args):
    """
    This script will compile cross-null splits spectra
    into a single `sacc` file for data and sign-flip
    simulations.
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory
    sacc_dir = f"{out_dir}/null_saccs"
    BBmeta.make_dir(sacc_dir)

    cl_sims_dir = f"{out_dir}/cells_sims/noise"
    cl_dir = f"{out_dir}/cells"

    map_set_pairs = meta.get_ps_names_list(type="cross", coadd=True)
    null_groups = {
        ms: meta.null_group_from_map_set(ms)
        for ms in meta.map_sets
    }
    unique_groups = list(set(null_groups.values()))
    required_nulls = {group: [] for group in unique_groups}
    for ms1, ms2 in map_set_pairs:
        if null_groups[ms1] == null_groups[ms2]:
            required_nulls[null_groups[ms1]].append((ms1, ms2))

    n_sims = meta.covariance["cov_num_sims"]
    sim_ids = np.arange(
        meta.covariance["cov_id_start"],
        meta.covariance["cov_id_start"] + n_sims
    )
    s = sacc.Sacc()

    # Add tracers
    for group, ms_list in required_nulls.items():
        for ms1, ms2 in ms_list:
            if ms1 not in s.tracers:
                s.add_tracer("Misc", ms1)
            if ms2 not in s.tracers:
                s.add_tracer("Misc", ms2)

    # Since we take power spectrum of the map difference
    # we only need those field pairs. Symmetric pairs
    # are strictly equal.
    field_pairs = [
        "TT",
        "TE", "TB",
        "EE", "EB", "BB"
    ]

    for group, ms_lists in required_nulls.items():
        meta.logger.info(f"Adding null group {group} to sacc file...")
        for ms1, ms2 in ms_lists:
            meta.logger.info(f"  * map diff {ms1} - {ms2}...")

            cl11_dict = np.load(f"{cl_dir}/decoupled_cross_pcls_{ms1}_x_{ms1}.npz") # noqa
            cl22_dict = np.load(f"{cl_dir}/decoupled_cross_pcls_{ms2}_x_{ms2}.npz") # noqa
            cl12_dict = np.load(f"{cl_dir}/decoupled_cross_pcls_{ms1}_x_{ms2}.npz") # noqa
            lb = cl11_dict["lb"]

            for field_pair in field_pairs:
                cl11 = cl11_dict[field_pair]
                cl22 = cl22_dict[field_pair]
                cl12 = cl12_dict[field_pair]
                cl21 = cl12_dict[field_pair[::-1]]
                cldiff = cl11 + cl22 - cl12 - cl21

                data_type = f"cl_{field_pair.lower().replace('t', '0')}"
                for ell, value in zip(lb, cldiff):
                    s.add_data_point(
                        data_type=data_type,
                        tracers=(ms1, ms2),
                        ell=ell,
                        value=value,
                        group=group,
                        freq=meta.freq_tag_from_map_set(ms1)
                    )
            for iii in sim_ids:
                cl11_dict = np.load(f"{cl_sims_dir}/decoupled_cross_pcls_{ms1}_x_{ms1}_{iii:04d}.npz") # noqa
                cl22_dict = np.load(f"{cl_sims_dir}/decoupled_cross_pcls_{ms2}_x_{ms2}_{iii:04d}.npz") # noqa
                cl12_dict = np.load(f"{cl_sims_dir}/decoupled_cross_pcls_{ms1}_x_{ms2}_{iii:04d}.npz") # noqa
                lb = cl11_dict["lb"]

                for field_pair in field_pairs:
                    cl11 = cl11_dict[field_pair]
                    cl22 = cl22_dict[field_pair]
                    cl12 = cl12_dict[field_pair]
                    cl21 = cl12_dict[field_pair[::-1]]
                    cldiff = cl11 + cl22 - cl12 - cl21

                    data_type = f"cl_{field_pair.lower().replace('t', '0')}"
                    for ell, value in zip(lb, cldiff):
                        s.add_data_point(
                            data_type=data_type,
                            tracers=(ms1, ms2),
                            ell=ell,
                            value=value,
                            group=group,
                            freq=meta.freq_tag_from_map_set(ms1),
                            sim=iii
                        )
        s.save_fits(f"{sacc_dir}/null_sacc.fits", overwrite=args.overwrite)


def cli():
    parser = argparse.ArgumentParser(
        description="Sacc compilation of power spectra and covariances."
    )

    parser.add_argument(
        "--globals",
        type=str,
        help="Path to the yaml file"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode."
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
