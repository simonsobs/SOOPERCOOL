import argparse
import subprocess
from pathlib import Path
import healpy as hp
from soopercool import BBmeta
import soopercool.mpi_utils as mpi_utils


def filter(args):
    """
    Filtering main routine. Calls the appropriate
    filterer depending on the type of filterer specified
    in the yaml file.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    """
    meta = BBmeta(args.globals)

    # Read the mask
    mask = meta.read_mask("binary")

    if args.transfer:

        meta.timer.start(f"Filter {meta.tf_est_num_sims} sims for TF estimation.")

        filtering_tags = meta.get_filtering_tags()
        filter_funcs = {ftag: meta.get_filter_function(ftag) for ftag in filtering_tags}
        filtering_type_list = [
            meta.tags_settings[ftag]["filtering_type"] for ftag in filtering_tags
        ]

        for cl_type in ["cosmo", "tf_est", "tf_val"]:
            cases_list = (
                ["pureE", "pureB", "pureT"] if cl_type == "tf_est" else [None]
            )  # noqa

            # Initialize MPI for non-TOAST filters
            use_mpi4py = "toast" not in filtering_type_list
            mpi_utils.init(use_mpi4py)

            for id_sim in mpi_utils.taskrange(meta.tf_est_num_sims - 1):
                for case in cases_list:
                    for ftag in filtering_tags:
                        map_file = meta.get_map_filename_transfer(
                            id_sim, cl_type, pure_type=case, filter_tag=ftag
                        )
                        map = hp.read_map(map_file, field=[0, 1, 2])
                        filter_map = filter_funcs[ftag]
                        if (
                            meta.tags_settings[ftag]["filtering_type"] == "toast"
                        ):  # noqa
                            sbatch_job_name = "sbatch_tf__{}_{}".format(
                                cl_type, str(Path(map_file).name)
                            )
                            filter_map(
                                map,
                                map_file,
                                mask,
                                extra_kwargs={"sbatch_job_name": sbatch_job_name},
                            )
                        else:
                            filter_map(map, map_file, mask)

        if "toast" in filtering_type_list:
            if meta.slurm:
                # Running with SLURM job scheduller
                cmd = "find '{}' -type f -name 'sbatch_tf__*.sh' -exec sbatch {{}} \\;".format(  # noqa
                    Path(meta.scripts_dir).resolve()
                )
                if meta.slurm_autosubmit:
                    subprocess.run(cmd, shell=True, check=True)
                    print(
                        "Submitted {} sims to SLURM for TF estimation.".format(
                            meta.tf_est_num_sims
                        )
                    )
                else:
                    meta.print_banner(
                        msg="To submit these scripts to SLURM:\n    {}".format(cmd)
                    )
            else:
                cmd = "find '{}' -type f -name 'sbatch_tf__*.sh' -exec {{}} \\;".format(  # noqa
                    Path(meta.scripts_dir).resolve()
                )
                subprocess.run(cmd, shell=True, check=True)

        meta.timer.stop(
            f"Filter {meta.tf_est_num_sims} sims for TF estimation.", verbose=True
        )

    if args.sims or args.data:
        Nsims = meta.num_sims if args.sims else 1
        meta.timer.start(f"Filter {Nsims} sims.")
        for map_name in meta.maps_list:
            map_set, id_split = map_name.split("__")

            ftag = meta.filtering_tag_from_map_set(map_set)
            filter_map = meta.get_filter_function(ftag)

            # Initialize MPI for non-TOAST filters
            use_mpi4py = "toast" not in ftag and args.sims
            mpi_utils.init(use_mpi4py)

            for id_sim in mpi_utils.taskrange(Nsims - 1):
                map_file = meta.get_map_filename(
                    map_set, id_split, id_sim=id_sim if Nsims > 1 else None
                )
                map = hp.read_map(map_file, field=[0, 1, 2])

                if meta.tags_settings[ftag]["filtering_type"] == "toast":
                    if args.sims:
                        sbatch_job_name = "sbatch_sims__{:04d}_{}".format(
                            id_sim, str(Path(map_file).name)
                        )
                    elif args.data:
                        sbatch_job_name = "sbatch_data__{}".format(
                            str(Path(map_file).name)
                        )
                    filter_map(
                        map,
                        map_file,
                        mask,
                        extra_kwargs={"sbatch_job_name": sbatch_job_name},
                    )
                else:
                    filter_map(map, map_file, mask)

        filtering_tags = meta.get_filtering_tags()
        filtering_type_list = [
            meta.tags_settings[ftag]["filtering_type"] for ftag in filtering_tags
        ]
        if "toast" in filtering_type_list:
            _type = "sims" if args.sims else "data"
            if meta.slurm:
                # Running with SLURM job scheduller
                cmd = (
                    "find '{}' -type f "
                    "-name 'sbatch_{}__*.sh' "
                    "-exec sbatch {{}} \\;"
                ).format(Path(meta.scripts_dir).resolve(), _type)
                if meta.slurm_autosubmit:
                    subprocess.run(cmd, shell=True, check=True)
                    print(
                        "Submitted {} sims to SLURM for TF estimation.".format(
                            meta.tf_est_num_sims
                        )
                    )
                else:
                    meta.print_banner(
                        msg="To submit these scripts to SLURM:\n    {}".format(cmd)
                    )
            else:
                cmd = (
                    "find '{}' -type f " "-name 'sbatch_{}__*.sh' " "-exec {{}} \\;"
                ).format(Path(meta.scripts_dir).resolve(), _type)
                subprocess.run(cmd, shell=True, check=True)

        meta.timer.stop(f"Filter {Nsims} sims.", verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filterer stage")
    parser.add_argument("--globals", type=str, help="Path to the yaml file")
    parser.add_argument("--plots", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transfer", action="store_true")
    mode.add_argument("--sims", action="store_true")
    mode.add_argument("--data", action="store_true")
    args = parser.parse_args()
    filter(args)
