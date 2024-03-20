import argparse
import subprocess
from pathlib import Path
import healpy as hp
from soopercool import BBmeta


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
    filter_map = meta.get_filter_function()

    # Read the mask
    mask = meta.read_mask("binary")

    meta.timer.start(f"Filter {meta.tf_est_num_sims} sims for TF estimation.")
    if args.transfer:
        for cl_type in ["cosmo", "tf_est", "tf_val"]:
            cases_list = ["pureE", "pureB"] if cl_type == "tf_est" else [None]
            for id_sim in range(meta.tf_est_num_sims):
                for case in cases_list:
                    map_file = meta.get_map_filename_transfer2(id_sim,
                                                               cl_type,
                                                               pure_type=case)
                    map = hp.read_map(map_file, field=[0, 1, 2])
                    if meta.filtering_type == "toast":
                        sbatch_job_name = 'sbatch_tf__{}_{}'.format(
                            cl_type, str(Path(map_file).name))
                        kwargs = {"instrument": meta.toast['tf_instrument'],
                                  "band": meta.toast['tf_band'],
                                  "sbatch_job_name": sbatch_job_name}
                    else:
                        kwargs = {}
                    filter_map(map, map_file, mask, kwargs)
        if meta.filtering_type == "toast":
            if meta.toast['slurm']:
                meta.timer.stop(
                    f"Filter {meta.tf_est_num_sims} sims for TF estimation.",
                    verbose=True)
                # Running with SLURM job scheduller
                cmd = (
                    "find '{}' -type f "
                    "-name 'sbatch_tf__*.sh' "
                    " -exec sbatch {{}} \\;").format(
                        Path(meta.toast['scripts_dir']).resolve())
                if meta.toast['slurm_autosubmit']:
                    subprocess.run(cmd, shell=True, check=True)
                    print('Submitted {} sims to SLURM for TF estimation.'
                          .format(meta.tf_est_num_sims))
                else:
                    meta.print_banner(
                        msg='To submit these scripts to SLURM:\n    {}'
                        .format(cmd))
            else:
                cmd = (
                    "find '{}' -type f "
                    "-name 'sbatch_tf__*.sh' "
                    "-exec {{}} \\;").format(
                        Path(meta.toast['scripts_dir']).resolve())
                subprocess.run(cmd, shell=True, check=True)
                meta.timer.stop(
                    f"Filter {meta.tf_est_num_sims} sims for TF estimation.",
                    verbose=True)
        else:
            meta.timer.stop(
                f"Filter {meta.tf_est_num_sims} sims for TF estimation.",
                verbose=True)

    if args.sims or args.data:
        Nsims = meta.num_sims if args.sims else 1
        meta.timer.start(f"Filter {Nsims} sims.")
        for map_name in meta.maps_list:
            map_set, id_split = map_name.split("__")
            for id_sim in range(Nsims):
                map_file = meta.get_map_filename(
                    map_set,
                    id_split,
                    id_sim=id_sim if Nsims > 1 else None
                )
                map = hp.read_map(map_file, field=[0, 1, 2])
                if meta.filtering_type == "toast":
                    if args.sims:
                        sbatch_job_name = 'sbatch_sims__{:04d}_{}'\
                            .format(id_sim, str(Path(map_file).name))
                    elif args.data:
                        sbatch_job_name = 'sbatch_data__{}'\
                            .format(str(Path(map_file).name))
                    kwargs = {"instrument": meta.toast['tf_instrument'],
                              "band": meta.toast['tf_band'],
                              "sbatch_job_name": sbatch_job_name}
                else:
                    kwargs = {}
                filter_map(map, map_file, mask, kwargs)
        if meta.filtering_type == "toast":
            _type = 'sims' if args.sims else 'data'
            if meta.toast['slurm']:
                meta.timer.stop(f"Filter {Nsims} sims.", verbose=True)
                # Running with SLURM job scheduller
                cmd = (
                    "find '{}' -type f "
                    "-name 'sbatch_{}__*.sh' "
                    "-exec sbatch {{}} \\;").format(
                        Path(meta.toast['scripts_dir']).resolve(),
                        _type)
                if meta.toast['slurm_autosubmit']:
                    subprocess.run(cmd, shell=True, check=True)
                    print('Submitted {} sims to SLURM for TF estimation.'
                          .format(meta.tf_est_num_sims))
                else:
                    meta.print_banner(
                        msg='To submit these scripts to SLURM:\n    {}'
                        .format(cmd))
            else:
                cmd = (
                    "find '{}' -type f "
                    "-name 'sbatch_{}__*.sh' "
                    "-exec {{}} \\;").format(
                        Path(meta.toast['scripts_dir']).resolve(),
                        _type)
                subprocess.run(cmd, shell=True, check=True)
                meta.timer.stop(f"Filter {Nsims} sims.", verbose=True)
        else:
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
