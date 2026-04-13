from soopercool import fft_utils as sfft
from soopercool import mpi_utils as mpi
from soopercool import map_utils as mu
from soopercool import BBmeta
import argparse
import os


def main(args):
    """ """

    rank, size, comm = mpi.init(True)

    meta = BBmeta(args.globals)

    out_dir = f"{meta.output_directory}/kspace_filtered_sims"
    meta.make_dir(out_dir)

    tf_settings = meta.transfer_settings
    id_start, n_sims = (
        tf_settings["sim_id_start"],
        tf_settings["tf_est_num_sims"],
    )
    pure_types = [f"pure{f}" for f in "TEB"]

    if rank == 0:
        files_list = []
        for map_set in meta.map_sets:
            print(f"Processing map set: {map_set}")

            kspace_tag = meta.kspace_tag_from_map_set(map_set)
            ftag = meta.filtering_tag_from_map_set(map_set)
            if kspace_tag is not None:
                print(f"  Applying k-space filter: {kspace_tag}")
                kspace_pars = tf_settings["kspace_pars"][kspace_tag]
                print(f"  k-space filter parameters: {kspace_pars}")

            else:
                print(
                    f"  No k-space filter to be applied to map set {map_set}. Skipping." # noqa
                )

            map_dir = tf_settings["filtered_map_dir"][ftag]
            for id_sim in range(id_start, id_start + n_sims):
                for pure_type in pure_types:
                    fname = tf_settings["filtered_map_template"][ftag].format(
                        pure_type=pure_type, id_sim=id_sim
                    )
                    path = f"{map_dir}/{fname}"
                    files_list.append((path, kspace_pars, kspace_tag))

    # Every rank must have the same list order
    mpi_shared_list = comm.bcast(files_list, root=0)

    task_ids = mpi.distribute_tasks(size, rank, len(files_list))
    local_files_list = [mpi_shared_list[i] for i in task_ids]

    for map_fname, kspace_pars, kspace_tag in local_files_list:

        m = mu.read_map(
            map_fname,
            pix_type=meta.pix_type,
            car_template=meta.car_template,
            fields_hp=[0, 1, 2],
        )

        # TODO: need to add a step before to mask noisy edges of the map
        # with bright pixels which makes the filtering more stable
        # Maybe using the binary + galactic mask is enough for this!
        m_filtered = sfft.kspace_filter(m, pix_type="car", **kspace_pars)
        fname_out = f"{out_dir}/{os.path.split(map_fname)[-1]}"
        fname_out = fname_out.replace(".fits", f"_kspace_{kspace_tag}.fits")
        mu.write_map(
            fname_out,
            m_filtered,
            pix_type=meta.pix_type,
            car_template=meta.car_template,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--globals", help="Path to the soopercool parameter file"
    )
    args = parser.parse_args()

    main(args)
