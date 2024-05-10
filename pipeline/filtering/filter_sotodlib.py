import argparse
from soopercool import BBmeta
from soopercool import map_utils as mu
import sotodlib_utils as su
import re
import numpy as np
from pixell import enmap


def main(args):
    """
    """
    meta = BBmeta(args.globals)

    out_dir = meta.output_directory

    bundle_id = re.findall(r"bundle[0-9]{1}", args.atomics_list)[0]
    freq_tag = re.findall(r"f[0-9]{3}", args.atomics_list)[0]

    fsims_dir = f"{out_dir}/sotodlib_filtered/{freq_tag}_bundle{bundle_id}"
    BBmeta.make_dir(fsims_dir)

    # First read the provided atomic maps list
    with open(args.atomics_list, "r") as f:
        atomics = f.read().splitlines()

    id_min, id_max = args.sim_ids.split(",")
    id_min = int(id_min)
    id_max = int(id_max)

    # Pre-load atomic ids
    atomic_ids = []
    for atom in atomics:
        obs_id = re.findall(r"[0-9]{10}", atom)[0]
        wafer = re.findall(r"ws[0-9]{1}", atom)[0]
        atomic_ids.append((obs_id, wafer))

    # Pre-load the access manager for filtering
    aman = {}
    for obs_id, wafer in atomic_ids:
        db_obs_id = su.get_obs_id_from_map_id(args.context, obs_id)
        aman[(wafer, obs_id)] = su.get_aman(
            args.context, 
            db_obs_id,
            wafer, freq_tag,
            thinning_factor=1.0,
            seed=1234
        )

    # Loop over the simulations
    for sim_id in range(id_min, id_max + 1):
        map_fname = args.map_template.format(sim_id=sim_id)
        map_file = f"{args.map_dir}/{map_fname}"
        print(map_file)
        sim = mu.read_map(map_file, ncomp=3)
        _, wcs = sim.geometry

        template = su.get_CAR_template(3, 10)
        template_w = su.get_CAR_template(3, 10)

        for obs_id, wafer in atomic_ids:

            fsim_wmap, fsim_w = su.filter_sim(
                aman[wafer, obs_id],
                sim, wcs, return_nofilter=False
            )
            fsim_w = np.moveaxis(fsim_w.diagonal(), -1, 0)
            template = enmap.insert(template, fsim_wmap, op=np.ndarray.__iadd__)
            template_w = enmap.insert(template_w, fsim_w, op=np.ndarray.__iadd__)

        template_w[template_w == 0] = np.inf
        fsim = template / template_w

        out_fname = args.map_template.format(sim_id=sim_id).replace(".fits", "_filtered.fits")
        out_file = f"{fsims_dir}/{out_fname}"

        enmap.write_map(out_file, fsim, dtype=np.float32, overwrite=True)
            

b="""
    for i in Nsims:
        sim = ...
        for a in atomics:
            P=read_pointing
            t = map2tod(sim, P)
            tf = filter(t)
            m1 = tod2map(tf, P)
            m2 = tod2map(t, P)

        sim1 = bundling(m1)
        sim2 = bundling(m2)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--atomic-maps-dir", help="Path to the atomic maps root dir.", required=False)
    parser.add_argument("--atomics-list", help="List of atomic maps", required=False)
    parser.add_argument("--context", help="Context file", required=False)
    parser.add_argument("--map-dir", help="Map to filter", required=False)
    parser.add_argument("--map-template", help="Map template", required=False)
    parser.add_argument("--sim-ids", help="Sim id", required=False)

    args = parser.parse_args()
    main(args)
