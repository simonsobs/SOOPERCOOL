import argparse
from soopercool import BBmeta


def main(args):
    """ """
    meta = BBmeta(args.globals)

    filtered_sims_dir = f"{meta.output_directory}/{args.out_dir}"
    BBmeta.make_dir(filtered_sims_dir)

    mask_file = meta.masks["analysis_mask"]

    id_min, id_max = args.sim_ids.split(",")
    id_min = int(id_min)
    id_max = int(id_max)

    filter_function = meta.get_filter_function(args.filter_tag)

    for sim_id in range(id_min, id_max + 1):
        map_name = f"{args.map_dir}/{args.map_template.format(sim_id=sim_id)}"

        filter_function(map_name, mask_file, filtered_sims_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals", help="Path to the global parameter file.")
    parser.add_argument("--map-dir", help="Path to the map file.")
    parser.add_argument("--map-template", help="Template for the map file.")
    parser.add_argument("--sim-ids", help="Bundle ID.")
    parser.add_argument("--out-dir", help="Name of the output directory")
    parser.add_argument("--filter-tag", help="Filtering tag.")
    args = parser.parse_args()
    main(args)
