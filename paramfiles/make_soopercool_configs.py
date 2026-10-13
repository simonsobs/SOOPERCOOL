import argparse
import json
import os
import re
import yaml


def load_yaml(config):
    """
    Load a YAML file with a custom constructor
    for path objects, allowing to reuse the same
    path across multiple lines in the YAML file.
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_group_label(splits, with_variants=False):
    """
    Underscore-separated tokens shared by all split names in a group.
    e.g. ["det_left", "det_right"] -> "det".
    The `with_variants` argument, when True, includes the
    remaining tokens in the label (e.g. "det_left_right").
    """
    tokens = [name.split("_") for name in splits]
    common = [t for t in tokens[0] if all(t in other for other in tokens)]
    if with_variants:
        variants = [t for toks in tokens for t in toks if t not in common]
        common += variants
    return "_".join(common)


def fill_map_string(template, patch, freq, wafer, split, bundle_id, map_type):
    """
    Replace placeholders in the `map_string_format` key.
    """
    # `wafer` already carries its underscore prefix
    template = template.replace("_{wafer}", wafer)
    for key, value in (
        ("patch", patch),
        ("freq_channel", freq),
        ("split", split),
        ("bundle_id", bundle_id),
        ("map_type", map_type),
    ):
        template = template.replace("{" + key + "}", value)
    return template


def replace_block(text, header, new_block):
    """
    Replace the template block starting at the line `header` and made of
    all the following lines indented deeper than it, by `new_block`.
    """
    indent = len(header) - len(header.lstrip())
    pattern = (
        re.escape(header)
        + r"\n(?: *#.*\n)*(?: {%d}.*\n)*" % (indent + 1)
    )
    if not re.search(pattern, text):
        raise KeyError(header)

    return re.sub(pattern, lambda _: new_block, text, count=1)


def replace_section(text, key, entries):
    """
    Rewrite the `  key:` section as one `name: "value"` line per entry.
    """
    new_block = f"  {key}:\n"
    for name, value in entries.items():
        new_block += f"    {name}: {json.dumps(value)}\n"

    return replace_block(text, f"  {key}:", new_block)


def generate_config(args, config, run_name, run_splits, output_config):
    """
    Generate a configuration file based on CLI arguments (args),
    an input configuration (config), a label for the run (run_name),
    a list of splits (run_splits) and the output path to save the
    configuration (output_config).
    """
    bundling = config["bundling"]
    filtering = config["filtering"]
    signflip = config["signflip"]

    with open("_TEMPLATE_CONFIG.yaml", "r") as f:
        sc_cfg = f.read()

    # Telescope name, e.g. "satp1", found in the map string format
    # TODO: this is a bit hacky, I think the telescope name could
    # be somewhere in a configuration file ?
    match = re.search(r"satp\d+", bundling["map_string_format"], re.IGNORECASE)
    if match is None:
        raise ValueError(
            "Could not find a match for 'satp%d' in "
            "bundling['map_string_format']"
        )
    tel = match.group()

    # Format wafer names as a list of
    # strings with an underscore prefix
    # e.g. "_ws0", "_ws1", ...
    wafers = bundling["wafer"]
    if wafers is None:
        wafers = [""]
    elif isinstance(wafers, str):
        wafers = [f"_{wafers}"]
    elif isinstance(wafers, list):
        wafers = [f"_{waf}" for waf in wafers]
    else:
        raise ValueError("wafer entry must be None, str, list.")

    # Per-map-set entries of the sections that list every map set
    sections = {
        key: {} for key in (
            "noise_map_sims_dir", "noise_map_sims_template", "hits_files",
            "unfiltered_map_dir", "unfiltered_map_template",
            "filtered_map_dir", "filtered_map_template",
        )
    }
    sim_template = filtering["sim_string_format"].replace(
        "{sim_type}", "{pure_type}"
    )
    block_template = re.search(
        r'  "\{MAP_SET_NAME\}":\n(?:    .*\n)*', sc_cfg
    ).group()

    blocks = []
    for freq in config["freq_channel"]:
        for patch in config["patch"]:
            for wafer in wafers:
                for split in run_splits:
                    name = f"{tel}_{freq}_{patch}{wafer}"
                    if run_name != "science":
                        name += f"_{split}"

                    map_template = fill_map_string(
                        bundling["map_string_format"], patch, freq, wafer,
                        split, "{id_bundle}", "{map|hits}"
                    )
                    hits_file = fill_map_string(
                        bundling["map_string_format"], patch, freq, wafer,
                        split, "0", "hits"
                    )
                    filtering_dir = (
                        filtering["output_dir_filtering"]
                        .replace("{patch}", patch)
                        .replace("{freq_channel}", freq)
                    )

                    # Map set block
                    block = block_template
                    for tag, value in (
                        ('"{MAP_SET_NAME}"', name),
                        ("{MAP_DIR}", bundling["output_dir_bundling"]),
                        ("{MAP_TEMPLATE}", map_template),
                        ('"{N_BUNDLES}"', str(config["n_bundles"])),
                        ("{FREQ_TAG}", freq),
                        ("{EXP_TAG}", tel),
                        ("{FILTERING_TAG}", name),
                        ('"{KSPACE_TAG}"', "null"),
                        ("{HITS_TAG}", name),
                    ):
                        block = block.replace(tag, value)
                    blocks.append(block)

                    # Entries in the other sections
                    bundles_dir = bundling["output_dir_bundling"].rstrip("/")
                    sections["noise_map_sims_dir"][name] = (
                        signflip["output_dir_signflip"]
                    )
                    sections["noise_map_sims_template"][name] = (
                        "{id_sim:04d}/{map_set}_bundle{id_bundle}_noise.fits"
                    )
                    sections["hits_files"][name] = (
                        f"{bundles_dir}/{hits_file}"
                    )
                    # HACK: resolution of input maps is different than input
                    # map for filtering we scan higher resolution maps and
                    # then use low-res
                    sections["unfiltered_map_dir"][name] = \
                        filtering["sim_dir"].replace("1.0arcmin", "4.0arcmin")
                    sections["unfiltered_map_template"][name] = sim_template
                    sections["filtered_map_dir"][name] = filtering_dir
                    # HACK: filtered map name is hardcoded in the code.
                    sections["filtered_map_template"][name] = \
                        sim_template.replace(".fits", "_filtered.fits")

    sc_cfg = sc_cfg.replace(block_template, "\n".join(blocks))
    for key, entries in sections.items():
        sc_cfg = replace_section(sc_cfg, key, entries)

    sc_cfg = sc_cfg.replace(
        "{OUTPUT_DIR}",
        os.path.join(args.output_dir, run_name)
    )
    sc_cfg = sc_cfg.replace("{CAR_TEMPLATE}", config["car_map_template"])
    sc_cfg = sc_cfg.replace('"{COV_NUM_SIMS}"', str(signflip["n_sims"]))
    tf_sim_ids = filtering["sim_ids"].split(",")
    idstart, idstop = int(tf_sim_ids[0]), int(tf_sim_ids[-1])
    nsims = idstop - idstart + 1
    sc_cfg = sc_cfg.replace('"{TF_EST_NUM_SIMS}"', str(nsims))
    sc_cfg = sc_cfg.replace('"{TF_SIMS_ID_START}"', str(idstart))

    with open(output_config, "w") as f:
        f.write(sc_cfg)


def main(args):
    """
    """
    # Load bundling-filtering config
    config = load_yaml(args.config_file)

    output_root, output_ext = os.path.splitext(args.output_config)
    if not output_ext:
        output_ext = ".yaml"

    # One run per group of splits listed in the
    # inter_obs_splits and intra_obs_splits entries
    groups = [
        group for key in ("intra_obs_splits", "inter_obs_splits")
        for group in config.get(key, [])
    ]

    # Get label once
    labels = [get_group_label(group) for group in groups]

    # Filter for groups sharing the same label
    # (e.g. det_left/det_right and det_in/det_out)
    labels = [
        get_group_label(group, with_variants=labels.count(label) > 1)
        for group, label in zip(groups, labels)
    ]

    # Concantenate all configuration files to
    # generate as a [(label, splits)] list.
    split_groups = [("science", [config["bundling"]["coadd_splits_name"]])]
    split_groups += zip(labels, groups)

    for run_name, run_splits in split_groups:
        output_config = f"{output_root}_{run_name}{output_ext}"
        generate_config(args, config, run_name, run_splits, output_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the bundling YAML configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="SOOPERCOOL outputs directory"
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default="config.yaml",
        help="Path to write the assembled SOOPERCOOL config file"
    )
    args = parser.parse_args()
    main(args)
