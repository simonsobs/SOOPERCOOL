import argparse
import re
import yaml
from pprint import pprint
import warnings
from copy import copy


def load_yaml(config):
    """
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def compose_yaml(config):
    """
    """
    def path_constructor(loader, node):
        return "/".join(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor("!path", path_constructor)
    with open(config, "r") as f:
        return yaml.compose(f, Loader=yaml.SafeLoader)


def get_child_node(node, key):
    """Return the value-node for `key` inside a MappingNode."""
    for key_node, value_node in node.value:
        if key_node.value == key:
            return value_node
    raise KeyError(key)


def get_section_text(text, node, key):
    """Raw text span of `key: ...` inside the given MappingNode."""
    for key_node, value_node in node.value:
        if key_node.value == key:
            start = key_node.start_mark.index
            # include the line's leading indentation
            start = text.rfind("\n", 0, start) + 1
            end = value_node.end_mark.index
            return text[start:end].rstrip("\n")
    raise KeyError(key)


def get_section_span(text, node, key):
    """Return (start, end) character offsets of `key: ...` in `text`."""
    for key_node, value_node in node.value:
        if key_node.value == key:
            start = key_node.start_mark.index
            start = text.rfind("\n", 0, start) + 1
            end = value_node.end_mark.index
            return start, end
    raise KeyError(key)


def strip_trailing_comments(text):
    lines = text.splitlines(keepends=True)
    while lines and (
        lines[-1].strip() == "" or lines[-1].lstrip().startswith("#")
    ):
        lines.pop()
    return "".join(lines)


class ConfigText(str):
    """
    A string class for template configurations
    in txt format.
    """
    @classmethod
    def from_file(cls, file_name):
        with open(file_name, "r") as f:
            return cls(f.read())

    def to_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(self)

    def replace(self, *args, **kwargs):
        return ConfigText(super().replace(*args, **kwargs))


def main(args):
    """
    """
    # Load bundling-filtering config
    config = load_yaml(args.config_file)

    # Load sub-configurations
    bundling = config["bundling"]
    filtering = config["filtering"]
    signflip = config["signflip"]

    sc_cfg = ConfigText.from_file("_TEMPLATE_CONFIG.yaml")

    # Define output directory
    sc_cfg = sc_cfg.replace(
        "{OUTPUT_DIR}",
        args.output_dir
    )

    # Find the string "satp{%d}" in bundling["map_string_format"] in
    # a case insentive manner where %d can be 1/2/3....
    # and assign it to the variable tel.
    match = re.search(r"satp\d+", bundling["map_string_format"], re.IGNORECASE)
    tel = match.group() if match else None
    if tel is None:
        raise ValueError(
            "Could not find a match for 'satp%d' in "
            "bundling['map_string_format']"
        )
    freqs = config["freq_channel"]
    patches = config["patch"]
    wafers = [""]
    if args.perwafer:
        wafers = [f"_ws{i}" for i in range(7)]

    # Loop over map sets and build blocks
    root = compose_yaml("_TEMPLATE_CONFIG.yaml")
    map_sets_node = get_child_node(root, "map_sets")
    block_template = get_section_text(
        sc_cfg,
        map_sets_node,
        "{MAP_SET_NAME}"
    )
    block_template = strip_trailing_comments(block_template)
    print(block_template)

    blocks = []
    for freq in freqs:
        for patch in patches:
            for wafer in wafers:
                block = block_template
                # Replace block name
                block = block.replace(
                    '"{MAP_SET_NAME}"',
                    f"{tel}_{freq}_{patch}{wafer}"
                )
                # Replace map_dir 
                block = block.replace(
                    "{MAP_DIR}",
                    bundling["output_dir_bundling"]
                )
                # Replace map_file
                map_fname = bundling["map_string_format"]
                map_fname = map_fname.replace(
                    "{patch}",
                    patch
                )
                map_fname = map_fname.replace(
                    "{freq_channel}",
                    freq
                )
                map_fname = map_fname.replace(
                    "_{wafer}",
                    wafer
                )
                map_fname = map_fname.replace(
                    "{bundle_id}",
                    "{id_bundle}"
                )
                map_fname = map_fname.replace(
                    "{map_type}",
                    "{map|hits}"
                )
                block = block.replace(
                    "{MAP_TEMPLATE}",
                    map_fname
                )
                # Replace n_bundles
                block = block.replace(
                    '"{N_BUNDLES}"',
                    str(config["n_bundles"])
                )
                # Replace freq tag
                block = block.replace(
                    "{FREQ_TAG}",
                    freq
                )
                # Replace exp tag
                block = block.replace(
                    "{EXP_TAG}",
                    tel
                )
                # Replace filtering tag
                block = block.replace(
                    "{FILTERING_TAG}",
                    f"{tel}_{freq}_{patch}{wafer}"
                )
                # Replace kspace tag
                block = block.replace(
                    '"{KSPACE_TAG}"',
                    "null"
                )
                # Replace hits tag
                block = block.replace(
                    "{HITS_TAG}",
                    f"{tel}_{freq}_{patch}{wafer}"
                )

                blocks.append(block)

    new_map_sets_text = "map_sets:\n" + "\n".join(blocks)

    start, end = get_section_span(sc_cfg, root, "map_sets")
    sc_cfg = ConfigText(sc_cfg[:start] + new_map_sets_text + sc_cfg[end:])

    # Replace CAR template
    sc_cfg = sc_cfg.replace(
        "{CAR_TEMPLATE}",
        config["car_map_template"]
    )


    sc_cfg.to_file(args.output_config)
    root = yaml.compose(sc_cfg, Loader=yaml.SafeLoader)
    start, end = get_section_span(sc_cfg, root, "noise_map_sims_dir")
    sc_cfg = ConfigText(sc_cfg[start:end])
    print(sc_cfg)


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
    parser.add_argument(
        "--perwafer",
        action="store_true",
        help="If set, the config will be generated for each wafer"
    )
    args = parser.parse_args()
    main(args)