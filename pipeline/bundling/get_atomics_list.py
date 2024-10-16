import argparse
import sqlite3
import numpy as np


def get_atomic_maps_list(map_dir, query_dict, map_type="wmap", db_fpath=None,
                         verbose=False):
    """
    Outputs list of atomic maps
    (e.g. "{map_dir}/17107/atomic_1710705196_ws0_f090_full_{map_type}.fits")
    based on map_dir (base directory for atomic maps) and query_dict.

    Parameters
    ----------
    map_dir: str
        Path to directory of atomic maps and atomics database.
    db_fpath: str
        Full path to file name of atomic maps database for map_dir. If not
        provided, default to "{map_dir}/atomic_maps.db". Default: None
    map_type: str
        Map type to output, e.g. 'wmap' fo weighted map, 'weights' for inverse
        variance weights map, 'hits' for hits map. Default: 'wmap'.
    query_dict: dict of str
        Dictionary with keys being the queryable parameters, e.g.
        freq_channel or elevation, and values being the associated (str) values
        or SQL query strings, e.g. " = 'f090'", ">= '30'"
    verbose: boolean
        Verbose output? Default: False

    Returns
    -------
    atomic_maps_prefix_list: list of str
        List with path prefixes to all atomic maps requested.
    """
    if db_fpath is None:
        db_fpath = f"{map_dir}/atomic_maps.db"
    if verbose:
        print(f" Querying {db_fpath}")
    assert map_type in ["wmap", "weights", "hits"]

    query_target = "prefix_path"
    table_name = "atomic"
    where_statement = " AND ".join([f"{k}{v}" for k, v in query_dict.items()])

    conn = sqlite3.connect(db_fpath)
    cursor = conn.cursor()

    # Print columns
    data = cursor.execute(f"SELECT * FROM {table_name};")
    cols = [col[0] for col in data.description]
    if verbose:
        print(f"Columns of {table_name}:\n  ", " ".join(cols))

    # FIXME: Give warning if query key is not present in the database.

    # Read data
    cursor.execute(f"select {query_target} from {table_name} where {where_statement};")  # noqa
    result = cursor.fetchall()
    if verbose:
        print(f" Found {len(result)} matching entries.")
    atomic_maps_list = []

    for r in result:
        atomic_maps_list.append(
            f"{map_dir}/{'/'.join(r[0].split('/')[-2:])}_{map_type}.fits"
        )
    conn.commit()
    conn.close()

    return atomic_maps_list


def main(args):
    """
    Get list of NERSC filepaths of atomic maps corresponding to an SQL query
    to the associated atomic maps database.
    """
    
    atomic_maps_list = {}
    atomic_maps_list["wmap"] = get_atomic_maps_list(
        args.map_dir, args.query_dict, map_type="wmap"
    )
    
    for typ in ["weights", "hits"]:
        atomic_maps_list[typ] = [a.replace("wmap", typ)
                                 for a in atomic_maps_list["wmap"]]
    np.savez(f"{args.outdir}/atomic_maps_list.npz", **atomic_maps_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_dir",
                        help="Atomic maps directory.")
    parser.add_argument("--query_dict",
                        help="Dictionary with SQL query of form "
                        "{key}{value}, i.e. key is \"freq_channel\" and"
                        " value is  \"= 'f090'\".")
    parser.add_argument("--outdir",
                        help="Output directory for atomic maps list.")

    args = parser.parse_args()
    main(args.preprocess_config, args.query_str)
