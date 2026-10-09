import argparse
from pipeline.nulls import NullArchive
import os


def main(args):
    """
    """
    sacc_files = args.sacc_files
    null_archive = NullArchive(sacc_files)

    if args.inspect:
        null_archive.inspect()

    else:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        lmins = args.lmins
        lmaxs = args.lmaxs
        field_pairs = args.field_pairs

        for lmin, lmax in zip(lmins, lmaxs):
            null_archive.summary_stats(
                field_pairs=field_pairs,
                ellmin=lmin,
                ellmax=lmax,
                fname=f"{out_dir}/null_summary_lmin{lmin}_lmax{lmax}"
            )
            for fp in field_pairs:
                null_archive.summary_stats(
                    field_pairs=[fp],
                    ellmin=lmin,
                    ellmax=lmax,
                    fname=f"{out_dir}/null_summary_{fp}_lmin{lmin}_lmax{lmax}"
                )

        null_archive.to_file(
            f"{out_dir}/null_archive.fits"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sacc-files",
        nargs="+"
    )
    parser.add_argument(
        "--inspect",
        action="store_true"
    )
    parser.add_argument(
        "--out-dir",
        type=str
    )
    parser.add_argument(
        "--lmins",
        nargs="+",
        type=float
    )
    parser.add_argument(
        "--lmaxs",
        nargs="+",
        type=float
    )
    parser.add_argument(
        "--field-pairs",
        nargs="+",
        default=["EE", "EB", "BB"]
    )

    args = parser.parse_args()

    main(args)
