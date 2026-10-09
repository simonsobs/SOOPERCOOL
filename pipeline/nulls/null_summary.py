import argparse
from pipeline.nulls import NullArchive
import sacc
import os


def main(args):
    """
    """
    sacc_files = args.sacc_files
    saccs = [sacc.Sacc.load_fits(f) for f in sacc_files]
    null_archive = NullArchive(saccs)

    if args.inspect:
        unique_groups = list(set(null_archive.group))
        diffs = [
            (ms1, ms2)
            for ms1, ms2 in zip(
                null_archive.tracer1,
                null_archive.tracer2
            )
        ]
        unique_diffs = list(set(diffs))
        field_pairs = [
            dtype.split("_")[-1].upper().replace("0", "T")
            for dtype in null_archive.dtype
        ]
        unique_field_pairs = list(set(field_pairs))

        # Run some inspection of the null archive to
        # indicate which nulls, map sets are present.
        print("Inspecting null archive")
        print("-----------------------")
        print("  Provided files:")
        for f in sacc_files:
            print(f"    {f}")
        print("  Available null groups:")
        for group in unique_groups:
            print(f"    {group}")
        print("  Available map set differences:")
        for ms1, ms2 in unique_diffs:
            print(f"    {ms1} -- {ms2}")
        print("  Available field pairs:")
        for field_pair in unique_field_pairs:
            print(f"    {field_pair}")
        print(f"  Number of simulations: {null_archive.n_sims}")

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
