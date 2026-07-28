"""Build one or more datasets from their raw sources.

    python -m src.prepare delicious
    python -m src.prepare all
    python -m src.prepare goodreads-comics goodreads-children

Raw downloads are cached under `--raw_dir` (default `<data_root>/_raw/<name>`)
so re-running is cheap. Output goes to `<data_root>/<name>/`, where data_root
is `<repo>/data` unless `GRAPHREC_DATA_ROOT` says otherwise.

Beauty / Sports / Toys are not rebuilt from raw reviews — the `grid_bundle`
module downloads the GRID project's pre-built splits from Google Drive instead;
see the README.
"""
import argparse
import importlib
import os
import sys
import time
import traceback

# src/ holds flat modules (dataset_configs, tgh, ...) rather than a package, so
# put it on the path regardless of how this file was invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_configs import (
    DATA_ROOT, DATASET_CONFIGS, MANUAL_DOWNLOAD, PREBUILT, resolve_datasets,
)


def build(name, raw_dir=None, out_dir=None):
    """Prepare a single dataset. Returns its stats dict."""
    cfg = DATASET_CONFIGS[name]
    module_name = cfg["prepare"]
    if module_name is None:
        raise SystemExit(
            f"{name} is distributed pre-built by the GRID project and is not "
            f"prepared from raw data here. See the README 'Data preparation' section."
        )

    raw_dir = raw_dir or os.path.join(DATA_ROOT, "_raw", name)
    out_dir = out_dir or cfg["data_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    module = importlib.import_module(f"{__package__ or 'prepare'}.{module_name}")
    # flush so the header cannot be reordered behind a message written to stderr
    print(f"\n{'=' * 72}\n{cfg['display']}  ({name})\n{'=' * 72}", flush=True)
    print(f"  raw: {raw_dir}\n  out: {out_dir}", flush=True)

    t0 = time.time()
    # A few modules serve several datasets from one entry point and need to know
    # which: goodreads (comics/children) and grid_bundle (beauty/sports/toys).
    if module_name in ("goodreads", "grid_bundle"):
        stats = module.prepare(raw_dir, out_dir, dataset=name)
    else:
        stats = module.prepare(raw_dir, out_dir)
    print(f"  took {time.time() - t0:.1f}s")

    expected = (cfg["n_users"], cfg["n_items"])
    actual = (stats["users"], stats["items"])
    if actual != expected:
        print(
            f"  WARNING: got {actual[0]:,} users / {actual[1]:,} items, "
            f"paper reports {expected[0]:,} / {expected[1]:,}. "
            f"The upstream source may have changed."
        )
    else:
        print("  matches the paper statistics.")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="+",
                        help="dataset name(s), or 'all'")
    parser.add_argument("--raw_dir", default=None,
                        help="Cache dir for raw downloads (default: <data_root>/_raw/<name>).")
    parser.add_argument("--out_dir", default=None,
                        help="Override the output dir (single dataset only).")
    parser.add_argument("--skip_manual", action="store_true",
                        help="Skip datasets whose raw data needs an account "
                             "(H&M, Amazon-M2) instead of failing on them.")
    parser.add_argument("--keep_going", action="store_true",
                        help="Continue to the next dataset when one fails.")
    args = parser.parse_args()

    names = resolve_datasets(args.dataset)
    if args.out_dir and len(names) > 1:
        parser.error("--out_dir only makes sense with a single dataset")

    skipped, failed, built = [], [], []
    for name in names:
        if name in PREBUILT:
            skipped.append((name, "distributed pre-built by GRID"))
            continue
        if args.skip_manual and name in MANUAL_DOWNLOAD:
            skipped.append((name, "needs a manual download"))
            continue
        try:
            build(name, raw_dir=args.raw_dir, out_dir=args.out_dir)
            built.append(name)
        except Exception as exc:
            if not args.keep_going:
                raise
            traceback.print_exc()
            failed.append((name, str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__))

    if len(names) > 1:
        print(f"\n{'=' * 72}\nSummary\n{'=' * 72}")
        for n in built:
            print(f"  built   {n}")
        for n, why in skipped:
            print(f"  skipped {n}  ({why})")
        for n, why in failed:
            print(f"  FAILED  {n}  ({why})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
