"""Download the GRID pre-built bundle: Beauty / Sports / Toys.

These three are not rebuilt from raw reviews here. We reuse the pre-processed
TFRecord splits released by the GRID project (which re-use the P5 paper's
splits) so the item-id space stays identical to prior work. GRID ships all
three datasets in a single Google Drive archive:

    https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view

This module fetches that archive once (cached under `_raw/grid_bundle/`, shared
by all three datasets) and extracts the requested dataset's folder into its data
dir. Large Drive files serve an HTML "virus scan" confirmation page instead of
the bytes, so the download goes through `gdown`, which handles the confirm-token
handshake. Google Drive also enforces a daily quota on popular large files; when
it is exhausted the download fails and we raise a message pointing at the manual
route, which is the same bundle extracted by hand.

Because the splits are already in GRID TFRecord form, we do not re-shard them —
we place the files as-is and only count users/items so the caller can check the
result against the paper statistics.
"""
import os
import shutil
import zipfile

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dataset_configs import DATA_ROOT, resolve_split_dir

# One Google Drive file holds beauty + sports + toys.
BUNDLE_FILE_ID = "1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz"
BUNDLE_NAME = "grid_bundle.zip"
# Shared cache so requesting all three downloads the archive only once,
# regardless of the per-dataset `raw_dir` the caller passes.
BUNDLE_DIR = os.path.join(DATA_ROOT, "_raw", "grid_bundle")

_MANUAL_HINT = (
    "Could not download the GRID bundle automatically. Google Drive enforces a "
    "daily quota on large files, so this can happen even when the link is fine; "
    "try again later or fetch it by hand:\n"
    f"    https://drive.google.com/file/d/{BUNDLE_FILE_ID}/view\n"
    "Extract it under your data root so you get "
    "data/{beauty,sports,toys}/{training,evaluation,testing,items}/ "
    "(the release's train/validation/test names also work)."
)


def _download_bundle():
    """Fetch the shared bundle archive via gdown, unless already cached."""
    dest = os.path.join(BUNDLE_DIR, BUNDLE_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [cached] {dest}")
        return dest

    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "gdown is required to download the GRID bundle "
            "(pip install gdown, or install requirements.txt)."
        )

    os.makedirs(BUNDLE_DIR, exist_ok=True)
    print(f"  downloading GRID bundle from Google Drive (id {BUNDLE_FILE_ID})")
    print(f"          --> {dest}")
    tmp = dest + ".part"
    try:
        out = gdown.download(id=BUNDLE_FILE_ID, output=tmp, quiet=False)
    except Exception as exc:  # gdown raises assorted errors on quota/HTML pages
        raise SystemExit(f"{_MANUAL_HINT}\n\n(gdown error: {exc})")
    if not out or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise SystemExit(_MANUAL_HINT)
    os.replace(tmp, dest)
    return dest


def _extract_dataset(archive, dataset, out_dir):
    """Extract just `<dataset>/…` from the bundle into `out_dir`.

    The archive holds beauty/sports/toys either at the top level or under one
    wrapper folder, so we locate the `<dataset>` component in the member paths
    and copy everything beneath it, preserving the split subdirectory names.
    """
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        prefix = None
        for n in names:
            parts = n.split("/")
            if dataset in parts:
                prefix = "/".join(parts[: parts.index(dataset) + 1]) + "/"
                break
        if prefix is None:
            tops = sorted({n.split("/")[0] for n in names if n})
            raise SystemExit(
                f"{dataset!r} was not found inside {archive}. "
                f"Top-level entries: {tops}"
            )

        os.makedirs(out_dir, exist_ok=True)
        extracted = 0
        for n in names:
            if n.endswith("/") or not n.startswith(prefix):
                continue
            rel = n[len(prefix):]                    # e.g. "test/data_0.tfrecord.gz"
            target = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(n) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    return extracted


def _count_records(split_dir):
    """Number of TFRecord examples across a split directory's *.tfrecord.gz."""
    files = tf.io.gfile.glob(os.path.join(split_dir, "*.tfrecord.gz"))
    if not files:
        return 0
    return sum(1 for _ in tf.data.TFRecordDataset(files, compression_type="GZIP"))


def prepare(raw_dir, out_dir, dataset):
    """Download + extract one GRID dataset. Returns a stats dict.

    `raw_dir` is ignored in favour of the shared bundle cache — the archive is
    one file for all three datasets. Users = records in the testing split;
    items = records in the items split. Interactions/avg-len are not recomputed
    (the splits are prebuilt); use `verify_datasets.py` for a full check.
    """
    archive = _download_bundle()
    n_files = _extract_dataset(archive, dataset, out_dir)
    print(f"  extracted {n_files} files -> {out_dir}")

    items_dir = resolve_split_dir(out_dir, "items")
    test_dir = resolve_split_dir(out_dir, "testing")
    n_items = _count_records(items_dir)
    n_users = _count_records(test_dir)
    return {
        "users": n_users,
        "items": n_items,
        "interactions": 0,
        "avg_seq_len": 0.0,
    }
