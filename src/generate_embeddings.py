"""Generate item embeddings with google/flan-t5-xl over the items/*.tfrecord.gz
shards from the GRID release.

For each item record the script reads the `text` bytes field
(e.g. `b"Title: ...; Brand: ...; Categories: ..."`), tokenises it (max_length=128,
padded), runs the T5 encoder, and mean-pools the last hidden state with the
attention mask. Output:

    data/<dataset>/t5xl.pt       # tensor [n_items, 2048] float32
    data/<dataset>/t5xl.pkl      # list[{"item_id": int, "embedding": list[float]}]

This matches GRID's `sem_embeds_inference_flat` recipe (T5EncoderModel +
MeanAggregation, max_length=128) without the Hydra/Lightning dependency, so
the release stays self-contained.

Usage:
    python src/generate_embeddings.py --dataset beauty
    python src/generate_embeddings.py --dataset beauty --batch_size 16 --device cuda
"""
import argparse
import glob
import os
import pickle

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer, T5EncoderModel

from dataset_configs import DATASET_CONFIGS

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODEL_NAME = "google/flan-t5-xl"
MAX_LENGTH = 128


def load_items(data_dir):
    """Yield (item_id, text) pairs from items/*.tfrecord.gz."""
    paths = sorted(glob.glob(os.path.join(data_dir, "items", "*.tfrecord.gz")))
    if not paths:
        raise FileNotFoundError(
            f"No items/*.tfrecord.gz under {data_dir}. Place the GRID-format "
            f"item shards there first."
        )
    for fp in paths:
        for raw in tf.data.TFRecordDataset([fp], compression_type="GZIP"):
            ex = tf.train.Example(); ex.ParseFromString(raw.numpy())
            feat = ex.features.feature
            iid = int(feat["id"].int64_list.value[0])
            text = feat["text"].bytes_list.value[0].decode("utf-8", errors="replace")
            yield iid, text


def encode(model, tokenizer, texts, device):
    enc = tokenizer(
        texts, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, add_special_tokens=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    h = out.last_hidden_state                         # [B, L, D]
    mask = enc.attention_mask.unsqueeze(-1).float()   # [B, L, 1]
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty",
                        choices=list(DATASET_CONFIGS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="float32",
                        help="Encoder dtype. The release ships float32 embeddings; "
                             "bfloat16/float16 cuts time in half on most GPUs with "
                             "negligible downstream impact.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or DATASET_CONFIGS[args.dataset]["data_dir"]
    pt_path = os.path.join(data_dir, "t5xl.pt")
    pkl_path = os.path.join(data_dir, "t5xl.pkl")
    if not args.overwrite and os.path.exists(pkl_path) and os.path.exists(pt_path):
        print(f"[skip] {pt_path} and {pkl_path} already exist (pass --overwrite).")
        return

    print(f"Loading items from {data_dir}/items/ ...")
    items = list(load_items(data_dir))
    items.sort(key=lambda x: x[0])
    n = len(items)
    print(f"  {n:,} items.")
    if [iid for iid, _ in items] != list(range(n)):
        max_iid = items[-1][0]
        print(f"  warning: item ids are not 0..{n-1} (max={max_iid}); "
              f"will pack into a [{max_iid+1}, D] tensor.")
        n_out = max_iid + 1
    else:
        n_out = n

    print(f"Loading {MODEL_NAME} (this downloads ~10 GB on first use) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=dtype_map[args.dtype])
    model.eval().to(args.device)
    emb_dim = model.config.d_model
    print(f"  encoder dim = {emb_dim}, dtype = {args.dtype}")

    emb_matrix = np.zeros((n_out, emb_dim), dtype=np.float32)
    seen = np.zeros(n_out, dtype=bool)
    n_batches = (n + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        chunk = items[bi * args.batch_size : (bi + 1) * args.batch_size]
        ids = [iid for iid, _ in chunk]
        texts = [t for _, t in chunk]
        out = encode(model, tokenizer, texts, args.device)
        for iid, vec in zip(ids, out):
            emb_matrix[iid] = vec
            seen[iid] = True
        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
            print(f"  batch {bi+1}/{n_batches}  ({(bi+1)*args.batch_size}/{n} items)",
                  flush=True)
    if not seen.all():
        missing = int((~seen).sum())
        print(f"  warning: {missing} item ids in [0, {n_out-1}] never appeared "
              f"in items/ — their rows are zero.")

    print(f"Saving {pt_path} ...")
    torch.save(torch.from_numpy(emb_matrix), pt_path)

    print(f"Saving {pkl_path} ...")
    records = [{"item_id": int(i), "embedding": emb_matrix[i].tolist()}
               for i in range(n_out) if seen[i]]
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Done. {len(records):,} embeddings (dim={emb_dim}).")


if __name__ == "__main__":
    main()
