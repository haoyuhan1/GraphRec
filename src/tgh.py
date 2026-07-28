"""TGH-1 (single-source) and TGH-2 (multi-source) ring-topk recommenders.

Given a user's interaction history, both methods score candidate items by:
    semantic_anchor . item_embedding   +   alpha * log1p(edge_count)/row_max
inside hop rings around the user's last item(s) in the item-transition graph.

  TGH-1: rings (1, 2, 3) around the user's last item, hop budgets `--hop_k`.
  TGH-2: rings (1, 2) around the user's last item AND rings (1, 2) around the
         user's second-to-last item, with budgets `--src2_hop_k` / `--src3_hop_k`.
         Items already picked by source 2 are excluded from source 3.

Defaults reproduce the paper:
    # TGH-1
    python tgh.py --dataset beauty --hop_k 7 2 1 --edge_weight_alpha 0.5
    # TGH-2
    python tgh.py --dataset beauty --src2_hop_k 5 1 --src3_hop_k 3 1 \\
                  --edge_weight_alpha 0.5
"""

import argparse
import collections
import glob
import multiprocessing as mp
import os
import pickle
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import tensorflow as tf

from dataset_configs import DATASET_CONFIGS, resolve_split_dir

tf.config.set_visible_devices([], "GPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_sequences(data_dir):
    gt, sec_last, history = {}, {}, {}
    test_dir = resolve_split_dir(data_dir, "testing")
    for fpath in sorted(glob.glob(os.path.join(test_dir, "*.tfrecord.gz"))):
        ds = tf.data.TFRecordDataset([fpath], compression_type="GZIP")
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            feat = ex.features.feature
            uid = feat["user_id"].int64_list.value[0]
            seq = list(feat["sequence_data"].int64_list.value)
            gt[uid] = seq[-1]
            sec_last[uid] = seq[-2]
            history[uid] = seq[:-1]
    print(f"Loaded {len(gt):,} test sequences.")
    return gt, sec_last, history


def load_item_embeddings(emb_path):
    print(f"Loading item embeddings from {emb_path} ...")
    with open(emb_path, "rb") as f:
        records = pickle.load(f)
    item_embs = {r["item_id"]: r["embedding"] for r in records}
    max_id = max(item_embs.keys())
    emb_dim = len(next(iter(item_embs.values())))
    emb_matrix = np.zeros((max_id + 1, emb_dim), dtype=np.float32)
    for iid, emb in item_embs.items():
        emb_matrix[iid] = emb
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms
    print(f"Loaded {len(item_embs):,} item embeddings, dim={emb_dim}.")
    return emb_matrix


def load_graph(data_dir):
    pkl_path = os.path.join(data_dir, "graph", "transition_graph.pkl")
    if os.path.exists(pkl_path):
        print(f"Loading graph from {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    print("Graph not found — building from training data ...")
    edge_counts = collections.Counter()
    train_dir = resolve_split_dir(data_dir, "training")
    for fpath in sorted(glob.glob(os.path.join(train_dir, "*.tfrecord.gz"))):
        ds = tf.data.TFRecordDataset([fpath], compression_type="GZIP")
        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.features.feature["sequence_data"].int64_list.value)
            for a, b in zip(seq[:-1], seq[1:]):
                edge_counts[(a, b)] += 1
    os.makedirs(os.path.join(data_dir, "graph"), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(edge_counts), f)
    return dict(edge_counts)


# ── Sparse k-hop ring construction ────────────────────────────────────────────

def build_sparse_adj(edge_counts, n_items):
    rows, cols = [], []
    for (a, b) in edge_counts:
        rows.append(a); cols.append(b)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
    A.sum_duplicates()
    A.data = np.ones_like(A.data, dtype=np.float32)
    return A.tocsr()


def compute_hop_rings_csr(A_bool, max_depth, max_explicit_depth=None):
    """rings_csr[h][src, item] = True iff item is at exactly (h+1) hops from src.

    `max_explicit_depth` caps how many rings are materialised explicitly. Deeper
    rings can be substituted at scoring time with NOT(visited ∪ history).
    """
    if max_explicit_depth is None:
        max_explicit_depth = max_depth
    n_explicit = min(max_depth, max_explicit_depth)

    n = A_bool.shape[0]
    A = A_bool.astype(np.float32).copy()
    rings = []
    visited = sp.eye(n, format="csr", dtype=np.float32) + A
    visited.data = np.minimum(visited.data, 1.0)
    ring_prev = A.copy()
    rings.append(ring_prev.astype(np.bool_).tocsr())
    print(f"    ring 1: nnz={ring_prev.nnz:,}", flush=True)
    for h in range(2, n_explicit + 1):
        t_h = time.time()
        reach = ring_prev @ A
        reach.data = np.ones_like(reach.data, dtype=np.float32)
        reach = reach.tocsr(); reach.eliminate_zeros()
        intersect = reach.multiply(visited)
        ring = (reach - intersect).tocsr(); ring.eliminate_zeros()
        rings.append(ring.astype(np.bool_).tocsr())
        visited = (visited + ring).tocsr()
        visited.data = np.minimum(visited.data, 1.0)
        ring_prev = ring
        print(f"    ring {h}: nnz={ring.nnz:,}  ({time.time()-t_h:.1f}s)", flush=True)
    return rings, visited.astype(np.bool_).tocsr()


# ── Anchor builders ────────────────────────────────────────────────────────────

def build_user_anchors_mean(emb_t, history, n_users, last_n):
    device = emb_t.device
    hist_arr = np.full((n_users, last_n), -1, dtype=np.int64)
    for u in range(n_users):
        h = history[u][-last_n:]
        if h:
            hist_arr[u, -len(h):] = h
    ids = torch.from_numpy(hist_arr).to(device).clamp(min=0)
    valid = torch.from_numpy(hist_arr >= 0).to(device).float()
    embs = emb_t[ids]
    summed = (embs * valid.unsqueeze(-1)).sum(dim=1)
    counts = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    raw = summed / counts
    return raw / raw.norm(dim=1, keepdim=True).clamp(min=1e-8)


def build_user_anchors_single(emb_t, ids_np):
    device = emb_t.device
    ids = torch.from_numpy(ids_np).to(device)
    raw = emb_t[ids]
    return raw / raw.norm(dim=1, keepdim=True).clamp(min=1e-8)


# ── Edge-weight CSR ────────────────────────────────────────────────────────────

def _build_edge_weight_csr(edge_counts, n_items):
    n_edges = len(edge_counts)
    rows = np.empty(n_edges, dtype=np.int64)
    cols = np.empty(n_edges, dtype=np.int64)
    vals = np.empty(n_edges, dtype=np.float32)
    for i, ((a, b), c) in enumerate(edge_counts.items()):
        rows[i] = a; cols[i] = b; vals[i] = c
    EW = sp.csr_matrix(
        (np.log1p(vals), (rows, cols)), shape=(n_items, n_items),
    )
    row_max = np.asarray(EW.max(axis=1).todense()).flatten()
    row_max = np.where(row_max == 0, 1.0, row_max)
    return (sp.diags(1.0 / row_max) @ EW).tocsr()


# ── GPU per-ring topk ──────────────────────────────────────────────────────────
NEG_INF = -1e9


def gpu_per_ring_topk(
    anchor_t, emb_t, src_arr, depth, total_k,
    rings_csr, visited_csr=None, history=None,
    edge_w_alpha=0.0, edge_w_csr=None,
    already_arr=None, chunk_size=8000,
    return_pad=False,
):
    """For each user u and each ring h, return top-`total_k` items in ring h
    sorted by score desc. Output: list of `depth` int64 arrays [n_users, total_k];
    invalid entries = -1.

    Ring masks come from rings_csr[h] when h < len(rings_csr); otherwise the
    catch-all mask NOT(visited_csr ∪ history) is used. Edge-weight bonus is
    added to ring 0 only. `already_arr` masks items already picked by source 2.
    `return_pad=True` also returns an anchor-similarity ranking over
    NOT(visited ∪ history) for filling deficits semantically.
    """
    n_explicit = len(rings_csr)
    need_catchall = depth > n_explicit or return_pad
    if need_catchall:
        assert visited_csr is not None
    device = anchor_t.device
    n_users = anchor_t.shape[0]
    n_items = emb_t.shape[0]
    use_ew = edge_w_csr is not None and edge_w_alpha > 0
    k_take = min(total_k, n_items)
    out = [np.full((n_users, total_k), -1, dtype=np.int64) for _ in range(depth)]
    pad_out = (np.full((n_users, total_k), -1, dtype=np.int64)
               if return_pad else None)

    for cstart in range(0, n_users, chunk_size):
        cend = min(cstart + chunk_size, n_users)
        cs = cend - cstart
        chunk_src = src_arr[cstart:cend]

        anc = anchor_t[cstart:cend]
        scores = anc @ emb_t.T

        if use_ew:
            ew_np = edge_w_csr[chunk_src].toarray()
            bonus = torch.from_numpy(ew_np).to(device) * edge_w_alpha

        if already_arr is not None:
            chunk_already = already_arr[cstart:cend]
            already_mask = torch.zeros((cs, n_items), dtype=torch.bool, device=device)
            row_idx, col_idx = np.where(chunk_already >= 0)
            if row_idx.size:
                cols = chunk_already[row_idx, col_idx]
                already_mask[torch.from_numpy(row_idx).to(device),
                             torch.from_numpy(cols).to(device)] = True
        else:
            already_mask = None

        catchall_t = None
        if need_catchall:
            visited_np = visited_csr[chunk_src].toarray().astype(np.bool_)
            if history is not None:
                for u_local in range(cs):
                    h = history[cstart + u_local]
                    if h:
                        visited_np[u_local, h] = True
            catchall_t = torch.from_numpy(~visited_np).to(device)

        for h_idx in range(depth):
            if h_idx < n_explicit:
                ring_np = rings_csr[h_idx][chunk_src].toarray().astype(np.bool_)
                ring_t = torch.from_numpy(ring_np).to(device)
            else:
                ring_t = catchall_t

            scores_h = scores.clone()
            if h_idx == 0 and use_ew:
                scores_h = scores_h + bonus
            valid = ring_t
            if already_mask is not None:
                valid = valid & ~already_mask
            scores_h = scores_h.masked_fill(~valid, NEG_INF)

            top_v, top_i = torch.topk(scores_h, k=k_take, dim=1)
            invalid = top_v < (NEG_INF / 2)
            top_i = top_i.masked_fill(invalid, -1)
            out[h_idx][cstart:cend] = top_i.cpu().numpy()

        if return_pad:
            valid_pad = catchall_t
            if already_mask is not None:
                valid_pad = valid_pad & ~already_mask
            scores_pad = scores.masked_fill(~valid_pad, NEG_INF)
            top_v, top_i = torch.topk(scores_pad, k=k_take, dim=1)
            invalid = top_v < (NEG_INF / 2)
            top_i = top_i.masked_fill(invalid, -1)
            pad_out[cstart:cend] = top_i.cpu().numpy()

    if return_pad:
        return out, pad_out
    return out


def _vectorized_easy_assemble(per_ring_arrays, hop_k, total_k):
    """Fast path: when every ring already has hop_k[h] valid picks, the result
    is just concat(per_ring[h][:, :hop_k[h]]). Returns (result, is_easy)."""
    n_users = per_ring_arrays[0].shape[0]
    n_rings = min(len(per_ring_arrays), len(hop_k))
    K = per_ring_arrays[0].shape[1]
    is_easy = np.ones(n_users, dtype=bool)
    for h in range(n_rings):
        if hop_k[h] > K:
            is_easy[:] = False
            break
        is_easy &= (per_ring_arrays[h][:, :hop_k[h]] >= 0).all(axis=1)
    result = np.full((n_users, total_k), -1, dtype=np.int64)
    offset = 0
    for h in range(n_rings):
        result[is_easy, offset:offset + hop_k[h]] = per_ring_arrays[h][is_easy, :hop_k[h]]
        offset += hop_k[h]
    return result, is_easy


# ── Per-user hard-path overflow assembly (parallelisable) ─────────────────────

def _random_pad_sample(rng, all_items, used, n):
    if n <= 0:
        return []
    pool = [x for x in all_items if x not in used]
    return rng.sample(pool, min(n, len(pool)))


def _assemble_with_overflow(per_ring_picks, hop_k, total_k, all_items, rng,
                            already_set=None, pad_picks=None):
    result = []
    overflow = []
    for h_idx, k_want in enumerate(hop_k):
        if h_idx >= len(per_ring_picks):
            continue
        picks = per_ring_picks[h_idx]
        picks = picks[picks >= 0]
        if already_set is not None and len(picks):
            mask = np.fromiter((p not in already_set for p in picks),
                               dtype=np.bool_, count=len(picks))
            picks = picks[mask]
        ring_size = len(picks)

        take_ov = min(len(overflow), max(0, k_want - ring_size))
        result += overflow[:take_ov]
        if already_set is not None:
            already_set.update(overflow[:take_ov])
        overflow = overflow[take_ov:]
        k_want -= take_ov

        if ring_size == 0:
            continue
        sel = picks[:k_want].tolist()
        result += sel
        if already_set is not None:
            already_set.update(sel)
        overflow += picks[k_want:].tolist()

    if len(result) < total_k:
        extra = [x for x in overflow
                 if already_set is None or x not in already_set]
        result += extra[:total_k - len(result)]
        if already_set is not None:
            already_set.update(extra[:total_k - len(result)])
    if len(result) < total_k:
        used = set(result) if already_set is None else already_set
        if pad_picks is not None:
            for p in pad_picks:
                if len(result) >= total_k:
                    break
                p = int(p)
                if p < 0 or p in used:
                    continue
                result.append(p); used.add(p)
        if len(result) < total_k:
            result += _random_pad_sample(rng, all_items, used,
                                         total_k - len(result))
    return result[:total_k]


_HARD_WORKER_STATE = {}


def _hard_worker_init(per_ring_arrays, hop_k, total_k, all_items, seed,
                     already_arr=None, pad_picks=None):
    _HARD_WORKER_STATE.update({
        "per_ring": per_ring_arrays, "hop_k": hop_k, "total_k": total_k,
        "all_items": all_items, "seed": seed,
        "already_arr": already_arr, "pad_picks": pad_picks,
    })


def _hard_worker_process(uids):
    s = _HARD_WORKER_STATE
    out = []
    n_rings = len(s["per_ring"])
    for uid in uids:
        rng = random.Random(s["seed"] + int(uid))
        user_picks = [s["per_ring"][h][uid] for h in range(n_rings)]
        already = None
        if s["already_arr"] is not None:
            row = s["already_arr"][uid]
            already = set(int(x) for x in row[row >= 0])
        pad_user = s["pad_picks"][uid] if s["pad_picks"] is not None else None
        out.append((int(uid), _assemble_with_overflow(
            user_picks, s["hop_k"], s["total_k"], s["all_items"], rng,
            already_set=already, pad_picks=pad_user,
        )))
    return out


def _process_hard_users(hard_uids, per_ring_arrays, hop_k, total_k, all_items,
                       seed, already_arr=None, pad_picks=None, n_workers=0):
    if len(hard_uids) == 0:
        return {}
    if n_workers <= 1:
        _hard_worker_init(per_ring_arrays, hop_k, total_k, all_items, seed,
                         already_arr=already_arr, pad_picks=pad_picks)
        return dict(_hard_worker_process(list(hard_uids)))
    chunk = max(1, len(hard_uids) // (n_workers * 4))
    chunks = [hard_uids[i:i + chunk] for i in range(0, len(hard_uids), chunk)]
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers,
        initializer=_hard_worker_init,
        initargs=(per_ring_arrays, hop_k, total_k, all_items, seed,
                  already_arr, pad_picks),
    ) as pool:
        results = {}
        for batch in pool.imap_unordered(_hard_worker_process, chunks):
            for uid, r in batch:
                results[uid] = r
        return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(ranked_lists, targets, ks):
    out = {}
    for k in ks:
        recall = np.mean([1.0 if t in r[:k] else 0.0
                          for r, t in zip(ranked_lists, targets)])
        ndcg = []
        for r, t in zip(ranked_lists, targets):
            top = r[:k]
            if t in top:
                rank = top.index(t) + 1
                ndcg.append(1.0 / np.log2(rank + 1))
            else:
                ndcg.append(0.0)
        out[f"Recall@{k}"] = recall
        out[f"NDCG@{k}"] = float(np.mean(ndcg))
    return out


# ── TGH-1: single-source ring-topk ────────────────────────────────────────────

def run_tgh1(
    sec_last, history, all_items, emb_t, hop_k, last_n, seed,
    edge_counts, edge_weight_alpha,
    rings_csr, visited_csr, n_users,
    chunk_size=8000, n_assemble_workers=1, semantic_pad=False,
):
    use_edge_w = edge_counts is not None and edge_weight_alpha > 0
    depth = len(hop_k)
    total_k = sum(hop_k)
    src_arr = np.array([sec_last[u] for u in range(n_users)], dtype=np.int64)
    anchor_t = build_user_anchors_mean(emb_t, history, n_users, last_n)
    ew_csr = _build_edge_weight_csr(edge_counts, emb_t.shape[0]) if use_edge_w else None

    gpu_out = gpu_per_ring_topk(
        anchor_t, emb_t, src_arr, depth, total_k,
        rings_csr=rings_csr, visited_csr=visited_csr, history=history,
        edge_w_alpha=edge_weight_alpha, edge_w_csr=ew_csr,
        chunk_size=chunk_size, return_pad=semantic_pad,
    )
    if semantic_pad:
        per_ring_picks, pad_picks = gpu_out
    else:
        per_ring_picks, pad_picks = gpu_out, None

    easy_result, is_easy = _vectorized_easy_assemble(per_ring_picks, hop_k, total_k)
    hard_uids = np.where(~is_easy)[0]
    n_workers = max(1, min(n_assemble_workers or 1, len(hard_uids) or 1))
    print(f"  TGH-1 assemble: {n_users:,} users, "
          f"{n_users - len(hard_uids):,} easy / {len(hard_uids):,} hard "
          f"(workers={n_workers}).")
    hard_results = _process_hard_users(
        hard_uids, per_ring_picks, hop_k, total_k, all_items, seed,
        pad_picks=pad_picks, n_workers=n_workers,
    )
    ranked_lists = [easy_result[u].tolist() for u in range(n_users)]
    for uid, r in hard_results.items():
        ranked_lists[uid] = r
    return ranked_lists


# ── TGH-2: multi-source ring-topk ─────────────────────────────────────────────

def run_tgh2(
    sec_last, history, all_items, emb_t,
    src2_hop_k, src3_hop_k, seed,
    edge_counts, edge_weight_alpha,
    rings_csr, visited_csr, n_users,
    chunk_size=8000, n_assemble_workers=1, semantic_pad=False,
):
    use_edge_w = edge_counts is not None and edge_weight_alpha > 0
    total_k = sum(src2_hop_k) + sum(src3_hop_k)
    ranked_lists = [None] * n_users

    src2_arr = np.array([sec_last[u] for u in range(n_users)], dtype=np.int64)
    src3_arr = np.array([
        history[u][-2] if len(history[u]) >= 2 else sec_last[u]
        for u in range(n_users)
    ], dtype=np.int64)
    src3_avail = np.array([len(history[u]) >= 2 for u in range(n_users)])

    anchor2_t = build_user_anchors_single(emb_t, src2_arr)
    anchor3_t = build_user_anchors_single(emb_t, src3_arr)
    ew_csr = _build_edge_weight_csr(edge_counts, emb_t.shape[0]) if use_edge_w else None

    src2_total_k = sum(src2_hop_k)
    gpu_out2 = gpu_per_ring_topk(
        anchor2_t, emb_t, src2_arr, len(src2_hop_k), src2_total_k,
        rings_csr=rings_csr, visited_csr=visited_csr, history=history,
        edge_w_alpha=edge_weight_alpha, edge_w_csr=ew_csr,
        chunk_size=chunk_size, return_pad=semantic_pad,
    )
    if semantic_pad:
        per_ring_src2, pad_picks2 = gpu_out2
    else:
        per_ring_src2, pad_picks2 = gpu_out2, None

    src2_easy, is_easy_2 = _vectorized_easy_assemble(per_ring_src2, src2_hop_k, src2_total_k)
    hard2 = np.where(~is_easy_2)[0]
    n_workers_eff = max(1, min(n_assemble_workers or 1, max(1, len(hard2))))
    print(f"  TGH-2 src2 assemble: {n_users:,} users, "
          f"{n_users - len(hard2):,} easy / {len(hard2):,} hard "
          f"(workers={n_workers_eff}).")
    hard2_results = _process_hard_users(
        hard2, per_ring_src2, src2_hop_k, src2_total_k, all_items, seed,
        pad_picks=pad_picks2, n_workers=n_workers_eff,
    )
    src2_results = [src2_easy[u].tolist() for u in range(n_users)]
    for uid, r in hard2_results.items():
        src2_results[uid] = r

    already_arr = np.full((n_users, src2_total_k), -1, dtype=np.int64)
    for uid in range(n_users):
        r = [int(x) for x in src2_results[uid]]
        already_arr[uid, :len(r)] = r

    pad_picks3 = None
    if src3_hop_k:
        src3_total_k = sum(src3_hop_k)
        gpu_out3 = gpu_per_ring_topk(
            anchor3_t, emb_t, src3_arr, len(src3_hop_k), src3_total_k,
            rings_csr=rings_csr, visited_csr=visited_csr, history=history,
            edge_w_alpha=edge_weight_alpha, edge_w_csr=ew_csr,
            already_arr=already_arr, chunk_size=chunk_size,
            return_pad=semantic_pad,
        )
        if semantic_pad:
            per_ring_src3, pad_picks3 = gpu_out3
        else:
            per_ring_src3 = gpu_out3

        src3_easy, is_easy_3 = _vectorized_easy_assemble(
            per_ring_src3, src3_hop_k, src3_total_k,
        )
        is_easy_3 &= src3_avail
        hard3 = np.where(~is_easy_3)[0]
        n_workers_eff3 = max(1, min(n_assemble_workers or 1, max(1, len(hard3))))
        print(f"  TGH-2 src3 assemble: {n_users:,} users, "
              f"{n_users - len(hard3):,} easy / {len(hard3):,} hard "
              f"(workers={n_workers_eff3}).")
        hard3_results = _process_hard_users(
            hard3, per_ring_src3, src3_hop_k, src3_total_k, all_items, seed,
            already_arr=already_arr, pad_picks=pad_picks3,
            n_workers=n_workers_eff3,
        )
    else:
        src3_easy = None
        hard3_results = {}

    for uid in range(n_users):
        result2 = src2_results[uid]
        if src3_hop_k:
            if uid in hard3_results:
                result3 = hard3_results[uid] if src3_avail[uid] else []
            else:
                result3 = src3_easy[uid].tolist()
        else:
            result3 = []
        combined = list(result2) + list(result3)
        if len(combined) < total_k:
            already = set(int(x) for x in combined)
            user_rng = random.Random(seed + uid)
            if semantic_pad and pad_picks2 is not None:
                for p in pad_picks2[uid]:
                    if len(combined) >= total_k:
                        break
                    p = int(p)
                    if p < 0 or p in already:
                        continue
                    combined.append(p); already.add(p)
            if len(combined) < total_k:
                combined += _random_pad_sample(user_rng, all_items, already,
                                               total_k - len(combined))
        ranked_lists[uid] = combined[:total_k]

    return ranked_lists


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty", choices=list(DATASET_CONFIGS),
                        help="Dataset name (folder under data/).")
    parser.add_argument("--data_dir", default=None,
                        help="Override data dir (default: data/<dataset>).")
    parser.add_argument("--emb_path", default=None,
                        help="Override item embedding path "
                             "(default: data/<dataset>/t5xl.pkl).")
    parser.add_argument("--method", choices=["TGH-1", "TGH-2"], default="TGH-1")
    parser.add_argument("--max_hops", type=int, default=2,
                        help="Max ring depth to materialise globally.")
    parser.add_argument("--last_n", type=int, default=1,
                        help="Number of last items averaged into the TGH-1 anchor.")
    parser.add_argument("--hop_k", type=int, nargs="+", default=[7, 2, 1],
                        help="TGH-1 per-ring budget (e.g. 7 2 1).")
    parser.add_argument("--src2_hop_k", type=int, nargs="+", default=[5, 1],
                        help="TGH-2 source-2 per-ring budget (sec_last anchor).")
    parser.add_argument("--src3_hop_k", type=int, nargs="+", default=[3, 1],
                        help="TGH-2 source-3 per-ring budget (history[-2] anchor).")
    parser.add_argument("--edge_weight_alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--user_chunk", type=int, default=None,
                        help="Users per GPU matmul. Defaults to the per-dataset "
                             "value in dataset_configs.py, which scales as "
                             "1/n_items; lower it if you OOM.")
    parser.add_argument("--max_explicit_depth", type=int, default=None,
                        help="Cap on rings materialised globally (deeper rings "
                             "become catch-all NOT(visited ∪ history)).")
    parser.add_argument("--assemble_workers", type=int, default=8)
    parser.add_argument("--semantic_pad", action="store_true",
                        help="Fill deficits from the catch-all pool by anchor "
                             "similarity instead of random sampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="predictions",
                        help="Where to save predictions/<dataset>/<method>.pkl.")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.data_dir is None: args.data_dir = cfg["data_dir"]
    if args.emb_path is None: args.emb_path = cfg["emb_path"]
    if args.user_chunk is None: args.user_chunk = cfg["user_chunk"]

    tf.get_logger().setLevel("ERROR")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    print("\n── Loading sequences ──")
    gt, sec_last, history = load_test_sequences(args.data_dir)
    print("\n── Loading transition graph ──")
    edge_counts = load_graph(args.data_dir)

    old_uids = sorted(gt.keys())
    if old_uids != list(range(len(old_uids))):
        print(f"  (renumbering {len(old_uids):,} non-contiguous uids)")
        uid_map = {old: new for new, old in enumerate(old_uids)}
        gt       = {uid_map[u]: gt[u]       for u in old_uids}
        sec_last = {uid_map[u]: sec_last[u] for u in old_uids}
        history  = {uid_map[u]: history[u]  for u in old_uids}
    n_users = len(gt)

    max_item_id = 0
    for (a, b) in edge_counts:
        max_item_id = max(max_item_id, a, b)
    for u in range(n_users):
        if history[u]:
            max_item_id = max(max_item_id, max(history[u]))
        max_item_id = max(max_item_id, gt[u], sec_last[u])
    n_items = max_item_id + 1
    print(f"  Users: {n_users:,}  Items: {n_items:,}  Edges: {len(edge_counts):,}")

    print("\n── Building adjacency / ring matrices ──")
    t1 = time.time()
    A_csr = build_sparse_adj(edge_counts, n_items)
    if args.method == "TGH-1":
        max_depth = max(args.max_hops, len(args.hop_k))
    else:
        max_depth = max(args.max_hops, len(args.src2_hop_k),
                        len(args.src3_hop_k) if args.src3_hop_k else 0)
    n_explicit = args.max_explicit_depth or max_depth
    n_explicit = min(max_depth, n_explicit)
    print(f"  explicit depth={n_explicit} of {max_depth}")
    rings_csr, visited_csr = compute_hop_rings_csr(
        A_csr, max_depth, max_explicit_depth=n_explicit,
    )
    all_items = list(set(n for edge in edge_counts for n in edge))
    print(f"  done in {time.time()-t1:.1f}s.")

    print("\n── Loading item embeddings → GPU ──")
    emb_matrix = load_item_embeddings(args.emb_path)
    if emb_matrix.shape[0] < n_items:
        pad = np.zeros((n_items - emb_matrix.shape[0], emb_matrix.shape[1]), dtype=np.float32)
        emb_matrix = np.concatenate([emb_matrix, pad], axis=0)
    emb_t = torch.from_numpy(emb_matrix[:n_items]).to(device)
    print(f"  emb_t shape={tuple(emb_t.shape)}, dtype={emb_t.dtype}")

    targets = [gt[uid] for uid in range(n_users)]
    ks = [1, 5, 10]

    if args.method == "TGH-1":
        print(f"\n── TGH-1 (hop_k={args.hop_k}, alpha={args.edge_weight_alpha}) ──")
        t1 = time.time()
        ranked_lists = run_tgh1(
            sec_last, history, all_items, emb_t,
            args.hop_k, args.last_n, args.seed,
            edge_counts, args.edge_weight_alpha,
            rings_csr, visited_csr, n_users,
            args.user_chunk, args.assemble_workers, args.semantic_pad,
        )
        print(f"  done in {time.time()-t1:.1f}s.")
    else:
        print(f"\n── TGH-2 (src2={args.src2_hop_k}, src3={args.src3_hop_k}, "
              f"alpha={args.edge_weight_alpha}) ──")
        t1 = time.time()
        ranked_lists = run_tgh2(
            sec_last, history, all_items, emb_t,
            args.src2_hop_k, args.src3_hop_k, args.seed,
            edge_counts, args.edge_weight_alpha,
            rings_csr, visited_csr, n_users,
            args.user_chunk, args.assemble_workers, args.semantic_pad,
        )
        print(f"  done in {time.time()-t1:.1f}s.")

    metrics = compute_metrics(ranked_lists, targets, ks)
    print(f"\n=== {args.method} METRICS (n={n_users:,}) ===")
    for m, v in metrics.items():
        print(f"  {m:<12}: {v*100:.2f}%")

    save_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{args.method}.pkl")
    preds = [{
        "user_id": uid,
        "item_ids": ranked_lists[uid][:args.top_k],
        "gt_ids": [targets[uid]],
    } for uid in range(n_users)]
    with open(out_path, "wb") as f:
        pickle.dump(preds, f)
    print(f"\nSaved {out_path} ({len(preds)} users)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
