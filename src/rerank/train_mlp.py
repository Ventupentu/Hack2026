"""Train a lightweight MLP reranker on top of embedding top-k candidates.

The script expects precomputed query/product embeddings and supervised
bundle->product matches. It builds top-k candidates by cosine similarity,
samples hard negatives from that pool, and optimizes a pairwise ranking loss.

Example:
    python -m src.rerank.train_mlp \
      --query-embeddings artifacts/embeddings/train_bundle_embeddings.pt \
      --product-embeddings artifacts/embeddings/product_embeddings.pt \
      --train-csv data/bundles_product_match_train.csv \
      --output artifacts/rerank/mlp_reranker.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureConfig:
    """Feature toggles for pairwise reranker inputs."""

    use_abs_diff: bool = True
    use_elem_product: bool = True
    use_sq_diff: bool = False
    use_raw_concat: bool = False


@dataclass
class QueryRecord:
    """Training/query metadata with positives and sampled hard negatives."""

    query_index: int
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    negative_probs: np.ndarray


class MLPReranker(nn.Module):
    """Small MLP scorer that outputs one relevance score per pair."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            if dim <= 0:
                raise ValueError(f"Hidden dim must be > 0, got {dim}.")
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_int_list(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_matching_key(payload: Mapping[str, object], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in payload:
            return key
    return None


def _coerce_id_list(value: object) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if torch.is_tensor(value):
        return [str(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in list(value)]
    raise TypeError(f"Unsupported ids container type: {type(value)!r}")


def _coerce_embedding_matrix(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported embedding container type: {type(value)!r}")
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N, D], got shape={arr.shape}.")
    return arr.astype(np.float32)


def load_embeddings(
    path: Path,
    id_keys: Sequence[str],
    emb_keys: Sequence[str],
    normalize: bool = True,
) -> Tuple[List[str], np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Expected dict-like payload in {path}. "
                "Save format should include ids + embeddings."
            )
        id_key = _first_matching_key(payload, id_keys)
        emb_key = _first_matching_key(payload, emb_keys)
        if id_key is None or emb_key is None:
            raise KeyError(
                f"Could not find ids/embeddings keys in {path}. "
                f"ids keys tried={list(id_keys)}, emb keys tried={list(emb_keys)}."
            )
        ids = _coerce_id_list(payload[id_key])
        embeddings = _coerce_embedding_matrix(payload[emb_key])
    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            id_key = _first_matching_key(payload, id_keys)
            emb_key = _first_matching_key(payload, emb_keys)
            if id_key is None or emb_key is None:
                raise KeyError(
                    f"Could not find ids/embeddings keys in {path}. "
                    f"ids keys tried={list(id_keys)}, emb keys tried={list(emb_keys)}."
                )
            ids = _coerce_id_list(payload[id_key])
            embeddings = _coerce_embedding_matrix(payload[emb_key])
    else:
        raise ValueError(
            f"Unsupported embedding format for {path}. "
            "Use .pt/.pth or .npz containing ids + embeddings."
        )

    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs/embedding length mismatch in {path}: "
            f"{len(ids)} ids vs {embeddings.shape[0]} rows."
        )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-8, a_max=None)
    return ids, embeddings.astype(np.float32)


def load_positive_map(train_csv: Path) -> Dict[str, Set[str]]:
    df = pd.read_csv(train_csv, usecols=["bundle_asset_id", "product_asset_id"], dtype=str)
    positives: Dict[str, Set[str]] = {}
    for row in df.itertuples(index=False):
        bid = str(row.bundle_asset_id)
        pid = str(row.product_asset_id)
        positives.setdefault(bid, set()).add(pid)
    return positives


def split_train_val(
    bundle_ids: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    ids = list(bundle_ids)
    if not ids:
        return [], []
    if val_ratio <= 0:
        return ids, []
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_size = int(round(len(ids) * val_ratio))
    val_size = max(1, min(len(ids) - 1, val_size)) if len(ids) > 1 else 0
    val_ids = sorted(ids[:val_size])
    train_ids = sorted(ids[val_size:])
    return train_ids, val_ids


def compute_topk_candidates(
    query_embeddings: np.ndarray,
    product_embeddings: np.ndarray,
    topk: int,
    device: torch.device,
    query_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = query_embeddings.shape[0]
    n_products = product_embeddings.shape[0]
    k = min(topk, n_products)
    if k <= 0:
        raise ValueError("topk must be > 0.")

    q = torch.from_numpy(query_embeddings).to(device)
    p = torch.from_numpy(product_embeddings).to(device)
    topk_indices = np.empty((n_queries, k), dtype=np.int64)
    topk_scores = np.empty((n_queries, k), dtype=np.float32)

    for start in range(0, n_queries, query_batch_size):
        end = min(start + query_batch_size, n_queries)
        sims = q[start:end] @ p.T
        vals, idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
        topk_indices[start:end] = idx.detach().cpu().numpy()
        topk_scores[start:end] = vals.detach().cpu().numpy().astype(np.float32)

    return topk_indices, topk_scores


def _ids_equal(a: Sequence[str], b: Sequence[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(str(x) == str(y) for x, y in zip(a, b))


def load_or_compute_candidates(
    cache_path: Optional[Path],
    query_ids: Sequence[str],
    product_ids: Sequence[str],
    query_embeddings: np.ndarray,
    product_embeddings: np.ndarray,
    topk: int,
    device: torch.device,
    query_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as payload:
                cached_query_ids = [str(v) for v in payload["query_ids"].tolist()]
                cached_product_ids = [str(v) for v in payload["product_ids"].tolist()]
                cached_topk = int(payload["topk"].item())
                if (
                    _ids_equal(cached_query_ids, list(query_ids))
                    and _ids_equal(cached_product_ids, list(product_ids))
                    and cached_topk == min(topk, len(product_ids))
                ):
                    print(f"Loaded candidate cache: {cache_path}")
                    return (
                        payload["topk_indices"].astype(np.int64),
                        payload["topk_scores"].astype(np.float32),
                    )
                print(f"Candidate cache mismatch, recomputing: {cache_path}")
        except Exception as exc:
            print(f"Failed to load candidate cache ({exc}), recomputing.")

    print(f"Computing top-{topk} candidates with embedding similarity...")
    topk_indices, topk_scores = compute_topk_candidates(
        query_embeddings=query_embeddings,
        product_embeddings=product_embeddings,
        topk=topk,
        device=device,
        query_batch_size=query_batch_size,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            query_ids=np.asarray(list(query_ids), dtype=object),
            product_ids=np.asarray(list(product_ids), dtype=object),
            topk=np.asarray([topk_indices.shape[1]], dtype=np.int64),
            topk_indices=topk_indices.astype(np.int32),
            topk_scores=topk_scores.astype(np.float32),
        )
        print(f"Saved candidate cache: {cache_path}")

    return topk_indices, topk_scores


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - float(np.max(x))
    exp_x = np.exp(shifted)
    return exp_x / np.clip(exp_x.sum(), a_min=1e-8, a_max=None)


def build_query_records(
    bundle_ids: Sequence[str],
    bundle_to_index: Mapping[str, int],
    positives_map: Mapping[str, Set[str]],
    product_to_index: Mapping[str, int],
    topk_indices: np.ndarray,
    topk_scores: np.ndarray,
    hard_pool_size: int,
    hard_temperature: float,
    min_negatives: int,
) -> List[QueryRecord]:
    records: List[QueryRecord] = []
    dropped_missing_gt = 0
    dropped_missing_negs = 0

    for bundle_id in bundle_ids:
        query_idx = bundle_to_index[bundle_id]
        positives = positives_map.get(bundle_id, set())
        pos_indices = sorted({product_to_index[pid] for pid in positives if pid in product_to_index})
        if not pos_indices:
            dropped_missing_gt += 1
            continue
        pos_set = set(pos_indices)

        neg_idx: List[int] = []
        neg_scores: List[float] = []
        row_indices = topk_indices[query_idx]
        row_scores = topk_scores[query_idx]
        for cand_idx, cand_score in zip(row_indices.tolist(), row_scores.tolist()):
            if cand_idx in pos_set:
                continue
            neg_idx.append(int(cand_idx))
            neg_scores.append(float(cand_score))
            if hard_pool_size > 0 and len(neg_idx) >= hard_pool_size:
                break

        if len(neg_idx) < max(1, min_negatives):
            dropped_missing_negs += 1
            continue

        neg_indices_arr = np.asarray(neg_idx, dtype=np.int64)
        neg_scores_arr = np.asarray(neg_scores, dtype=np.float32)
        if hard_temperature <= 0:
            neg_probs = np.full_like(neg_scores_arr, fill_value=1.0 / len(neg_scores_arr))
        else:
            neg_probs = _softmax(neg_scores_arr / hard_temperature).astype(np.float32)

        records.append(
            QueryRecord(
                query_index=query_idx,
                positive_indices=np.asarray(pos_indices, dtype=np.int64),
                negative_indices=neg_indices_arr,
                negative_probs=neg_probs,
            )
        )

    print(
        "Built query records: "
        f"{len(records)} usable | dropped(no gt)={dropped_missing_gt} "
        f"| dropped(no hard negatives)={dropped_missing_negs}"
    )
    return records


def infer_feature_dim(embedding_dim: int, cfg: FeatureConfig) -> int:
    dim = 1  # cosine similarity
    if cfg.use_abs_diff:
        dim += embedding_dim
    if cfg.use_elem_product:
        dim += embedding_dim
    if cfg.use_sq_diff:
        dim += embedding_dim
    if cfg.use_raw_concat:
        dim += embedding_dim * 2
    return dim


def build_pair_features(
    query_vecs: torch.Tensor,
    product_vecs: torch.Tensor,
    cfg: FeatureConfig,
) -> torch.Tensor:
    sim = (query_vecs * product_vecs).sum(dim=1, keepdim=True)
    parts = [sim]

    if cfg.use_abs_diff:
        parts.append(torch.abs(query_vecs - product_vecs))
    if cfg.use_elem_product:
        parts.append(query_vecs * product_vecs)
    if cfg.use_sq_diff:
        diff = query_vecs - product_vecs
        parts.append(diff * diff)
    if cfg.use_raw_concat:
        parts.extend([query_vecs, product_vecs])

    return torch.cat(parts, dim=1)


def sample_triplets(
    records: Sequence[QueryRecord],
    samples_per_query: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples_per_query <= 0:
        raise ValueError("samples_per_query must be > 0.")

    total = len(records) * samples_per_query
    q_idx = np.empty(total, dtype=np.int64)
    pos_idx = np.empty(total, dtype=np.int64)
    neg_idx = np.empty(total, dtype=np.int64)

    cursor = 0
    for record in records:
        for _ in range(samples_per_query):
            q_idx[cursor] = record.query_index
            pos_idx[cursor] = rng.choice(record.positive_indices)
            neg_idx[cursor] = rng.choice(record.negative_indices, p=record.negative_probs)
            cursor += 1

    order = rng.permutation(total)
    return q_idx[order], pos_idx[order], neg_idx[order]


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    query_tensor: torch.Tensor,
    product_tensor: torch.Tensor,
    records: Sequence[QueryRecord],
    samples_per_query: int,
    batch_size: int,
    feature_cfg: FeatureConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    model.train()
    q_idx, pos_idx, neg_idx = sample_triplets(records, samples_per_query, rng)
    total = len(q_idx)
    loss_sum = 0.0
    margin_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx_q = torch.from_numpy(q_idx[start:end]).to(device=query_tensor.device, dtype=torch.long)
        idx_pos = torch.from_numpy(pos_idx[start:end]).to(device=product_tensor.device, dtype=torch.long)
        idx_neg = torch.from_numpy(neg_idx[start:end]).to(device=product_tensor.device, dtype=torch.long)
        batch_q = query_tensor.index_select(0, idx_q)
        batch_pos = product_tensor.index_select(0, idx_pos)
        batch_neg = product_tensor.index_select(0, idx_neg)

        pos_features = build_pair_features(batch_q, batch_pos, feature_cfg)
        neg_features = build_pair_features(batch_q, batch_neg, feature_cfg)
        pos_scores = model(pos_features)
        neg_scores = model(neg_features)
        margin = pos_scores - neg_scores
        loss = F.softplus(-margin).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_count = end - start
        loss_sum += float(loss.item()) * batch_count
        margin_sum += float(margin.mean().item()) * batch_count

    return loss_sum / max(1, total), margin_sum / max(1, total)


@torch.no_grad()
def score_candidate_pairs(
    model: nn.Module,
    query_tensor: torch.Tensor,
    product_tensor: torch.Tensor,
    topk_indices: np.ndarray,
    feature_cfg: FeatureConfig,
    infer_batch_size: int,
) -> np.ndarray:
    model.eval()
    n_queries, k = topk_indices.shape
    flat_q = np.repeat(np.arange(n_queries, dtype=np.int64), k)
    flat_p = topk_indices.reshape(-1)
    scores = np.empty(flat_q.shape[0], dtype=np.float32)

    for start in range(0, flat_q.shape[0], infer_batch_size):
        end = min(start + infer_batch_size, flat_q.shape[0])
        q_idx = torch.from_numpy(flat_q[start:end]).to(device=query_tensor.device, dtype=torch.long)
        p_idx = torch.from_numpy(flat_p[start:end]).to(device=product_tensor.device, dtype=torch.long)
        q_vec = query_tensor.index_select(0, q_idx)
        p_vec = product_tensor.index_select(0, p_idx)
        features = build_pair_features(q_vec, p_vec, feature_cfg)
        batch_scores = model(features)
        scores[start:end] = batch_scores.detach().cpu().numpy().astype(np.float32)

    return scores.reshape(n_queries, k)


def evaluate_rankings(
    eval_query_ids: Sequence[str],
    bundle_to_index: Mapping[str, int],
    positives_map: Mapping[str, Set[str]],
    product_to_index: Mapping[str, int],
    ranked_indices: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    k_max = max(ks)
    sum_recall: Dict[int, float] = {k: 0.0 for k in ks}
    sum_hit: Dict[int, float] = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n_eval = 0

    for bundle_id in eval_query_ids:
        gt_products = positives_map.get(bundle_id, set())
        gt_idx = {product_to_index[pid] for pid in gt_products if pid in product_to_index}
        if not gt_idx:
            continue
        q_idx = bundle_to_index[bundle_id]
        pred = ranked_indices[q_idx, :k_max].tolist()
        n_eval += 1

        for k in ks:
            top = pred[:k]
            hits = sum(1 for idx in top if idx in gt_idx)
            sum_recall[k] += hits / max(1, len(gt_idx))
            sum_hit[k] += 1.0 if hits > 0 else 0.0

        reciprocal_rank = 0.0
        for rank, idx in enumerate(pred, start=1):
            if idx in gt_idx:
                reciprocal_rank = 1.0 / rank
                break
        mrr_sum += reciprocal_rank

    metrics: Dict[str, float] = {"queries": float(n_eval)}
    if n_eval == 0:
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0.0
        metrics[f"mrr@{k_max}"] = 0.0
        return metrics

    for k in ks:
        metrics[f"recall@{k}"] = sum_recall[k] / n_eval
        metrics[f"hit@{k}"] = sum_hit[k] / n_eval
    metrics[f"mrr@{k_max}"] = mrr_sum / n_eval
    return metrics


def metrics_to_string(metrics: Mapping[str, float], ks: Sequence[int]) -> str:
    parts = []
    for k in ks:
        parts.append(f"R@{k}={metrics[f'recall@{k}']:.4f}")
    parts.append(f"MRR@{max(ks)}={metrics[f'mrr@{max(ks)}']:.4f}")
    return " | ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--product-embeddings", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rerank/mlp_reranker.pt"))
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path("artifacts/rerank/top200_candidates.npz"),
        help="Cache for top-k candidate indices/scores.",
    )

    parser.add_argument("--query-id-key", type=str, default="bundle_ids,query_ids,ids")
    parser.add_argument("--query-emb-key", type=str, default="embeddings,query_embeddings,bundle_embeddings")
    parser.add_argument("--product-id-key", type=str, default="product_ids,pids,ids")
    parser.add_argument("--product-emb-key", type=str, default="embeddings,product_embeddings")
    parser.add_argument("--disable-normalize", action="store_true")

    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--hard-pool-size", type=int, default=80)
    parser.add_argument("--hard-temperature", type=float, default=0.04)
    parser.add_argument("--min-negatives", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-query", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=str, default="512,128")
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-ks", type=str, default="5,10,15")
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.20,
        help="Final score = mlp_score + blend_alpha * cosine_score.",
    )

    parser.add_argument("--use-sq-diff", action="store_true")
    parser.add_argument("--use-raw-concat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    query_id_keys = [k.strip() for k in args.query_id_key.split(",") if k.strip()]
    query_emb_keys = [k.strip() for k in args.query_emb_key.split(",") if k.strip()]
    product_id_keys = [k.strip() for k in args.product_id_key.split(",") if k.strip()]
    product_emb_keys = [k.strip() for k in args.product_emb_key.split(",") if k.strip()]
    hidden_dims = parse_int_list(args.hidden_dims)
    eval_ks = sorted(set(parse_int_list(args.eval_ks)))

    feature_cfg = FeatureConfig(
        use_abs_diff=True,
        use_elem_product=True,
        use_sq_diff=bool(args.use_sq_diff),
        use_raw_concat=bool(args.use_raw_concat),
    )

    query_ids, query_embeddings = load_embeddings(
        path=args.query_embeddings,
        id_keys=query_id_keys,
        emb_keys=query_emb_keys,
        normalize=not args.disable_normalize,
    )
    product_ids, product_embeddings = load_embeddings(
        path=args.product_embeddings,
        id_keys=product_id_keys,
        emb_keys=product_emb_keys,
        normalize=not args.disable_normalize,
    )
    if query_embeddings.shape[1] != product_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: query={query_embeddings.shape[1]} "
            f"vs product={product_embeddings.shape[1]}"
        )

    positives_map = load_positive_map(args.train_csv)
    bundle_to_index = {bid: idx for idx, bid in enumerate(query_ids)}
    product_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

    eligible_bundle_ids = [
        bid
        for bid in query_ids
        if bid in positives_map and any(pid in product_to_index for pid in positives_map[bid])
    ]
    if len(eligible_bundle_ids) < 2:
        raise RuntimeError(
            "Not enough eligible bundles with positives after alignment. "
            "Check embeddings ids vs train csv ids."
        )
    train_ids, val_ids = split_train_val(eligible_bundle_ids, val_ratio=args.val_ratio, seed=args.seed)
    print(
        f"Queries={len(query_ids)} (eligible={len(eligible_bundle_ids)}) | "
        f"Products={len(product_ids)} | Train queries={len(train_ids)} | Val queries={len(val_ids)}"
    )

    topk_indices, topk_scores = load_or_compute_candidates(
        cache_path=args.candidate_cache,
        query_ids=query_ids,
        product_ids=product_ids,
        query_embeddings=query_embeddings,
        product_embeddings=product_embeddings,
        topk=args.topk,
        device=device,
        query_batch_size=args.query_batch_size,
    )

    train_records = build_query_records(
        bundle_ids=train_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        hard_pool_size=args.hard_pool_size,
        hard_temperature=args.hard_temperature,
        min_negatives=args.min_negatives,
    )
    if not train_records:
        raise RuntimeError("No valid training records. Try larger topk/hard_pool_size.")

    embedding_dim = query_embeddings.shape[1]
    input_dim = infer_feature_dim(embedding_dim, feature_cfg)
    model = MLPReranker(input_dim=input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    query_tensor = torch.from_numpy(query_embeddings).to(device)
    product_tensor = torch.from_numpy(product_embeddings).to(device)
    rng = np.random.default_rng(args.seed)

    baseline_metrics = evaluate_rankings(
        eval_query_ids=val_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        ranked_indices=topk_indices,
        ks=eval_ks,
    )
    print(f"Baseline (cosine only): {metrics_to_string(baseline_metrics, eval_ks)}")

    best_metric_name = f"recall@{max(eval_ks)}"
    best_metric = baseline_metrics.get(best_metric_name, 0.0)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_margin = train_one_epoch(
            model=model,
            optimizer=optimizer,
            query_tensor=query_tensor,
            product_tensor=product_tensor,
            records=train_records,
            samples_per_query=args.samples_per_query,
            batch_size=args.batch_size,
            feature_cfg=feature_cfg,
            rng=rng,
        )

        if val_ids:
            rerank_scores = score_candidate_pairs(
                model=model,
                query_tensor=query_tensor,
                product_tensor=product_tensor,
                topk_indices=topk_indices,
                feature_cfg=feature_cfg,
                infer_batch_size=args.eval_batch_size,
            )
            final_scores = rerank_scores + (args.blend_alpha * topk_scores)
            rerank_order = np.argsort(-final_scores, axis=1)
            reranked_indices = np.take_along_axis(topk_indices, rerank_order, axis=1)

            val_metrics = evaluate_rankings(
                eval_query_ids=val_ids,
                bundle_to_index=bundle_to_index,
                positives_map=positives_map,
                product_to_index=product_to_index,
                ranked_indices=reranked_indices,
                ks=eval_ks,
            )
            current_metric = val_metrics.get(best_metric_name, 0.0)
            is_best = current_metric > best_metric
        else:
            val_metrics = {f"recall@{k}": 0.0 for k in eval_ks}
            val_metrics[f"mrr@{max(eval_ks)}"] = 0.0
            val_metrics["queries"] = 0.0
            current_metric = train_margin
            is_best = True

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_margin": float(train_margin),
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | margin={train_margin:.4f} | "
            + (metrics_to_string(val_metrics, eval_ks) if val_ids else "validation skipped")
            + (" | best" if is_best else "")
        )

    payload = {
        "model_state": best_state,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "embedding_dim": embedding_dim,
        "feature_config": asdict(feature_cfg),
        "blend_alpha": args.blend_alpha,
        "query_embeddings_path": str(args.query_embeddings),
        "product_embeddings_path": str(args.product_embeddings),
        "train_csv": str(args.train_csv),
        "topk": int(topk_indices.shape[1]),
        "eval_ks": eval_ks,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "baseline_metrics": baseline_metrics,
        "best_val_metric_name": best_metric_name,
        "best_val_metric": best_metric,
        "history": history,
    }

    report_payload = {k: v for k, v in payload.items() if k != "model_state"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved best reranker checkpoint to: {args.output}")

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"Saved training report to: {metrics_path}")
    print(
        f"Best epoch={best_epoch} | baseline {best_metric_name}="
        f"{baseline_metrics.get(best_metric_name, 0.0):.4f} | "
        f"rerank {best_metric_name}={best_metric:.4f}"
    )


if __name__ == "__main__":
    main()
