"""Train a lightweight listwise MLP reranker on top of embedding retrieval.

The script expects precomputed query/product embeddings and supervised
bundle->product matches. It builds top-k candidates by cosine similarity,
then optimizes a listwise multi-positive objective:

  loss(q) = -log( sum_{p in Pos(q)} exp(s(q,p)) / sum_{c in TopK(q)} exp(s(q,c)) )

It also reports retrieval coverage (fraction of queries with >=1 positive in topK),
which is the practical ceiling for reranking.

Example:
    python -m src.rerank.train_mlp \
      --query-embeddings artifacts/embeddings/train_bundle_embeddings.pt \
      --product-embeddings outputs/product_embeddings.pt \
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


@dataclass
class FeatureConfig:
    """Feature toggles for reranker input vector."""

    use_abs_diff: bool = True
    use_elem_product: bool = True
    use_sq_diff: bool = False
    use_raw_concat: bool = False
    use_query_features: bool = False


@dataclass
class QueryRecord:
    """One listwise training sample = one query + candidate list + positive mask."""

    query_index: int
    candidate_indices: np.ndarray  # [K]
    positive_mask: np.ndarray  # [K] bool


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


def _coerce_str_list(value: object) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in list(value)]
    return []


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


def _load_payload(path: Path) -> Mapping[str, object]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"Expected dict-like payload in {path}, got {type(payload)!r}")
        return payload
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}
    raise ValueError(f"Unsupported embedding format for {path}. Use .pt/.pth or .npz.")


def load_embeddings(
    path: Path,
    id_keys: Sequence[str],
    emb_keys: Sequence[str],
    normalize: bool = True,
    extra_keys: Optional[Sequence[str]] = None,
    extra_names_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], List[str]]:
    payload = _load_payload(path)
    id_key = _first_matching_key(payload, id_keys)
    emb_key = _first_matching_key(payload, emb_keys)
    if id_key is None or emb_key is None:
        raise KeyError(
            f"Could not find ids/embeddings keys in {path}. "
            f"ids tried={list(id_keys)}, emb tried={list(emb_keys)}."
        )

    ids = _coerce_id_list(payload[id_key])
    embeddings = _coerce_embedding_matrix(payload[emb_key])
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs/embedding length mismatch in {path}: "
            f"{len(ids)} ids vs {embeddings.shape[0]} rows."
        )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-8, a_max=None)

    extra_features: Optional[np.ndarray] = None
    extra_names: List[str] = []
    if extra_keys:
        extra_key = _first_matching_key(payload, extra_keys)
        if extra_key is not None:
            extra_candidate = _coerce_embedding_matrix(payload[extra_key])
            if extra_candidate.shape[0] == len(ids):
                extra_features = extra_candidate.astype(np.float32)
            else:
                print(
                    "Warning: query extra features ignored due to row mismatch "
                    f"({extra_candidate.shape[0]} vs {len(ids)})."
                )
    if extra_names_keys:
        names_key = _first_matching_key(payload, extra_names_keys)
        if names_key is not None:
            extra_names = _coerce_str_list(payload[names_key])

    return ids, embeddings.astype(np.float32), extra_features, extra_names


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


def compute_positive_coverage(
    bundle_ids: Sequence[str],
    bundle_to_index: Mapping[str, int],
    positives_map: Mapping[str, Set[str]],
    product_to_index: Mapping[str, int],
    topk_indices: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    if not ks:
        return {}
    max_k = topk_indices.shape[1]
    ks = sorted({min(max(1, k), max_k) for k in ks})
    total = 0
    at_least_one: Dict[int, int] = {k: 0 for k in ks}
    pos_in_k_sum: Dict[int, float] = {k: 0.0 for k in ks}

    for bid in bundle_ids:
        gt = positives_map.get(bid, set())
        gt_idx = {product_to_index[pid] for pid in gt if pid in product_to_index}
        if not gt_idx:
            continue
        q_idx = bundle_to_index[bid]
        row = topk_indices[q_idx].tolist()
        total += 1
        for k in ks:
            top = row[:k]
            hits = sum(1 for idx in top if idx in gt_idx)
            if hits > 0:
                at_least_one[k] += 1
            pos_in_k_sum[k] += float(hits)

    out: Dict[str, float] = {"queries": float(total)}
    if total == 0:
        for k in ks:
            out[f"coverage@{k}"] = 0.0
            out[f"avg_pos_in_top{k}"] = 0.0
        return out

    for k in ks:
        out[f"coverage@{k}"] = at_least_one[k] / total
        out[f"avg_pos_in_top{k}"] = pos_in_k_sum[k] / total
    return out


def coverage_to_string(metrics: Mapping[str, float], ks: Sequence[int]) -> str:
    parts = []
    for k in ks:
        parts.append(f"cov@{k}={metrics.get(f'coverage@{k}', 0.0):.4f}")
    return " | ".join(parts)


def build_listwise_records(
    bundle_ids: Sequence[str],
    bundle_to_index: Mapping[str, int],
    positives_map: Mapping[str, Set[str]],
    product_to_index: Mapping[str, int],
    topk_indices: np.ndarray,
) -> List[QueryRecord]:
    records: List[QueryRecord] = []
    dropped_no_gt = 0
    dropped_no_pos_in_pool = 0
    dropped_all_positive = 0

    for bid in bundle_ids:
        q_idx = bundle_to_index[bid]
        gt = positives_map.get(bid, set())
        gt_idx = {product_to_index[pid] for pid in gt if pid in product_to_index}
        if not gt_idx:
            dropped_no_gt += 1
            continue

        candidates = topk_indices[q_idx].astype(np.int64)
        pos_mask = np.asarray([idx in gt_idx for idx in candidates.tolist()], dtype=bool)
        pos_count = int(pos_mask.sum())
        if pos_count == 0:
            dropped_no_pos_in_pool += 1
            continue
        if pos_count == len(candidates):
            dropped_all_positive += 1
            continue

        records.append(
            QueryRecord(
                query_index=q_idx,
                candidate_indices=candidates,
                positive_mask=pos_mask,
            )
        )

    print(
        "Built listwise records: "
        f"{len(records)} usable | dropped(no gt)={dropped_no_gt} | "
        f"dropped(no pos in pool)={dropped_no_pos_in_pool} | "
        f"dropped(all positive)={dropped_all_positive}"
    )
    return records


def infer_feature_dim(embedding_dim: int, cfg: FeatureConfig, query_extra_dim: int) -> int:
    dim = 1  # cosine similarity
    if cfg.use_abs_diff:
        dim += embedding_dim
    if cfg.use_elem_product:
        dim += embedding_dim
    if cfg.use_sq_diff:
        dim += embedding_dim
    if cfg.use_raw_concat:
        dim += embedding_dim * 2
    if cfg.use_query_features:
        dim += query_extra_dim
    return dim


def build_pair_features(
    query_vecs: torch.Tensor,
    product_vecs: torch.Tensor,
    cfg: FeatureConfig,
    query_extra_vecs: Optional[torch.Tensor] = None,
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
    if cfg.use_query_features and query_extra_vecs is not None:
        parts.append(query_extra_vecs)

    return torch.cat(parts, dim=1)


def train_one_epoch_listwise(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    query_tensor: torch.Tensor,
    product_tensor: torch.Tensor,
    query_extra_tensor: Optional[torch.Tensor],
    records: Sequence[QueryRecord],
    batch_size: int,
    feature_cfg: FeatureConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    model.train()
    if not records:
        return 0.0, 0.0

    order = rng.permutation(len(records))
    loss_sum = 0.0
    posprob_sum = 0.0
    count_batches = 0

    for start in range(0, len(order), batch_size):
        end = min(start + batch_size, len(order))
        batch_ids = order[start:end].tolist()
        batch_records = [records[i] for i in batch_ids]
        bsz = len(batch_records)
        k = len(batch_records[0].candidate_indices)

        q_idx_np = np.asarray([r.query_index for r in batch_records], dtype=np.int64)
        cand_idx_np = np.stack([r.candidate_indices for r in batch_records], axis=0).astype(np.int64)
        pos_mask_np = np.stack([r.positive_mask for r in batch_records], axis=0).astype(bool)

        q_idx = torch.from_numpy(q_idx_np).to(device=query_tensor.device, dtype=torch.long)
        cand_idx = torch.from_numpy(cand_idx_np.reshape(-1)).to(device=product_tensor.device, dtype=torch.long)
        pos_mask = torch.from_numpy(pos_mask_np).to(device=query_tensor.device)

        q_vec = query_tensor.index_select(0, q_idx)  # [B, D]
        p_vec = product_tensor.index_select(0, cand_idx).reshape(bsz, k, -1)  # [B, K, D]
        q_expand = q_vec.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, -1)
        p_flat = p_vec.reshape(bsz * k, -1)

        q_extra_flat: Optional[torch.Tensor] = None
        if feature_cfg.use_query_features and query_extra_tensor is not None:
            q_extra = query_extra_tensor.index_select(0, q_idx)  # [B, F]
            q_extra_flat = q_extra.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, -1)

        feats = build_pair_features(
            query_vecs=q_expand,
            product_vecs=p_flat,
            cfg=feature_cfg,
            query_extra_vecs=q_extra_flat,
        )
        logits = model(feats).reshape(bsz, k)  # [B, K]

        valid_mask = pos_mask.any(dim=1) & (~pos_mask.all(dim=1))
        if not bool(valid_mask.any().item()):
            continue
        logits_v = logits[valid_mask]
        pos_mask_v = pos_mask[valid_mask]

        pos_logits = logits_v.masked_fill(~pos_mask_v, -1e9)
        numer = torch.logsumexp(pos_logits, dim=1)
        denom = torch.logsumexp(logits_v, dim=1)
        loss = -(numer - denom).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pos_prob = torch.exp(numer - denom).mean()
        loss_sum += float(loss.item())
        posprob_sum += float(pos_prob.item())
        count_batches += 1

    if count_batches == 0:
        return 0.0, 0.0
    return loss_sum / count_batches, posprob_sum / count_batches


@torch.no_grad()
def score_candidate_pairs(
    model: nn.Module,
    query_tensor: torch.Tensor,
    product_tensor: torch.Tensor,
    query_extra_tensor: Optional[torch.Tensor],
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

        q_extra: Optional[torch.Tensor] = None
        if feature_cfg.use_query_features and query_extra_tensor is not None:
            q_extra = query_extra_tensor.index_select(0, q_idx)

        features = build_pair_features(q_vec, p_vec, feature_cfg, query_extra_vecs=q_extra)
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
        default=Path("artifacts/rerank/topk_candidates.npz"),
        help="Cache for top-k candidate indices/scores.",
    )

    parser.add_argument("--query-id-key", type=str, default="bundle_ids,query_ids,ids")
    parser.add_argument("--query-emb-key", type=str, default="embeddings,query_embeddings,bundle_embeddings")
    parser.add_argument("--product-id-key", type=str, default="product_ids,pids,ids")
    parser.add_argument("--product-emb-key", type=str, default="embeddings,product_embeddings")
    parser.add_argument(
        "--query-extra-keys",
        type=str,
        default="query_features,box_features,extra_features",
        help="Optional query-side scalar features to append (from query embedding file).",
    )
    parser.add_argument(
        "--query-extra-names-keys",
        type=str,
        default="query_feature_names,box_feature_names",
    )
    parser.add_argument("--disable-query-extra", action="store_true")
    parser.add_argument("--disable-normalize", action="store_true")

    parser.add_argument("--topk", type=int, default=500, help="Candidate pool used for training/rerank.")
    parser.add_argument("--coverage-ks", type=str, default="200,500,1000")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64, help="Number of queries per optimizer step.")
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
    query_extra_keys = [k.strip() for k in args.query_extra_keys.split(",") if k.strip()]
    query_extra_names_keys = [k.strip() for k in args.query_extra_names_keys.split(",") if k.strip()]
    hidden_dims = parse_int_list(args.hidden_dims)
    eval_ks = sorted(set(parse_int_list(args.eval_ks)))
    coverage_ks = sorted(set(parse_int_list(args.coverage_ks)))

    feature_cfg = FeatureConfig(
        use_abs_diff=True,
        use_elem_product=True,
        use_sq_diff=bool(args.use_sq_diff),
        use_raw_concat=bool(args.use_raw_concat),
        use_query_features=False,
    )

    query_ids, query_embeddings, query_extra_features, query_extra_names = load_embeddings(
        path=args.query_embeddings,
        id_keys=query_id_keys,
        emb_keys=query_emb_keys,
        normalize=not args.disable_normalize,
        extra_keys=query_extra_keys,
        extra_names_keys=query_extra_names_keys,
    )
    product_ids, product_embeddings, _, _ = load_embeddings(
        path=args.product_embeddings,
        id_keys=product_id_keys,
        emb_keys=product_emb_keys,
        normalize=not args.disable_normalize,
        extra_keys=None,
        extra_names_keys=None,
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

    # Optional query-side scalar features (e.g., box geometry/conf stats).
    query_extra_mean: Optional[np.ndarray] = None
    query_extra_std: Optional[np.ndarray] = None
    if query_extra_features is not None and not args.disable_query_extra:
        train_idx = np.asarray([bundle_to_index[bid] for bid in train_ids], dtype=np.int64)
        query_extra_mean = query_extra_features[train_idx].mean(axis=0).astype(np.float32)
        query_extra_std = query_extra_features[train_idx].std(axis=0).astype(np.float32)
        query_extra_std = np.where(query_extra_std < 1e-6, 1.0, query_extra_std).astype(np.float32)
        query_extra_features = ((query_extra_features - query_extra_mean) / query_extra_std).astype(np.float32)
        feature_cfg.use_query_features = True
        if not query_extra_names:
            query_extra_names = [f"query_feature_{i}" for i in range(query_extra_features.shape[1])]
        print(f"Query extra features enabled: dim={query_extra_features.shape[1]}")
    else:
        query_extra_features = None
        query_extra_names = []
        print("Query extra features disabled.")

    coverage_max_k = max(max(coverage_ks), int(args.topk))
    topk_indices_full, topk_scores_full = load_or_compute_candidates(
        cache_path=args.candidate_cache,
        query_ids=query_ids,
        product_ids=product_ids,
        query_embeddings=query_embeddings,
        product_embeddings=product_embeddings,
        topk=coverage_max_k,
        device=device,
        query_batch_size=args.query_batch_size,
    )
    k_train = min(int(args.topk), topk_indices_full.shape[1])
    topk_indices = topk_indices_full[:, :k_train]
    topk_scores = topk_scores_full[:, :k_train]

    coverage_all = compute_positive_coverage(
        bundle_ids=eligible_bundle_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        topk_indices=topk_indices_full,
        ks=coverage_ks,
    )
    coverage_train = compute_positive_coverage(
        bundle_ids=train_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        topk_indices=topk_indices_full,
        ks=coverage_ks,
    )
    coverage_val = compute_positive_coverage(
        bundle_ids=val_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        topk_indices=topk_indices_full,
        ks=coverage_ks,
    )
    print(f"Coverage all : {coverage_to_string(coverage_all, coverage_ks)}")
    print(f"Coverage train: {coverage_to_string(coverage_train, coverage_ks)}")
    if val_ids:
        print(f"Coverage val : {coverage_to_string(coverage_val, coverage_ks)}")

    train_records = build_listwise_records(
        bundle_ids=train_ids,
        bundle_to_index=bundle_to_index,
        positives_map=positives_map,
        product_to_index=product_to_index,
        topk_indices=topk_indices,
    )
    if not train_records:
        raise RuntimeError(
            "No valid training records. Increase --topk and inspect coverage@K."
        )

    embedding_dim = query_embeddings.shape[1]
    query_extra_dim = int(query_extra_features.shape[1]) if query_extra_features is not None else 0
    input_dim = infer_feature_dim(embedding_dim, feature_cfg, query_extra_dim=query_extra_dim)
    model = MLPReranker(input_dim=input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    query_tensor = torch.from_numpy(query_embeddings).to(device)
    product_tensor = torch.from_numpy(product_embeddings).to(device)
    query_extra_tensor: Optional[torch.Tensor] = None
    if query_extra_features is not None:
        query_extra_tensor = torch.from_numpy(query_extra_features).to(device)
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
        train_loss, train_posprob = train_one_epoch_listwise(
            model=model,
            optimizer=optimizer,
            query_tensor=query_tensor,
            product_tensor=product_tensor,
            query_extra_tensor=query_extra_tensor,
            records=train_records,
            batch_size=args.batch_size,
            feature_cfg=feature_cfg,
            rng=rng,
        )

        if val_ids:
            rerank_scores = score_candidate_pairs(
                model=model,
                query_tensor=query_tensor,
                product_tensor=product_tensor,
                query_extra_tensor=query_extra_tensor,
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
            current_metric = train_posprob
            is_best = True

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_pos_prob": float(train_posprob),
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | pos_prob={train_posprob:.4f} | "
            + (metrics_to_string(val_metrics, eval_ks) if val_ids else "validation skipped")
            + (" | best" if is_best else "")
        )

    payload = {
        "model_state": best_state,
        "objective": "listwise_multi_positive_ce",
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "embedding_dim": embedding_dim,
        "feature_config": asdict(feature_cfg),
        "query_extra_dim": query_extra_dim,
        "query_extra_names": query_extra_names,
        "query_extra_mean": query_extra_mean.tolist() if query_extra_mean is not None else None,
        "query_extra_std": query_extra_std.tolist() if query_extra_std is not None else None,
        "blend_alpha": args.blend_alpha,
        "query_embeddings_path": str(args.query_embeddings),
        "product_embeddings_path": str(args.product_embeddings),
        "train_csv": str(args.train_csv),
        "topk": int(topk_indices.shape[1]),
        "coverage_ks": coverage_ks,
        "coverage_all": coverage_all,
        "coverage_train": coverage_train,
        "coverage_val": coverage_val,
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
