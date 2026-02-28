"""Full GR-Lite fine-tuning (adapter) for fashion bundle-product retrieval.

Strategy:
1) Use `srpone/gr-lite` as frozen feature extractor for bundle and product images.
2) Train a projection adapter on top of GR-Lite embeddings with multi-positive contrastive loss.
3) Save checkpoint to be consumed by `src/fashion_grlite_retrieval.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Using CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def open_image_safe(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return None


def build_image_map(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return {p.stem: p for p in image_dir.iterdir() if p.is_file()}


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1)


def batched(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def split_train_val_bundles(bundle_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    uniq = list(dict.fromkeys(bundle_ids))
    if val_ratio <= 0:
        return set(uniq), set()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(len(uniq) * val_ratio)))
    return set(uniq[val_n:]), set(uniq[:val_n])


def build_gt_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        out[str(row.bundle_asset_id)].add(str(row.product_asset_id))
    return out


def evaluate_predictions(
    pred_map: Dict[str, List[str]],
    gt_map: Dict[str, Set[str]],
    ks: Sequence[int],
) -> Dict[str, float]:
    eval_ids = [bid for bid in gt_map if bid in pred_map]
    if not eval_ids:
        return {}
    out: Dict[str, float] = {}
    for k in ks:
        hits: List[float] = []
        recalls: List[float] = []
        for bid in eval_ids:
            gt = gt_map[bid]
            if not gt:
                continue
            preds = pred_map[bid][:k]
            inter = len(set(preds) & gt)
            hits.append(1.0 if inter > 0 else 0.0)
            recalls.append(inter / float(len(gt)))
        if hits:
            out[f"hit@{k}"] = float(np.mean(hits))
        if recalls:
            out[f"recall@{k}"] = float(np.mean(recalls))
    return out


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


class GRLiteExtractor:
    def __init__(
        self,
        repo_id: str,
        checkpoint_name: str,
        feature_dim: int,
        device: torch.device,
        local_path: Optional[Path] = None,
    ) -> None:
        self.repo_id = repo_id
        self.checkpoint_name = checkpoint_name
        self.feature_dim = int(feature_dim)
        self.device = device

        if local_path is not None and local_path.exists():
            ckpt_path = local_path
        else:
            from huggingface_hub import hf_hub_download

            ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename=checkpoint_name))
        try:
            self.model = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            self.model = torch.load(ckpt_path, map_location=device)
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if not hasattr(self.model, "search"):
            raise RuntimeError("Loaded GR-Lite checkpoint does not provide `search` method.")
        print(f"Loaded GR-Lite from: {ckpt_path}")

    def _to_tensor(self, value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().float()
        elif isinstance(value, np.ndarray):
            t = torch.from_numpy(value).float()
        elif isinstance(value, list):
            t = torch.tensor(value, dtype=torch.float32)
        else:
            raise RuntimeError(f"Unsupported embedding value type: {type(value)}")
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        if not images:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)
        out = self.model.search(image_paths=images, feature_dim=self.feature_dim)
        vectors = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
        emb = l2_normalize(self._to_tensor(vectors))
        if not tta_flip:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        out2 = self.model.search(image_paths=flipped, feature_dim=self.feature_dim)
        vectors2 = out2[1] if isinstance(out2, (tuple, list)) and len(out2) >= 2 else out2
        emb2 = l2_normalize(self._to_tensor(vectors2))
        return l2_normalize((emb + emb2) * 0.5)

    def encode_ids(
        self,
        ids: Sequence[str],
        image_map: Dict[str, Path],
        batch_size: int,
        tta_flip: bool,
        cache_prefix: Optional[Path] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        valid_ids = [x for x in ids if x in image_map]

        if cache_prefix is not None:
            ids_path = cache_prefix.with_suffix(".ids.txt")
            emb_path = cache_prefix.with_suffix(".emb.npy")
            if ids_path.exists() and emb_path.exists():
                cached_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if cached_ids == valid_ids:
                    emb = np.load(emb_path)
                    print(f"Loaded cache: {emb_path}")
                    return cached_ids, torch.from_numpy(emb).float()

        out_ids: List[str] = []
        out_embs: List[torch.Tensor] = []
        total_batches = int(math.ceil(len(valid_ids) / float(batch_size))) if valid_ids else 0

        for chunk_ids in tqdm(
            batched(valid_ids, batch_size),
            total=total_batches,
            desc=f"Encoding {len(valid_ids)} images with GR-Lite",
        ):
            images: List[Image.Image] = []
            keep_ids: List[str] = []
            for asset_id in chunk_ids:
                image = open_image_safe(image_map[asset_id])
                if image is None:
                    continue
                images.append(image)
                keep_ids.append(asset_id)
            if not images:
                continue

            emb = self.encode_images(images=images, tta_flip=tta_flip)
            if emb.ndim != 2 or emb.shape[0] != len(keep_ids):
                raise RuntimeError(
                    f"GR-Lite embedding mismatch: got {tuple(emb.shape)} expected ({len(keep_ids)}, dim)."
                )
            out_ids.extend(keep_ids)
            out_embs.append(emb.cpu())

        if not out_embs:
            return [], torch.zeros((0, self.feature_dim), dtype=torch.float32)

        all_embs = torch.cat(out_embs, dim=0).float()
        if cache_prefix is not None:
            cache_prefix.parent.mkdir(parents=True, exist_ok=True)
            ids_path = cache_prefix.with_suffix(".ids.txt")
            emb_path = cache_prefix.with_suffix(".emb.npy")
            ids_path.write_text("\n".join(out_ids) + "\n", encoding="utf-8")
            np.save(emb_path, all_embs.numpy())
            print(f"Saved cache: {emb_path}")
        return out_ids, all_embs


class SharedAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.norm = nn.LayerNorm(self.in_dim)
        self.skip = nn.Linear(self.in_dim, self.out_dim, bias=False) if self.in_dim != self.out_dim else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        base = self.skip(xn)
        delta = self.mlp(xn)
        return l2_normalize(base + delta)


class PairIndexDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str]], bundle_to_idx: Dict[str, int], product_to_idx: Dict[str, int]) -> None:
        self.samples: List[Tuple[int, int, str, str]] = []
        for bundle_id, product_id in pairs:
            b_idx = bundle_to_idx.get(bundle_id)
            p_idx = product_to_idx.get(product_id)
            if b_idx is None or p_idx is None:
                continue
            self.samples.append((b_idx, p_idx, bundle_id, product_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, int, str, str]:
        return self.samples[idx]


def make_positive_masks(
    batch_bundle_ids: Sequence[str],
    batch_product_ids: Sequence[str],
    bundle_to_products: Dict[str, Set[str]],
    product_to_bundles: Dict[str, Set[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = len(batch_bundle_ids)
    pos_bp = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)
    pos_pb = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)

    for i, bundle_id in enumerate(batch_bundle_ids):
        positives = bundle_to_products.get(bundle_id, set())
        for j, product_id in enumerate(batch_product_ids):
            if product_id in positives:
                pos_bp[i, j] = True

    for i, product_id in enumerate(batch_product_ids):
        positives = product_to_bundles.get(product_id, set())
        for j, bundle_id in enumerate(batch_bundle_ids):
            if bundle_id in positives:
                pos_pb[i, j] = True

    diag = torch.arange(bsz, device=device)
    pos_bp[diag, diag] = True
    pos_pb[diag, diag] = True
    return pos_bp, pos_pb


def multi_positive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_logprob = torch.logsumexp(
        torch.where(positive_mask, log_probs, torch.full_like(log_probs, -1e9)),
        dim=1,
    )
    return (-pos_logprob).mean()


@torch.inference_mode()
def project_embeddings(
    base_emb: torch.Tensor,
    adapter: SharedAdapter,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if base_emb.numel() == 0:
        return base_emb
    adapter.eval()
    out_chunks: List[torch.Tensor] = []
    total = int(math.ceil(base_emb.shape[0] / float(batch_size)))
    for i in tqdm(range(total), desc=f"Projecting {base_emb.shape[0]} embeddings", leave=False):
        s = i * batch_size
        e = min((i + 1) * batch_size, base_emb.shape[0])
        chunk = base_emb[s:e].to(device, non_blocking=True)
        out = adapter(chunk).cpu()
        out_chunks.append(out)
    return torch.cat(out_chunks, dim=0)


def rank_retrieval(
    query_ids: Sequence[str],
    query_emb: torch.Tensor,
    product_ids: Sequence[str],
    product_emb: torch.Tensor,
    top_k: int,
    fallback_products: Sequence[str],
) -> Dict[str, List[str]]:
    if query_emb.numel() == 0 or product_emb.numel() == 0:
        return {bid: list(fallback_products[:top_k]) for bid in query_ids}
    scores = query_emb @ product_emb.T
    kk = min(top_k, scores.shape[1])
    top_idx = torch.topk(scores, k=kk, dim=1).indices.cpu().tolist()

    out: Dict[str, List[str]] = {}
    for bid, idxs in zip(query_ids, top_idx):
        preds = [product_ids[i] for i in idxs]
        preds = unique_keep_order(preds)
        if len(preds) < top_k:
            for pid in fallback_products:
                if pid in preds:
                    continue
                preds.append(pid)
                if len(preds) >= top_k:
                    break
        out[bid] = preds[:top_k]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full GR-Lite fine-tuning via adapter learning.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--bundle-images-dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product-images-dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grlite_finetune"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/grlite_cache"))
    parser.add_argument("--checkpoint-name", type=str, default="best_grlite_adapter.pt")
    parser.add_argument("--history-name", type=str, default="train_history_grlite.json")

    parser.add_argument("--grlite-repo", type=str, default="srpone/gr-lite")
    parser.add_argument("--grlite-checkpoint", type=str, default="gr_lite.pt")
    parser.add_argument("--grlite-dim", type=int, default=1024)
    parser.add_argument("--grlite-local-path", type=Path, default=None)
    parser.set_defaults(tta_flip=True, eval_all_products=True)
    parser.add_argument("--tta-flip", action="store_true", dest="tta_flip")
    parser.add_argument("--no-tta-flip", action="store_false", dest="tta_flip")
    parser.add_argument("--eval-all-products", action="store_true", dest="eval_all_products")
    parser.add_argument("--eval-train-products-only", action="store_false", dest="eval_all_products")

    parser.add_argument("--adapter-hidden-dim", type=int, default=1536)
    parser.add_argument("--adapter-out-dim", type=int, default=1024)
    parser.add_argument("--adapter-dropout", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--project-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-every", type=int, default=1)

    parser.add_argument("--max-train-pairs", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-train-bundles", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-val-bundles", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-train-products", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-eval-products", type=int, default=0, help="Debug only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df = pd.read_csv(args.train_csv)
    train_df["bundle_asset_id"] = train_df["bundle_asset_id"].astype(str)
    train_df["product_asset_id"] = train_df["product_asset_id"].astype(str)

    bundle_image_map = build_image_map(args.bundle_images_dir)
    product_image_map = build_image_map(args.product_images_dir)

    valid_mask = train_df["bundle_asset_id"].isin(bundle_image_map) & train_df["product_asset_id"].isin(product_image_map)
    train_df = train_df.loc[valid_mask].reset_index(drop=True)
    if len(train_df) == 0:
        raise RuntimeError("No valid training pairs after filtering missing images.")

    all_train_bundle_ids = train_df["bundle_asset_id"].drop_duplicates().tolist()
    all_train_product_ids = train_df["product_asset_id"].drop_duplicates().tolist()

    if args.max_train_bundles > 0:
        keep = set(all_train_bundle_ids[: args.max_train_bundles])
        train_df = train_df[train_df["bundle_asset_id"].isin(keep)].reset_index(drop=True)
        all_train_bundle_ids = train_df["bundle_asset_id"].drop_duplicates().tolist()
    if args.max_train_products > 0:
        keep = set(all_train_product_ids[: args.max_train_products])
        train_df = train_df[train_df["product_asset_id"].isin(keep)].reset_index(drop=True)
        all_train_product_ids = train_df["product_asset_id"].drop_duplicates().tolist()

    train_bundle_set, val_bundle_set = split_train_val_bundles(all_train_bundle_ids, args.val_ratio, args.seed)
    train_pairs_df = train_df[train_df["bundle_asset_id"].isin(train_bundle_set)].reset_index(drop=True)
    val_pairs_df = train_df[train_df["bundle_asset_id"].isin(val_bundle_set)].reset_index(drop=True)

    if args.max_train_pairs > 0:
        train_pairs_df = train_pairs_df.iloc[: args.max_train_pairs].copy()
    if args.max_val_bundles > 0:
        keep_val = set(list(dict.fromkeys(val_pairs_df["bundle_asset_id"].tolist()))[: args.max_val_bundles])
        val_pairs_df = val_pairs_df[val_pairs_df["bundle_asset_id"].isin(keep_val)].copy()

    gt_train = build_gt_map(train_pairs_df)
    gt_val = build_gt_map(val_pairs_df)
    product_to_bundles_train: Dict[str, Set[str]] = defaultdict(set)
    for row in train_pairs_df.itertuples(index=False):
        product_to_bundles_train[str(row.product_asset_id)].add(str(row.bundle_asset_id))

    train_pairs = list(
        zip(train_pairs_df["bundle_asset_id"].astype(str).tolist(), train_pairs_df["product_asset_id"].astype(str).tolist())
    )
    val_bundle_ids = list(dict.fromkeys(val_pairs_df["bundle_asset_id"].astype(str).tolist()))

    print(f"Train pairs: {len(train_pairs_df)} | Val pairs: {len(val_pairs_df)}")
    print(f"Train bundles: {train_pairs_df['bundle_asset_id'].nunique()} | Val bundles: {val_pairs_df['bundle_asset_id'].nunique()}")

    extractor = GRLiteExtractor(
        repo_id=args.grlite_repo,
        checkpoint_name=args.grlite_checkpoint,
        feature_dim=args.grlite_dim,
        device=device,
        local_path=args.grlite_local_path,
    )

    bundle_cache = args.cache_dir / "train_bundles_grlite"
    train_product_cache = args.cache_dir / "train_products_grlite"
    eval_product_cache = args.cache_dir / "eval_products_grlite"

    train_bundle_ids_enc, train_bundle_base = extractor.encode_ids(
        ids=all_train_bundle_ids,
        image_map=bundle_image_map,
        batch_size=args.embed_batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=bundle_cache,
    )
    train_product_ids_enc, train_product_base = extractor.encode_ids(
        ids=all_train_product_ids,
        image_map=product_image_map,
        batch_size=args.embed_batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=train_product_cache,
    )

    if args.eval_all_products:
        eval_product_ids = sorted(product_image_map.keys())
    else:
        eval_product_ids = list(all_train_product_ids)
    if args.max_eval_products > 0:
        eval_product_ids = eval_product_ids[: args.max_eval_products]

    eval_product_ids_enc, eval_product_base = extractor.encode_ids(
        ids=eval_product_ids,
        image_map=product_image_map,
        batch_size=args.embed_batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=eval_product_cache,
    )

    bundle_to_idx = {bid: i for i, bid in enumerate(train_bundle_ids_enc)}
    product_to_idx_train = {pid: i for i, pid in enumerate(train_product_ids_enc)}

    train_ds = PairIndexDataset(train_pairs, bundle_to_idx=bundle_to_idx, product_to_idx=product_to_idx_train)
    if len(train_ds) == 0:
        raise RuntimeError("No train pairs left after GR-Lite embedding filtering.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    adapter = SharedAdapter(
        in_dim=args.grlite_dim,
        hidden_dim=args.adapter_hidden_dim,
        out_dim=args.adapter_out_dim,
        dropout=args.adapter_dropout,
    ).to(device)
    logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / max(args.temperature, 1e-6)), dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + [logit_scale],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.output_dir / args.checkpoint_name
    history_path = args.output_dir / args.history_name
    history: List[Dict[str, object]] = []
    best_recall15 = -1.0

    train_bundle_base = l2_normalize(train_bundle_base).float().cpu()
    train_product_base = l2_normalize(train_product_base).float().cpu()
    eval_product_base = l2_normalize(eval_product_base).float().cpu()
    fallback_products = [pid for pid, _ in Counter(train_df["product_asset_id"]).most_common(100)]
    fallback_products = unique_keep_order(fallback_products + eval_product_ids_enc)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for b_idx, p_idx, b_ids, p_ids in pbar:
            b_vec = train_bundle_base[b_idx].to(device, non_blocking=True)
            p_vec = train_product_base[p_idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            q = adapter(b_vec)
            g = adapter(p_vec)
            scale = torch.clamp(logit_scale.exp(), min=1.0, max=100.0)
            logits = scale * (q @ g.T)

            pos_bp, pos_pb = make_positive_masks(
                batch_bundle_ids=list(b_ids),
                batch_product_ids=list(p_ids),
                bundle_to_products=gt_train,
                product_to_bundles=product_to_bundles_train,
                device=device,
            )
            loss_bp = multi_positive_nce(logits, pos_bp)
            loss_pb = multi_positive_nce(logits.T, pos_pb)
            loss = 0.5 * (loss_bp + loss_pb)
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + [logit_scale], args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{total_loss / max(steps, 1):.4f}", scale=f"{float(scale.item()):.2f}")

        epoch_record: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "logit_scale": float(torch.clamp(logit_scale.exp(), min=1.0, max=100.0).item()),
        }

        if args.eval_every > 0 and epoch % args.eval_every == 0 and len(val_bundle_ids) > 0:
            adapter.eval()
            val_bundle_ids_enc = [bid for bid in val_bundle_ids if bid in bundle_to_idx]
            val_idx = [bundle_to_idx[bid] for bid in val_bundle_ids_enc]
            val_base = train_bundle_base[val_idx]

            val_proj = project_embeddings(
                base_emb=val_base,
                adapter=adapter,
                device=device,
                batch_size=args.project_batch_size,
            )
            eval_proj = project_embeddings(
                base_emb=eval_product_base,
                adapter=adapter,
                device=device,
                batch_size=args.project_batch_size,
            )

            val_pred = rank_retrieval(
                query_ids=val_bundle_ids_enc,
                query_emb=val_proj,
                product_ids=eval_product_ids_enc,
                product_emb=eval_proj,
                top_k=15,
                fallback_products=fallback_products,
            )
            val_metrics = evaluate_predictions(val_pred, gt_val, ks=(5, 10, 15))
            epoch_record["val_metrics"] = val_metrics
            metric = float(val_metrics.get("recall@15", 0.0))
            print(f"Epoch {epoch} val: {val_metrics}")

            if metric > best_recall15:
                best_recall15 = metric
                payload = {
                    "backbone": "grlite_adapter",
                    "grlite_repo": args.grlite_repo,
                    "grlite_checkpoint": args.grlite_checkpoint,
                    "grlite_dim": int(args.grlite_dim),
                    "adapter_state_dict": adapter.state_dict(),
                    "adapter_in_dim": int(args.grlite_dim),
                    "adapter_hidden_dim": int(args.adapter_hidden_dim),
                    "adapter_out_dim": int(args.adapter_out_dim),
                    "adapter_dropout": float(args.adapter_dropout),
                    "temperature": float(args.temperature),
                    "logit_scale": float(torch.clamp(logit_scale.exp(), min=1.0, max=100.0).item()),
                    "best_recall15": float(best_recall15),
                    "epoch": int(epoch),
                    "train_args": vars(args),
                }
                torch.save(payload, best_ckpt_path)
                print(f"Saved new best checkpoint: {best_ckpt_path}")

        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training complete. Best recall@15={best_recall15:.6f}")
    print(f"Checkpoint: {best_ckpt_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
