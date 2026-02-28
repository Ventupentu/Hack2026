"""Fine-tuning for fashion bundle -> product retrieval.

This script trains a dual-image retrieval encoder with contrastive learning:
- Query side: bundle images
- Gallery side: product images

Supported backbones:
- `clip` (recommended): Hugging Face CLIP image encoder
- `torchvision_resnet50` (offline-friendly fallback)

Outputs:
- best checkpoint for inference (`state_dict` + preprocessing metadata)
- JSON history with train/val metrics
"""

from __future__ import annotations

import argparse
import json
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


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1)


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


def split_train_val_bundles(bundle_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    uniq = list(dict.fromkeys(bundle_ids))
    if val_ratio <= 0:
        return set(uniq), set()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(len(uniq) * val_ratio)))
    val_ids = set(uniq[:val_n])
    train_ids = set(uniq[val_n:])
    return train_ids, val_ids


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


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        bundle_image_map: Dict[str, Path],
        product_image_map: Dict[str, Path],
        bundle_transform,
        product_transform,
    ) -> None:
        self.pairs = list(pairs)
        self.bundle_image_map = bundle_image_map
        self.product_image_map = product_image_map
        self.bundle_transform = bundle_transform
        self.product_transform = product_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Optional[Dict[str, object]]:
        bundle_id, product_id = self.pairs[idx]
        b_path = self.bundle_image_map.get(bundle_id)
        p_path = self.product_image_map.get(product_id)
        if b_path is None or p_path is None:
            return None

        b_img = open_image_safe(b_path)
        p_img = open_image_safe(p_path)
        if b_img is None or p_img is None:
            return None

        return {
            "bundle_id": bundle_id,
            "product_id": product_id,
            "bundle_tensor": self.bundle_transform(b_img),
            "product_tensor": self.product_transform(p_img),
        }


def collate_skip_none(batch: Sequence[Optional[Dict[str, object]]]) -> Optional[Dict[str, object]]:
    valid = [x for x in batch if x is not None]
    if not valid:
        return None
    return {
        "bundle_id": [x["bundle_id"] for x in valid],
        "product_id": [x["product_id"] for x in valid],
        "bundle_tensor": torch.stack([x["bundle_tensor"] for x in valid], dim=0),
        "product_tensor": torch.stack([x["product_tensor"] for x in valid], dim=0),
    }


def build_transform(image_size: int, image_mean: Sequence[float], image_std: Sequence[float], is_train: bool):
    from torchvision import transforms

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )


class RetrievalModelBase(nn.Module):
    image_size: int
    image_mean: Sequence[float]
    image_std: Sequence[float]
    embed_dim: int
    backbone_name: str

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ClipRetrievalModel(RetrievalModelBase):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import CLIPModel

        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.model_name = model_name
        self.backbone_name = "clip"
        self.embed_dim = int(self.clip_model.config.projection_dim)

        vision_cfg = self.clip_model.config.vision_config
        self.image_size = int(getattr(vision_cfg, "image_size", 224))
        # OpenAI CLIP normalization constants.
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_image_features(pixel_values=pixel_values)


class TorchvisionRetrievalModel(RetrievalModelBase):
    def __init__(self, model_name: str, embed_dim: int) -> None:
        super().__init__()
        from torchvision import models

        if model_name != "resnet50":
            raise ValueError(f"Unsupported torchvision model: {model_name}")
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
        except Exception:
            weights = None
            backbone = models.resnet50(weights=None)

        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(in_features, embed_dim)
        self.model_name = model_name
        self.backbone_name = "torchvision_resnet50"
        self.embed_dim = int(embed_dim)
        self.image_size = 224
        if weights is not None:
            self.image_mean = list(weights.transforms().mean)
            self.image_std = list(weights.transforms().std)
        else:
            self.image_mean = [0.485, 0.456, 0.406]
            self.image_std = [0.229, 0.224, 0.225]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        return self.projection(feats)


def create_model(backbone: str, model_name: str, tv_embed_dim: int) -> RetrievalModelBase:
    if backbone == "clip":
        return ClipRetrievalModel(model_name=model_name)
    if backbone == "torchvision_resnet50":
        return TorchvisionRetrievalModel(model_name=model_name, embed_dim=tv_embed_dim)
    raise ValueError(f"Unsupported backbone: {backbone}")


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
        pos_products = bundle_to_products.get(bundle_id, set())
        for j, product_id in enumerate(batch_product_ids):
            if product_id in pos_products:
                pos_bp[i, j] = True
    for i, product_id in enumerate(batch_product_ids):
        pos_bundles = product_to_bundles.get(product_id, set())
        for j, bundle_id in enumerate(batch_bundle_ids):
            if bundle_id in pos_bundles:
                pos_pb[i, j] = True

    diag = torch.arange(bsz, device=device)
    pos_bp[diag, diag] = True
    pos_pb[diag, diag] = True
    return pos_bp, pos_pb


def multi_positive_nce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    # Stable log-softmax based formulation for multi-positive contrastive loss.
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # At least one positive per row by construction.
    pos_logprob = torch.logsumexp(
        torch.where(positive_mask, log_probs, torch.full_like(log_probs, -1e9)),
        dim=1,
    )
    return (-pos_logprob).mean()


@torch.inference_mode()
def encode_assets(
    ids: Sequence[str],
    image_map: Dict[str, Path],
    transform,
    model: RetrievalModelBase,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
) -> Tuple[List[str], torch.Tensor]:
    class _DS(Dataset):
        def __init__(self, _ids: Sequence[str], _map: Dict[str, Path], _transform) -> None:
            self.ids = [x for x in _ids if x in _map]
            self.map = _map
            self.transform = _transform

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, idx: int) -> Optional[Tuple[str, torch.Tensor]]:
            asset_id = self.ids[idx]
            img = open_image_safe(self.map[asset_id])
            if img is None:
                return None
            return asset_id, self.transform(img)

    def _collate(batch):
        valid = [x for x in batch if x is not None]
        if not valid:
            return None
        return [x[0] for x in valid], torch.stack([x[1] for x in valid], dim=0)

    ds = _DS(ids, image_map, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    out_ids: List[str] = []
    out_embs: List[torch.Tensor] = []

    for pack in tqdm(loader, desc=f"Encoding {len(ds)} assets", leave=False):
        if pack is None:
            continue
        batch_ids, batch_tensors = pack
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            emb = model.encode_image(batch_tensors)
        emb = l2_normalize(emb).cpu()
        out_ids.extend(batch_ids)
        out_embs.append(emb)

    if not out_embs:
        return [], torch.zeros((0, model.embed_dim), dtype=torch.float32)
    return out_ids, torch.cat(out_embs, dim=0)


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

    query_emb = l2_normalize(query_emb)
    product_emb = l2_normalize(product_emb)
    scores = query_emb @ product_emb.T  # [Q, P]
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
    parser = argparse.ArgumentParser(description="Fine-tune retrieval model for fashion bundle-product matching.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--bundle-images-dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product-images-dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--checkpoint-name", type=str, default="best_finetuned.pt")
    parser.add_argument("--history-name", type=str, default="train_history.json")

    parser.add_argument("--backbone", type=str, default="clip", choices=["clip", "torchvision_resnet50"])
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--tv-embed-dim", type=int, default=768)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--top-k-eval", type=int, default=15)
    parser.add_argument("--freeze-backbone", action="store_true", default=False)
    parser.set_defaults(use_amp=True)
    parser.add_argument("--use-amp", action="store_true", dest="use_amp")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")

    parser.add_argument("--max-train-pairs", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-val-bundles", type=int, default=0, help="Debug only.")
    parser.add_argument("--max-products", type=int, default=0, help="Debug only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = args.use_amp and device.type == "cuda"

    train_df = pd.read_csv(args.train_csv)
    train_df["bundle_asset_id"] = train_df["bundle_asset_id"].astype(str)
    train_df["product_asset_id"] = train_df["product_asset_id"].astype(str)

    bundle_image_map = build_image_map(args.bundle_images_dir)
    product_image_map = build_image_map(args.product_images_dir)

    # Filter invalid pairs (missing image either side).
    valid_mask = train_df["bundle_asset_id"].isin(bundle_image_map) & train_df["product_asset_id"].isin(product_image_map)
    train_df = train_df.loc[valid_mask].reset_index(drop=True)
    if len(train_df) == 0:
        raise RuntimeError("No valid train pairs after image filtering.")

    bundle_ids_all = train_df["bundle_asset_id"].drop_duplicates().tolist()
    train_bundle_set, val_bundle_set = split_train_val_bundles(bundle_ids_all, args.val_ratio, args.seed)
    train_pairs_df = train_df[train_df["bundle_asset_id"].isin(train_bundle_set)].reset_index(drop=True)
    val_pairs_df = train_df[train_df["bundle_asset_id"].isin(val_bundle_set)].reset_index(drop=True)

    if args.max_train_pairs > 0:
        train_pairs_df = train_pairs_df.iloc[: args.max_train_pairs].copy()
    if args.max_val_bundles > 0:
        keep_val = set(list(dict.fromkeys(val_pairs_df["bundle_asset_id"].tolist()))[: args.max_val_bundles])
        val_pairs_df = val_pairs_df[val_pairs_df["bundle_asset_id"].isin(keep_val)].copy()

    print(f"Train pairs: {len(train_pairs_df)} | Val pairs: {len(val_pairs_df)}")
    print(f"Train bundles: {train_pairs_df['bundle_asset_id'].nunique()} | Val bundles: {val_pairs_df['bundle_asset_id'].nunique()}")

    bundle_to_products = build_gt_map(train_df)
    product_to_bundles: Dict[str, Set[str]] = defaultdict(set)
    for row in train_df.itertuples(index=False):
        product_to_bundles[str(row.product_asset_id)].add(str(row.bundle_asset_id))

    model = create_model(backbone=args.backbone, model_name=args.model_name, tv_embed_dim=args.tv_embed_dim).to(device)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Backbone frozen (training projection layers only).")

    bundle_transform_train = build_transform(model.image_size, model.image_mean, model.image_std, is_train=True)
    product_transform_train = build_transform(model.image_size, model.image_mean, model.image_std, is_train=True)
    eval_transform = build_transform(model.image_size, model.image_mean, model.image_std, is_train=False)

    train_pairs: List[Tuple[str, str]] = list(
        zip(train_pairs_df["bundle_asset_id"].astype(str), train_pairs_df["product_asset_id"].astype(str))
    )
    val_gt_map = build_gt_map(val_pairs_df)
    val_bundle_ids = list(dict.fromkeys(val_pairs_df["bundle_asset_id"].astype(str).tolist()))

    # Gallery is all product images (best match for final inference scenario).
    product_ids_gallery = sorted(product_image_map.keys())
    if args.max_products > 0:
        product_ids_gallery = product_ids_gallery[: args.max_products]
    fallback_products = [pid for pid, _ in Counter(train_df["product_asset_id"]).most_common(100)]
    fallback_products = unique_keep_order(fallback_products + product_ids_gallery)

    train_ds = PairDataset(
        pairs=train_pairs,
        bundle_image_map=bundle_image_map,
        product_image_map=product_image_map,
        bundle_transform=bundle_transform_train,
        product_transform=product_transform_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.GradScaler(enabled=amp_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.output_dir / args.checkpoint_name
    history_path = args.output_dir / args.history_name

    history: List[Dict[str, object]] = []
    best_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            if batch is None:
                continue

            bundle_ids_batch = batch["bundle_id"]
            product_ids_batch = batch["product_id"]
            bundle_tensor = batch["bundle_tensor"].to(device, non_blocking=True)
            product_tensor = batch["product_tensor"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                emb_b = l2_normalize(model.encode_image(bundle_tensor))
                emb_p = l2_normalize(model.encode_image(product_tensor))
                logits = (emb_b @ emb_p.T) / args.temperature
                pos_bp, pos_pb = make_positive_masks(
                    batch_bundle_ids=bundle_ids_batch,
                    batch_product_ids=product_ids_batch,
                    bundle_to_products=bundle_to_products,
                    product_to_bundles=product_to_bundles,
                    device=device,
                )
                loss_bp = multi_positive_nce(logits, pos_bp)
                loss_pb = multi_positive_nce(logits.T, pos_pb)
                loss = 0.5 * (loss_bp + loss_pb)

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{total_loss / max(steps, 1):.4f}")

        train_loss = total_loss / max(steps, 1)
        epoch_record: Dict[str, object] = {"epoch": epoch, "train_loss": train_loss}

        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            model.eval()
            product_ids_enc, product_emb = encode_assets(
                ids=product_ids_gallery,
                image_map=product_image_map,
                transform=eval_transform,
                model=model,
                device=device,
                batch_size=max(1, args.batch_size),
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )
            val_bundle_ids_enc, val_bundle_emb = encode_assets(
                ids=val_bundle_ids,
                image_map=bundle_image_map,
                transform=eval_transform,
                model=model,
                device=device,
                batch_size=max(1, args.batch_size),
                num_workers=args.num_workers,
                amp_enabled=amp_enabled,
            )

            val_pred = rank_retrieval(
                query_ids=val_bundle_ids_enc,
                query_emb=val_bundle_emb,
                product_ids=product_ids_enc,
                product_emb=product_emb,
                top_k=min(args.top_k_eval, 15),
                fallback_products=fallback_products,
            )
            val_metrics = evaluate_predictions(val_pred, val_gt_map, ks=(5, 10, 15))
            epoch_record["val_metrics"] = val_metrics
            metric_for_best = float(val_metrics.get("recall@15", 0.0))
            print(f"Epoch {epoch} val: {val_metrics}")

            if metric_for_best > best_metric:
                best_metric = metric_for_best
                ckpt_payload = {
                    "backbone": model.backbone_name,
                    "model_name": model.model_name,
                    "state_dict": model.state_dict(),
                    "image_size": int(model.image_size),
                    "image_mean": list(model.image_mean),
                    "image_std": list(model.image_std),
                    "embed_dim": int(model.embed_dim),
                    "best_recall15": float(best_metric),
                    "epoch": int(epoch),
                    "train_args": vars(args),
                }
                if model.backbone_name == "clip":
                    clip_cfg = getattr(model, "clip_model").config.to_dict()
                    ckpt_payload["clip_config"] = clip_cfg
                torch.save(ckpt_payload, best_ckpt_path)
                print(f"Saved new best checkpoint to: {best_ckpt_path}")

        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training complete. Best recall@15: {best_metric:.6f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
