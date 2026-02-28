"""OpenCLIP retrieval training implementation (bundle -> products)."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import InditexConfig


class OpenCLIPImageEncoder(nn.Module):
    """Wrapper exposing image-only forward for DataParallel."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_dir(path: Path) -> None:
    """Create folder if needed."""
    path.mkdir(parents=True, exist_ok=True)


def read_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    """Read jsonl or csv manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest extension: {path.suffix}. Use .jsonl or .csv")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _parse_id_list(value: Any) -> List[str]:
    """Parse list-like product ids from str/list/json."""
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_str(v) for v in value if _as_str(v)]
    text = _as_str(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [_as_str(v) for v in parsed if _as_str(v)]
        except json.JSONDecodeError:
            pass
    for sep in ("|", ",", ";", " "):
        if sep in text:
            parts = [_as_str(x) for x in text.split(sep)]
            cleaned = [x for x in parts if x]
            if cleaned:
                return cleaned
    return [text]


def _first_existing_key(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and _as_str(row.get(key)):
            return _as_str(row.get(key))
    return ""


def _default_bundle_path(bundle_id: str, bundles_images_dir: Path) -> Path:
    return (bundles_images_dir / f"{bundle_id}.jpg").resolve()


def _default_product_path(product_id: str, products_images_dir: Path) -> Path:
    return (products_images_dir / f"{product_id}.jpg").resolve()


def parse_products_manifest(path: Path, products_images_dir: Path) -> Dict[str, Path]:
    """Return product_id -> image_path map."""
    rows = read_manifest_rows(path)
    product_to_image: Dict[str, Path] = {}
    for row in rows:
        pid = _first_existing_key(row, ("product_asset_id", "product_id", "asset_id", "id"))
        if not pid:
            continue
        image_path = _first_existing_key(
            row,
            ("image_path", "product_image_path", "path", "local_image_path"),
        )
        if not image_path:
            image_path = str(_default_product_path(pid, products_images_dir))
        product_to_image[pid] = Path(image_path).expanduser().resolve()
    if not product_to_image:
        raise RuntimeError("No product entries found in products_manifest.")
    return product_to_image


def parse_bundle_manifest(
    path: Path,
    product_to_image: Dict[str, Path],
    bundles_images_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, Set[str]]]:
    """Parse train/val manifest into bundle_image and positives mapping."""
    rows = read_manifest_rows(path)
    bundle_to_image: Dict[str, Path] = {}
    bundle_to_products: Dict[str, Set[str]] = defaultdict(set)

    for row in rows:
        bundle_id = _first_existing_key(row, ("bundle_asset_id", "bundle_id", "query_id", "id"))
        if not bundle_id:
            continue

        bundle_img = _first_existing_key(
            row,
            ("bundle_image_path", "image_path", "query_image_path", "path"),
        )
        bundle_to_image[bundle_id] = (
            Path(bundle_img).expanduser().resolve()
            if bundle_img
            else _default_bundle_path(bundle_id, bundles_images_dir)
        )

        direct_pid = _first_existing_key(row, ("product_asset_id", "product_id", "candidate_id"))
        if direct_pid:
            bundle_to_products[bundle_id].add(direct_pid)

        for key in ("product_asset_ids", "product_ids", "positives", "positive_product_ids"):
            if key in row:
                for pid in _parse_id_list(row.get(key)):
                    bundle_to_products[bundle_id].add(pid)

    filtered: Dict[str, Set[str]] = {}
    dropped = 0
    for bid, pids in bundle_to_products.items():
        keep = {pid for pid in pids if pid in product_to_image}
        dropped += len(pids) - len(keep)
        if keep:
            filtered[bid] = keep
    if dropped > 0:
        print(f"Warning: dropped {dropped} bundle-product links missing in products_manifest.")

    bundle_to_image = {bid: bundle_to_image[bid] for bid in filtered.keys() if bid in bundle_to_image}
    if not filtered:
        raise RuntimeError(f"No valid bundle->product links found in manifest: {path}")
    return bundle_to_image, filtered


def open_image_safe(path: Path, retries: int = 1) -> Optional[Image.Image]:
    """Safely open an image with small retry count."""
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            last_err = exc
    print(f"Warning: failed to read image {path} ({last_err})")
    return None


class BundlePositiveDataset(Dataset):
    """One sample per bundle, choosing one positive product on the fly."""

    def __init__(
        self,
        bundle_to_image: Dict[str, Path],
        bundle_to_products: Dict[str, Set[str]],
        product_to_image: Dict[str, Path],
        bundle_transform: Callable[[Image.Image], torch.Tensor],
        product_transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.bundle_ids = sorted(bundle_to_products.keys())
        self.bundle_to_image = bundle_to_image
        self.bundle_to_products = {k: sorted(v) for k, v in bundle_to_products.items()}
        self.product_to_image = product_to_image
        self.bundle_transform = bundle_transform
        self.product_transform = product_transform

    def __len__(self) -> int:
        return len(self.bundle_ids)

    def __getitem__(self, idx: int):
        bundle_id = self.bundle_ids[idx]
        product_ids = self.bundle_to_products[bundle_id]
        product_id = random.choice(product_ids)
        bundle_img = open_image_safe(self.bundle_to_image[bundle_id])
        product_img = open_image_safe(self.product_to_image[product_id])
        if bundle_img is None or product_img is None:
            return None
        return {
            "bundle_id": bundle_id,
            "product_id": product_id,
            "bundle_img": self.bundle_transform(bundle_img),
            "product_img": self.product_transform(product_img),
        }


class AssetImageDataset(Dataset):
    """Simple asset image dataset for encoding."""

    def __init__(self, ids: Sequence[str], id_to_path: Dict[str, Path], transform) -> None:
        self.ids = list(ids)
        self.id_to_path = id_to_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        asset_id = self.ids[idx]
        img = open_image_safe(self.id_to_path[asset_id])
        if img is None:
            return None
        return {"id": asset_id, "img": self.transform(img)}


def collate_skip_none(batch: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Collate function that skips unreadable samples."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
    """Cosine scheduler with 10% warmup."""
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def encode_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    """Encode dataset images into normalized embeddings."""
    all_ids: List[str] = []
    all_embs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if batch is None:
                continue
            imgs = batch["img"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                feats = model(imgs)
            feats = F.normalize(feats.float(), p=2, dim=1)
            all_ids.extend(batch["id"])
            all_embs.append(feats)

    if not all_embs:
        return [], torch.empty((0, 0), dtype=torch.float32, device=device)
    return all_ids, torch.cat(all_embs, dim=0)


def validate_retrieval(
    model: nn.Module,
    preprocess_val,
    device: torch.device,
    amp: bool,
    val_bundle_to_image: Dict[str, Path],
    val_bundle_to_products: Dict[str, Set[str]],
    product_to_image: Dict[str, Path],
    batch_size: int,
    num_workers: int,
    max_val_k: int,
    recall_k: int,
) -> Tuple[float, Dict[str, int]]:
    """Compute Recall@K for bundle->product retrieval."""
    if recall_k > max_val_k:
        raise ValueError("--recall_k must be <= --max_val_k")

    product_ids = sorted(product_to_image.keys())
    product_loader = DataLoader(
        AssetImageDataset(product_ids, product_to_image, preprocess_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    encoded_product_ids, product_embs = encode_images(model, product_loader, device, amp)
    if product_embs.numel() == 0:
        raise RuntimeError("No product embeddings available for validation.")
    pid_to_index = {pid: idx for idx, pid in enumerate(encoded_product_ids)}

    val_bundle_ids = sorted(val_bundle_to_products.keys())
    bundle_loader = DataLoader(
        AssetImageDataset(val_bundle_ids, val_bundle_to_image, preprocess_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )
    encoded_bundle_ids, bundle_embs = encode_images(model, bundle_loader, device, amp)
    if bundle_embs.numel() == 0:
        raise RuntimeError("No bundle embeddings available for validation.")

    topk = min(max_val_k, product_embs.shape[0])
    positives_count = Counter()
    recall_sum = 0.0
    recall_count = 0

    gt_index_tensors: Dict[str, torch.Tensor] = {}
    for bundle_id in encoded_bundle_ids:
        gt_pids = val_bundle_to_products.get(bundle_id, set())
        gt_idx = [pid_to_index[pid] for pid in gt_pids if pid in pid_to_index]
        if not gt_idx:
            continue
        gt_tensor = torch.tensor(sorted(set(gt_idx)), dtype=torch.long, device=device)
        gt_index_tensors[bundle_id] = gt_tensor
        positives_count[int(gt_tensor.numel())] += 1

    for start in tqdm(range(0, len(encoded_bundle_ids), batch_size), desc="Val retrieval", leave=False):
        end = min(start + batch_size, len(encoded_bundle_ids))
        batch_ids = encoded_bundle_ids[start:end]
        batch_emb = bundle_embs[start:end]
        sims = batch_emb @ product_embs.T
        _, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)

        active_rows: List[int] = []
        row_gt_tensors: List[torch.Tensor] = []
        for row, bundle_id in enumerate(batch_ids):
            gt_tensor = gt_index_tensors.get(bundle_id)
            if gt_tensor is None:
                continue
            active_rows.append(row)
            row_gt_tensors.append(gt_tensor)

        if not active_rows:
            continue

        eval_idx = idx[active_rows, :recall_k]
        lengths = torch.tensor([gt.numel() for gt in row_gt_tensors], dtype=torch.float32, device=device)
        max_gt_len = int(max(lengths).item())
        gt_padded = torch.full(
            (len(row_gt_tensors), max_gt_len),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        for row, gt_tensor in enumerate(row_gt_tensors):
            gt_padded[row, : gt_tensor.numel()] = gt_tensor

        matches = (eval_idx.unsqueeze(-1) == gt_padded.unsqueeze(1)).any(dim=-1)
        hits = matches.sum(dim=1).to(torch.float32)
        recalls = hits / lengths
        recall_sum += float(recalls.sum().item())
        recall_count += int(recalls.numel())

    recall = (recall_sum / recall_count) if recall_count > 0 else 0.0
    return recall, dict(sorted(positives_count.items(), key=lambda x: x[0]))


def get_core_clip_model(image_model: nn.Module) -> nn.Module:
    """Return the underlying OpenCLIP model, with/without DataParallel."""
    if isinstance(image_model, nn.DataParallel):
        return image_model.module.clip_model
    return image_model.clip_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    cfg: InditexConfig,
) -> None:
    """Save checkpoint state."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": OmegaConf.to_container(OmegaConf.create(cfg), resolve=True),
    }
    torch.save(payload, path)


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL metrics row."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_gpu_ids(text: str) -> List[int]:
    """Parse comma-separated GPU ids."""
    ids: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids


def train_openclip_retrieval(cfg: InditexConfig, train_manifest: Path, val_manifest: Path, products_manifest: Path, bundles_images_dir: Path, products_images_dir: Path, output_dir: Path) -> None:
    """Train retrieval model with OpenCLIP backend."""
    params = cfg.params

    if params.grad_accum <= 0:
        raise ValueError("params.grad_accum must be >= 1")
    if params.batch_size <= 0:
        raise ValueError("params.batch_size must be > 0")
    if params.epochs <= 0:
        raise ValueError("params.epochs must be > 0")

    set_seed(params.seed)
    device = resolve_device(params.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.jsonl"

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    clip_model = clip_model.to(device)
    image_model: nn.Module = OpenCLIPImageEncoder(clip_model).to(device)

    used_gpu_ids: List[int] = []
    if device.type == "cuda" and params.multi_gpu and torch.cuda.device_count() > 1:
        candidate_ids = parse_gpu_ids(params.gpu_ids)
        max_idx = torch.cuda.device_count() - 1
        used_gpu_ids = [gid for gid in candidate_ids if 0 <= gid <= max_idx]
        if len(used_gpu_ids) >= 2:
            image_model = nn.DataParallel(image_model, device_ids=used_gpu_ids)
            print(f"Using DataParallel on GPUs: {used_gpu_ids}")
        else:
            print(
                "Warning: multi_gpu enabled but fewer than 2 valid gpu_ids found. "
                f"Available=0..{max_idx}, requested={candidate_ids}. Using single GPU."
            )
    image_model.train()
    core_model = get_core_clip_model(image_model)

    product_to_image = parse_products_manifest(products_manifest, products_images_dir=products_images_dir)
    train_bundle_to_image, train_bundle_to_products = parse_bundle_manifest(
        train_manifest, product_to_image, bundles_images_dir=bundles_images_dir
    )
    val_bundle_to_image, val_bundle_to_products = parse_bundle_manifest(
        val_manifest, product_to_image, bundles_images_dir=bundles_images_dir
    )

    train_dataset = BundlePositiveDataset(
        bundle_to_image=train_bundle_to_image,
        bundle_to_products=train_bundle_to_products,
        product_to_image=product_to_image,
        bundle_transform=preprocess_val,
        product_transform=preprocess_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
        drop_last=False,
    )

    optimizer = AdamW(core_model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / params.grad_accum))
    scheduler = build_scheduler(optimizer, total_steps=params.epochs * updates_per_epoch)
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler(enabled=True) if params.amp and device.type == "cuda" else None
    )
    temperature = 0.07

    best_recall = -1.0

    print(f"Train bundles: {len(train_dataset)} | Products indexed: {len(product_to_image)}")
    print(f"Device: {device} | AMP: {bool(scaler is not None)} | multi_gpu={len(used_gpu_ids) >= 2}")

    for epoch in range(1, params.epochs + 1):
        epoch_start = time.time()
        image_model.train()
        running_loss = 0.0
        count_steps = 0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{params.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if batch is None:
                continue
            bundle_imgs = batch["bundle_img"].to(device, non_blocking=True)
            product_imgs = batch["product_img"].to(device, non_blocking=True)
            if bundle_imgs.shape[0] < 2:
                continue

            with torch.autocast(device_type=device.type, enabled=params.amp and device.type == "cuda"):
                bundle_emb = F.normalize(image_model(bundle_imgs).float(), p=2, dim=1)
                product_emb = F.normalize(image_model(product_imgs).float(), p=2, dim=1)
                logits = (bundle_emb @ product_emb.T) / temperature
                targets = torch.arange(logits.shape[0], device=device)
                loss_b2p = F.cross_entropy(logits, targets)
                loss_p2b = F.cross_entropy(logits.T, targets)
                loss = 0.5 * (loss_b2p + loss_p2b)
                loss = loss / params.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_update = (step % params.grad_accum == 0) or (step == len(train_loader))
            if do_update:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            loss_item = float(loss.item() * params.grad_accum)
            running_loss += loss_item
            count_steps += 1

            if step % params.log_every == 0:
                avg_loss = running_loss / max(1, count_steps)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch} step {step}] loss={avg_loss:.5f} lr={lr_now:.7f} "
                    f"bs={bundle_imgs.shape[0]}"
                )

        train_loss = running_loss / max(1, count_steps)
        recall_val, pos_dist = validate_retrieval(
            model=image_model,
            preprocess_val=preprocess_val,
            device=device,
            amp=params.amp,
            val_bundle_to_image=val_bundle_to_image,
            val_bundle_to_products=val_bundle_to_products,
            product_to_image=product_to_image,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            max_val_k=params.max_val_k,
            recall_k=params.recall_k,
        )
        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} recall@{params.recall_k}={recall_val:.6f}")
        print(f"Val #positives per bundle distribution: {pos_dist}")

        metric_row = {
            "epoch": epoch,
            "loss_train": train_loss,
            f"recall@{params.recall_k}": recall_val,
            "lr": lr_now,
            "epoch_seconds": epoch_time,
        }
        append_metrics(metrics_path, metric_row)

        if epoch % params.save_every == 0:
            save_checkpoint(
                path=output_dir / f"epoch_{epoch}.pt",
                model=core_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_recall,
                cfg=cfg,
            )

        if recall_val > best_recall:
            best_recall = recall_val
            save_checkpoint(
                path=output_dir / "best.pt",
                model=core_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_recall,
                cfg=cfg,
            )

    print(f"Training complete. Best recall@{params.recall_k}: {best_recall:.6f}")
