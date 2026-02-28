"""Train image->image retrieval (bundle -> products) with OpenCLIP."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import hydra
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import InditexConfig


cs = ConfigStore.instance()
cs.store(name="inditex_config", node=InditexConfig)


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

    # keep only products that exist in products_manifest index
    filtered: Dict[str, Set[str]] = {}
    dropped = 0
    for bid, pids in bundle_to_products.items():
        keep = {pid for pid in pids if pid in product_to_image}
        dropped += len(pids) - len(keep)
        if keep:
            filtered[bid] = keep
    if dropped > 0:
        print(f"Warning: dropped {dropped} bundle-product links missing in products_manifest.")

    # ensure bundle image entries exist only for bundles with positives
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
        transform,
    ) -> None:
        self.bundle_ids = sorted(bundle_to_products.keys())
        self.bundle_to_image = bundle_to_image
        self.bundle_to_products = {k: sorted(v) for k, v in bundle_to_products.items()}
        self.product_to_image = product_to_image
        self.transform = transform

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
            "bundle_img": self.transform(bundle_img),
            "product_img": self.transform(product_img),
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
                feats = model.encode_image(imgs)
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
    recall_scores: List[float] = []
    positives_count = Counter()

    for start in tqdm(range(0, len(encoded_bundle_ids), batch_size), desc="Val retrieval", leave=False):
        end = min(start + batch_size, len(encoded_bundle_ids))
        batch_ids = encoded_bundle_ids[start:end]
        batch_emb = bundle_embs[start:end]
        sims = batch_emb @ product_embs.T
        _, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)
        idx_cpu = idx.cpu().numpy()

        for i, bundle_id in enumerate(batch_ids):
            gt = val_bundle_to_products.get(bundle_id, set())
            gt_indices = [pid_to_index[pid] for pid in gt if pid in pid_to_index]
            if not gt_indices:
                continue
            positives_count[len(gt_indices)] += 1
            pred = idx_cpu[i, :recall_k]
            hits = sum(1 for p in pred if p in set(gt_indices))
            recall_scores.append(hits / len(gt_indices))

    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    return recall, dict(sorted(positives_count.items(), key=lambda x: x[0]))


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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: InditexConfig) -> None:
    """Entrypoint."""
    params = cfg.params
    files = cfg.files

    train_manifest = Path(to_absolute_path(files.train_manifest))
    val_manifest = Path(to_absolute_path(files.val_manifest))
    products_manifest = Path(to_absolute_path(files.products_manifest))
    bundles_images_dir = Path(to_absolute_path(files.bundles_images))
    products_images_dir = Path(to_absolute_path(files.products_images))
    output_dir = Path(to_absolute_path(files.output_dir))

    if params.grad_accum <= 0:
        raise ValueError("--grad_accum must be >= 1")
    if params.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if params.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    set_seed(params.seed)
    device = resolve_device(params.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ensure_dir(output_dir)
    metrics_path = output_dir / "metrics.jsonl"

    # Required model load exactly as requested.
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    _ = tokenizer
    model = model.to(device)
    model.train()

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
        transform=preprocess_train,
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

    optimizer = AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / params.grad_accum))
    scheduler = build_scheduler(optimizer, total_steps=params.epochs * updates_per_epoch)
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler(enabled=True) if params.amp and device.type == "cuda" else None
    )
    temperature = 0.07

    best_recall = -1.0
    global_step = 0

    print(f"Train bundles: {len(train_dataset)} | Products indexed: {len(product_to_image)}")
    print(f"Device: {device} | AMP: {bool(scaler is not None)}")

    for epoch in range(1, params.epochs + 1):
        epoch_start = time.time()
        model.train()
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
                bundle_emb = F.normalize(model.encode_image(bundle_imgs).float(), p=2, dim=1)
                product_emb = F.normalize(model.encode_image(product_imgs).float(), p=2, dim=1)
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
                global_step += 1

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
            model=model,
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
                model=model,
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
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_recall,
                cfg=cfg,
            )

    print(f"Training complete. Best recall@{params.recall_k}: {best_recall:.6f}")


if __name__ == "__main__":
    main()
