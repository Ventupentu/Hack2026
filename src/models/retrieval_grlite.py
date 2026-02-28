"""GR-Lite retrieval training implementation (bundle -> products)."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import InditexConfig

Pair = Tuple[str, str]


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


def parse_gpu_ids(text: str) -> List[int]:
    """Parse comma-separated GPU ids."""
    ids: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _first_existing_key(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and _as_str(row.get(key)):
            return _as_str(row.get(key))
    return ""


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


def parse_manifest_pairs(path: Path) -> List[Pair]:
    """Parse manifest into (bundle_id, product_id) pairs."""
    rows = read_manifest_rows(path)
    seen: Set[Pair] = set()
    pairs: List[Pair] = []
    for row in rows:
        bundle_id = _first_existing_key(row, ("bundle_asset_id", "bundle_id", "query_id", "id"))
        product_id = _first_existing_key(row, ("product_asset_id", "product_id", "candidate_id"))
        if not bundle_id or not product_id:
            continue
        pair = (bundle_id, product_id)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    if not pairs:
        raise RuntimeError(f"No valid bundle-product pairs found in manifest: {path}")
    return pairs


def build_image_map(image_dir: Path) -> Dict[str, Path]:
    """Map asset_id (filename stem) to local image path."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_map: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file():
            image_map[path.stem] = path
    return image_map


def filter_pairs_with_images(
    pairs: Sequence[Pair],
    bundle_image_map: Dict[str, Path],
    product_image_map: Dict[str, Path],
) -> List[Pair]:
    """Drop pairs where bundle/product image is missing."""
    kept: List[Pair] = []
    dropped = 0
    for bundle_id, product_id in pairs:
        if bundle_id not in bundle_image_map or product_id not in product_image_map:
            dropped += 1
            continue
        kept.append((bundle_id, product_id))
    if dropped:
        print(f"Dropped {dropped} pairs without local images.")
    if not kept:
        raise RuntimeError("No train pairs left after filtering by available images.")
    return kept


def split_pairs_by_bundle(
    pairs: Sequence[Pair],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Pair], List[Pair]]:
    """Split pairs by bundle id to avoid leakage between train/val."""
    if val_ratio <= 0.0:
        return list(pairs), []

    bundle_ids = sorted({bundle_id for bundle_id, _ in pairs})
    if len(bundle_ids) < 2:
        return list(pairs), []

    rng = random.Random(seed)
    rng.shuffle(bundle_ids)
    val_count = max(1, int(round(len(bundle_ids) * val_ratio)))
    val_bundles = set(bundle_ids[:val_count])

    train_pairs = [pair for pair in pairs if pair[0] not in val_bundles]
    val_pairs = [pair for pair in pairs if pair[0] in val_bundles]
    if not train_pairs or not val_pairs:
        return list(pairs), []
    return train_pairs, val_pairs


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


class PairDataset(Dataset):
    """Dataset yielding (bundle_image, product_image) positive pairs."""

    def __init__(
        self,
        pairs: Sequence[Pair],
        bundle_image_map: Dict[str, Path],
        product_image_map: Dict[str, Path],
        transform: Any,
    ) -> None:
        self.pairs = list(pairs)
        self.bundle_image_map = bundle_image_map
        self.product_image_map = product_image_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        bundle_id, product_id = self.pairs[idx]
        bundle_img = open_image_safe(self.bundle_image_map[bundle_id])
        product_img = open_image_safe(self.product_image_map[product_id])
        if bundle_img is None or product_img is None:
            return None
        return {
            "bundle": self.transform(bundle_img),
            "product": self.transform(product_img),
        }


def collate_skip_none(batch: Sequence[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """Drop unreadable samples and collate tensors."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "bundle": torch.stack([item["bundle"] for item in batch], dim=0),
        "product": torch.stack([item["product"] for item in batch], dim=0),
    }


def _extract_features_from_outputs(outputs: Any) -> torch.Tensor:
    """Convert different model output shapes to [B, D] feature tensor."""
    if isinstance(outputs, torch.Tensor):
        features = outputs
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        features = outputs.last_hidden_state[:, 0]
    elif isinstance(outputs, dict):
        if "pooler_output" in outputs and outputs["pooler_output"] is not None:
            features = outputs["pooler_output"]
        elif "last_hidden_state" in outputs and outputs["last_hidden_state"] is not None:
            features = outputs["last_hidden_state"][:, 0]
        else:
            raise RuntimeError("Model output dict does not contain usable features.")
    else:
        raise RuntimeError(f"Unsupported output type for feature extraction: {type(outputs)}")

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim == 3:
        features = features[:, 0]
    if features.ndim != 2:
        raise RuntimeError(f"Expected 2D features [B, D], got shape {tuple(features.shape)}")
    return features


class GRLiteTensorEncoder(nn.Module):
    """Tensor-only GR-Lite wrapper suitable for DataParallel."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.base_model(images)
        except Exception:
            if hasattr(self.base_model, "model"):
                outputs = self.base_model.model(images)
            else:
                raise
        return _extract_features_from_outputs(outputs)


def get_core_model(model: nn.Module) -> nn.Module:
    """Return model.module when wrapped in DataParallel."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def encode_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode image batch into normalized embeddings."""
    outputs = model(images)
    feats = _extract_features_from_outputs(outputs).float()
    return F.normalize(feats, p=2, dim=1)


def batch_infonce_loss(
    model: nn.Module,
    bundle_imgs: torch.Tensor,
    product_imgs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE loss with in-batch negatives."""
    query_embs = encode_images(model, bundle_imgs)
    product_embs = encode_images(model, product_imgs)
    logits = (query_embs @ product_embs.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_q + loss_p)


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    temperature: float,
) -> float:
    """Compute average validation loss."""
    if len(loader) == 0:
        return float("nan")
    model.eval()
    amp_enabled = amp and device.type == "cuda"
    total_loss = 0.0
    total_batches = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        if batch is None:
            continue
        bundle_imgs = batch["bundle"].to(device, non_blocking=True)
        product_imgs = batch["product"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = batch_infonce_loss(
                model=model,
                bundle_imgs=bundle_imgs,
                product_imgs=product_imgs,
                temperature=temperature,
            )
        total_loss += float(loss.item())
        total_batches += 1
    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    model_name: str,
    input_size: int,
    feature_dim: int,
    temperature: float,
) -> None:
    """Persist one training checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "model_name": str(model_name),
            "input_size": int(input_size),
            "feature_dim": int(feature_dim),
            "temperature": float(temperature),
        },
        path,
    )


def write_metrics(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write epoch metrics to json."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")


def train_grlite_retrieval(
    cfg: InditexConfig,
    train_manifest: Path,
    val_manifest: Path,
    products_manifest: Path,
    bundles_images_dir: Path,
    products_images_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> None:
    """Train/fine-tune GR-Lite on bundle-product positives."""
    del products_manifest, cache_dir

    params = cfg.params
    set_seed(int(params.seed))
    device = resolve_device(str(params.device))
    amp_enabled = bool(params.amp and device.type == "cuda")
    if params.grad_accum <= 0:
        raise ValueError("params.grad_accum must be >= 1")

    model_name = str(getattr(params, "grlite_model_name", "srpone/gr-lite")).strip() or "srpone/gr-lite"
    input_size = int(getattr(params, "grlite_input_size", 518))
    feature_dim = int(getattr(params, "grlite_feature_dim", 256))
    temperature = float(getattr(params, "grlite_temperature", 0.07))
    split_val_ratio = float(getattr(params, "grlite_val_ratio", 0.1))
    resume_checkpoint = str(getattr(params, "grlite_resume_checkpoint", "")).strip()

    if input_size <= 0:
        raise ValueError("params.grlite_input_size must be > 0")
    if feature_dim <= 0:
        raise ValueError("params.grlite_feature_dim must be > 0")
    if temperature <= 0:
        raise ValueError("params.grlite_temperature must be > 0")

    try:
        from transformers import AutoConfig, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for GR-Lite training. Install with: pip install transformers"
        ) from exc

    print(f"Loading GR-Lite model from: {model_name}")
    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model_cfg, "is_crop"):
        model_cfg.is_crop = False
    base_model = AutoModel.from_pretrained(model_name, config=model_cfg, trust_remote_code=True)
    base_model = base_model.to(device)
    model: nn.Module = GRLiteTensorEncoder(base_model).to(device)
    used_gpu_ids: List[int] = []
    if device.type == "cuda" and params.multi_gpu and torch.cuda.device_count() > 1:
        candidate_ids = parse_gpu_ids(params.gpu_ids)
        max_idx = torch.cuda.device_count() - 1
        used_gpu_ids = [gid for gid in candidate_ids if 0 <= gid <= max_idx]
        if len(used_gpu_ids) >= 2:
            model = nn.DataParallel(model, device_ids=used_gpu_ids)
            print(f"Using DataParallel on GPUs: {used_gpu_ids}")
        else:
            print(
                "Warning: multi_gpu enabled but fewer than 2 valid gpu_ids found. "
                f"Available=0..{max_idx}, requested={candidate_ids}. Using single GPU."
            )
    core_model = get_core_model(model)

    preprocess = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    bundle_image_map = build_image_map(bundles_images_dir)
    product_image_map = build_image_map(products_images_dir)

    train_pairs = parse_manifest_pairs(train_manifest)
    if train_manifest.resolve() == val_manifest.resolve():
        train_pairs, val_pairs = split_pairs_by_bundle(
            pairs=train_pairs,
            val_ratio=split_val_ratio,
            seed=int(params.seed),
        )
    else:
        val_pairs = parse_manifest_pairs(val_manifest)

    train_pairs = filter_pairs_with_images(train_pairs, bundle_image_map, product_image_map)
    val_pairs = filter_pairs_with_images(val_pairs, bundle_image_map, product_image_map) if val_pairs else []

    train_ds = PairDataset(
        pairs=train_pairs,
        bundle_image_map=bundle_image_map,
        product_image_map=product_image_map,
        transform=preprocess,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params.batch_size),
        shuffle=True,
        num_workers=int(params.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_skip_none,
    )

    val_loader: Optional[DataLoader] = None
    if val_pairs:
        val_ds = PairDataset(
            pairs=val_pairs,
            bundle_image_map=bundle_image_map,
            product_image_map=product_image_map,
            transform=preprocess,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(params.batch_size),
            shuffle=False,
            num_workers=int(params.num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_skip_none,
        )

    print(
        f"GR-Lite training setup | train_pairs={len(train_pairs)} "
        f"val_pairs={len(val_pairs)} input_size={input_size} "
        f"feature_dim={feature_dim} temp={temperature}"
    )
    print(f"Device: {device} | AMP: {amp_enabled} | multi_gpu={len(used_gpu_ids) >= 2}")

    optimizer = AdamW(
        core_model.parameters(),
        lr=float(params.lr),
        weight_decay=float(params.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []
    ensure_dir(output_dir)

    total_epochs = int(params.epochs)
    start_epoch = 1
    grad_accum = int(params.grad_accum)
    log_every = max(1, int(params.log_every))
    save_every = max(1, int(params.save_every))

    if resume_checkpoint:
        from hydra.utils import to_absolute_path

        ckpt_path = Path(to_absolute_path(resume_checkpoint))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"params.grlite_resume_checkpoint does not exist: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Invalid checkpoint format at {ckpt_path}")
        missing, unexpected = core_model.load_state_dict(state_dict, strict=False)
        if isinstance(payload, dict) and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                print(f"Warning: could not load optimizer state from checkpoint ({exc})")
        if isinstance(payload, dict) and "epoch" in payload:
            start_epoch = int(payload["epoch"]) + 1
        if isinstance(payload, dict) and "val_loss" in payload and np.isfinite(payload["val_loss"]):
            best_val_loss = float(payload["val_loss"])
        if start_epoch > total_epochs:
            raise ValueError(
                f"params.epochs={total_epochs} must be > checkpoint epoch ({start_epoch - 1}) "
                "to continue training."
            )
        print(
            f"Resumed from checkpoint: {ckpt_path} | start_epoch={start_epoch} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        running_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if batch is None:
                continue

            bundle_imgs = batch["bundle"].to(device, non_blocking=True)
            product_imgs = batch["product"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = batch_infonce_loss(
                    model=model,
                    bundle_imgs=bundle_imgs,
                    product_imgs=product_imgs,
                    temperature=temperature,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            do_update = (step % grad_accum == 0) or (step == len(train_loader))
            if do_update:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = float(loss.item() * grad_accum)
            running_loss += batch_loss
            num_batches += 1

            if step % log_every == 0:
                progress.set_postfix(loss=f"{running_loss / max(1, num_batches):.4f}")

        train_loss = running_loss / max(1, num_batches)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate_loss(
                model=model,
                loader=val_loader,
                device=device,
                amp=amp_enabled,
                temperature=temperature,
            )

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(elapsed),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"time={elapsed:.1f}s"
        )

        if epoch % save_every == 0:
            save_checkpoint(
                path=output_dir / f"epoch_{epoch}.pt",
                model=core_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                model_name=model_name,
                input_size=input_size,
                feature_dim=feature_dim,
                temperature=temperature,
            )

        if val_loader is not None and np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=output_dir / "best.pt",
                model=core_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                model_name=model_name,
                input_size=input_size,
                feature_dim=feature_dim,
                temperature=temperature,
            )

    last = history[-1] if history else {"epoch": 0, "train_loss": float("nan"), "val_loss": float("nan")}
    save_checkpoint(
        path=output_dir / "last.pt",
        model=core_model,
        optimizer=optimizer,
        epoch=int(last["epoch"]),
        train_loss=float(last["train_loss"]),
        val_loss=float(last["val_loss"]),
        model_name=model_name,
        input_size=input_size,
        feature_dim=feature_dim,
        temperature=temperature,
    )

    write_metrics(output_dir / "train_metrics.json", history)
    print(f"Training finished. Outputs saved to: {output_dir}")
