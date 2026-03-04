"""GR-Lite retrieval training implementation (bundle -> products)."""

from __future__ import annotations

import csv
import json
import math
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
from src.utils.hf_hub_sync import build_hf_uploader

Pair = Tuple[str, str]


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen nn.Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha / rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


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


def torch_load_any(path: Path, map_location: Any) -> Any:
    """Load torch object across torch versions (weights_only arg compatibility)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_local_grlite_pt(path: Path) -> Optional[Path]:
    """Resolve local GR-Lite serialized model path."""
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    for candidate in ("gr_lite.pt", "gr-lite.pt", "model.pt", "best.pt", "last.pt"):
        resolved = path / candidate
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop DataParallel 'module.' prefix when present."""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        plen = len(prefix)
        return {key[plen:]: value for key, value in state_dict.items()}
    return state_dict


def _as_tensor_state_dict(payload: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Return payload as plain tensor state_dict when possible."""
    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return dict(payload)
    return None


def build_eomt_model_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    input_size: int,
) -> nn.Module:
    """Build EOMT-DINOv3 model and load a GR-Lite-like state_dict into it."""
    try:
        from transformers import EomtDinov3Config, EomtDinov3ForUniversalSegmentation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers (with EomtDinov3 classes) is required to load GR-Lite tensor checkpoints."
        ) from exc

    sd = strip_module_prefix(dict(state_dict))
    # Known wrappers seen across checkpoints.
    sd = _strip_state_dict_prefix(sd, "base_model.")
    sd = _strip_state_dict_prefix(sd, "model.model.")
    sd = _strip_state_dict_prefix(sd, "model.")

    # Convert key names to EOMT naming.
    converted: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = key
        if new_key.startswith("layer."):
            new_key = "layers." + new_key[len("layer."):]
        elif new_key.startswith("norm."):
            new_key = "layernorm." + new_key[len("norm."):]
        converted[new_key] = value

    if "embeddings.cls_token" not in converted:
        raise RuntimeError(
            "Could not identify GR-Lite backbone keys in tensor checkpoint "
            "(missing embeddings.cls_token after prefix conversion)."
        )

    hidden_size = int(converted["embeddings.cls_token"].shape[-1])
    patch_size = int(converted["embeddings.patch_embeddings.weight"].shape[-1])
    num_channels = int(converted["embeddings.patch_embeddings.weight"].shape[1])
    num_register_tokens = int(converted.get("embeddings.register_tokens", torch.zeros(1, 0, hidden_size)).shape[1])

    layer_ids = sorted(
        {
            int(key.split(".")[1])
            for key in converted.keys()
            if key.startswith("layers.") and len(key.split(".")) >= 3 and key.split(".")[1].isdigit()
        }
    )
    if not layer_ids:
        raise RuntimeError("Could not infer number of layers from state_dict keys.")
    num_hidden_layers = int(max(layer_ids) + 1)

    up_proj_key = next((k for k in converted.keys() if k.endswith("mlp.up_proj.weight")), None)
    if up_proj_key is None:
        raise RuntimeError("Could not infer intermediate_size (missing mlp.up_proj.weight).")
    intermediate_size = int(converted[up_proj_key].shape[0])

    # DINOv3-L style defaults (64 dim/head) when possible.
    num_attention_heads = int(hidden_size // 64) if hidden_size % 64 == 0 else 16
    if hidden_size % num_attention_heads != 0:
        # Fallback to any divisor >= 8 to avoid invalid config.
        divisors = [d for d in range(32, 7, -1) if hidden_size % d == 0]
        num_attention_heads = divisors[0] if divisors else 8

    cfg = EomtDinov3Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_register_tokens=num_register_tokens,
        patch_size=patch_size,
        num_channels=num_channels,
        image_size=input_size,
    )

    model = EomtDinov3ForUniversalSegmentation(cfg)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    print(
        "Loaded tensor checkpoint into EOMT-DINOv3 | "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"layers={num_hidden_layers} hidden={hidden_size} heads={num_attention_heads}"
    )
    return model.to(device)


def load_grlite_base_model(model_name: str, device: torch.device, input_size: int) -> nn.Module:
    """Load GR-Lite from transformers repo or serialized .pt fallback."""
    # First attempt: transformers-style repo (config + weights)
    try:
        from transformers import AutoConfig, AutoModel

        model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(model_cfg, "is_crop"):
            model_cfg.is_crop = False
        model = AutoModel.from_pretrained(model_name, config=model_cfg, trust_remote_code=True)
        return model.to(device)
    except Exception as exc:
        print(f"Transformers loader failed for '{model_name}', trying serialized .pt fallback ({exc})")

    # Fallback: serialized model file (as documented in srpone/gr-lite card)
    local_ref = find_local_grlite_pt(Path(model_name).expanduser())
    model_path: Path
    if local_ref is not None:
        model_path = local_ref.resolve()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as hub_exc:
            raise ModuleNotFoundError(
                "Failed to load GR-Lite via transformers and huggingface_hub is missing. "
                "Install with: pip install huggingface_hub"
            ) from hub_exc
        downloaded = hf_hub_download(repo_id=model_name, filename="gr_lite.pt")
        model_path = Path(downloaded).resolve()

    loaded = torch_load_any(model_path, map_location=device)
    if isinstance(loaded, nn.Module):
        return loaded.to(device)
    if isinstance(loaded, dict) and "model" in loaded:
        if isinstance(loaded["model"], nn.Module):
            return loaded["model"].to(device)
        state_dict = _as_tensor_state_dict(loaded["model"])
        if state_dict is not None:
            return build_eomt_model_from_state_dict(
                state_dict=state_dict,
                device=device,
                input_size=input_size,
            )
    state_dict = _as_tensor_state_dict(loaded)
    if state_dict is not None:
        return build_eomt_model_from_state_dict(
            state_dict=state_dict,
            device=device,
            input_size=input_size,
        )
    raise RuntimeError(
        f"Unsupported serialized GR-Lite payload type: {type(loaded)} at {model_path}. "
        "Expected torch.nn.Module or dict with key 'model'."
    )


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


def _extract_layer_ids_from_named_params(named_params: Iterable[Tuple[str, nn.Parameter]]) -> List[int]:
    layer_ids: Set[int] = set()
    for name, _param in named_params:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_ids.add(int(parts[i + 1]))
    return sorted(layer_ids)


def _extract_layer_id_from_module_name(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _get_parent_and_child_module(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def parse_csv_list(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    last_n_layers: int = 0,
) -> int:
    """Replace selected Linear layers with LoRA-wrapped layers."""
    named_modules = list(model.named_modules())
    linear_module_names = [name for name, module in named_modules if isinstance(module, nn.Linear)]
    if not linear_module_names:
        return 0

    target_tokens = [token.lower() for token in target_modules if token]
    layer_ids = sorted(
        {
            layer_id
            for layer_id in (_extract_layer_id_from_module_name(name) for name in linear_module_names)
            if layer_id is not None
        }
    )
    start_layer: Optional[int] = None
    if last_n_layers > 0 and layer_ids:
        start_layer = max(0, max(layer_ids) - last_n_layers + 1)

    replaced = 0
    for name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        if target_tokens and not any(token in lname for token in target_tokens):
            continue
        layer_id = _extract_layer_id_from_module_name(name)
        if start_layer is not None and layer_id is not None and layer_id < start_layer:
            continue
        parent, child = _get_parent_and_child_module(model, name)
        setattr(parent, child, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def configure_trainable_parameters(
    core_model: nn.Module,
    tune_mode: str,
    train_last_n_layers: int,
    unfreeze_layernorm: bool,
) -> None:
    """Configure parameter-efficient fine-tuning modes for GR-Lite."""
    mode = str(tune_mode).strip().lower()
    if mode in {"full", "all"}:
        for _name, param in core_model.named_parameters():
            param.requires_grad = True
        return

    if mode == "lora":
        for _name, param in core_model.named_parameters():
            param.requires_grad = False

        for name, param in core_model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad = True
            elif unfreeze_layernorm and (
                ".layernorm." in name
                or ".norm." in name
                or name.endswith("layernorm.weight")
                or name.endswith("layernorm.bias")
                or name.endswith("norm.weight")
                or name.endswith("norm.bias")
            ):
                param.requires_grad = True
        return

    if mode not in {"last_n", "lastn", "partial"}:
        raise ValueError(
            f"Unsupported params.grlite_tune_mode='{tune_mode}'. "
            "Available: full, last_n, lora"
        )

    if train_last_n_layers <= 0:
        raise ValueError("params.grlite_train_last_n_layers must be > 0 when tune_mode=last_n")

    # Freeze all first
    for _name, param in core_model.named_parameters():
        param.requires_grad = False

    named_params = list(core_model.named_parameters())
    layer_ids = _extract_layer_ids_from_named_params(named_params)
    if not layer_ids:
        raise RuntimeError("Could not detect transformer layers for last_n fine-tuning.")
    max_layer = max(layer_ids)
    start_layer = max(0, max_layer - train_last_n_layers + 1)

    for name, param in named_params:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_id = int(parts[i + 1])
                if layer_id >= start_layer:
                    param.requires_grad = True
                break

        if unfreeze_layernorm and (
            ".layernorm." in name
            or ".norm." in name
            or name.endswith("layernorm.weight")
            or name.endswith("layernorm.bias")
            or name.endswith("norm.weight")
            or name.endswith("norm.bias")
        ):
            param.requires_grad = True


def summarize_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = 0
    trainable = 0
    for p in model.parameters():
        count = int(p.numel())
        total += count
        if p.requires_grad:
            trainable += count
    return trainable, total


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
    tune_mode: str = "full",
    use_lora: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 0.0,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[Sequence[str]] = None,
    lora_last_n_layers: int = 0,
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
            "tune_mode": str(tune_mode),
            "use_lora": bool(use_lora),
            "lora_rank": int(lora_rank),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "lora_target_modules": list(lora_target_modules or []),
            "lora_last_n_layers": int(lora_last_n_layers),
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
    tune_mode = str(getattr(params, "grlite_tune_mode", "full")).strip().lower()
    train_last_n_layers = int(getattr(params, "grlite_train_last_n_layers", 2))
    unfreeze_layernorm = bool(getattr(params, "grlite_unfreeze_layernorm", True))
    use_lora = bool(getattr(params, "grlite_use_lora", False))
    lora_rank = int(getattr(params, "grlite_lora_r", 8))
    lora_alpha = float(getattr(params, "grlite_lora_alpha", 16.0))
    lora_dropout = float(getattr(params, "grlite_lora_dropout", 0.05))
    lora_target_modules = parse_csv_list(getattr(params, "grlite_lora_target_modules", "q_proj,v_proj"))
    lora_last_n_layers = int(getattr(params, "grlite_lora_last_n_layers", 0))

    if input_size <= 0:
        raise ValueError("params.grlite_input_size must be > 0")
    if feature_dim <= 0:
        raise ValueError("params.grlite_feature_dim must be > 0")
    if temperature <= 0:
        raise ValueError("params.grlite_temperature must be > 0")
    if tune_mode == "lora" and not use_lora:
        raise ValueError(
            "params.grlite_tune_mode=lora requires params.grlite_use_lora=true"
        )
    if use_lora and lora_rank <= 0:
        raise ValueError("params.grlite_lora_r must be > 0 when grlite_use_lora=true")

    ensure_dir(output_dir)
    metrics_path = output_dir / "train_metrics.json"
    uploader = build_hf_uploader(cfg=cfg, output_dir=output_dir, artifact_namespace="grlite")

    print(f"Loading GR-Lite model from: {model_name}")
    base_model = load_grlite_base_model(model_name=model_name, device=device, input_size=input_size)
    if use_lora:
        replaced = apply_lora_to_model(
            model=base_model,
            target_modules=lora_target_modules,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            last_n_layers=lora_last_n_layers,
        )
        if replaced <= 0:
            raise RuntimeError(
                "grlite_use_lora=true but no Linear layers matched grlite_lora_target_modules."
            )
        if tune_mode in {"full", "all"}:
            print("grlite_use_lora=true overrides grlite_tune_mode from full -> lora")
            tune_mode = "lora"
        print(
            "Applied LoRA adapters | "
            f"modules={replaced} rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout} "
            f"targets={','.join(lora_target_modules)} last_n_layers={lora_last_n_layers}"
        )

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
    configure_trainable_parameters(
        core_model=core_model,
        tune_mode=tune_mode,
        train_last_n_layers=train_last_n_layers,
        unfreeze_layernorm=unfreeze_layernorm,
    )

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
    trainable_params, total_params = summarize_trainable_parameters(core_model)
    if trainable_params <= 0:
        raise RuntimeError("No trainable parameters selected. Check grlite_tune_mode settings.")
    print(
        f"Device: {device} | AMP: {amp_enabled} | multi_gpu={len(used_gpu_ids) >= 2} | "
        f"tune_mode={tune_mode} | trainable={trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M"
    )

    optimizer = AdamW(
        [p for p in core_model.parameters() if p.requires_grad],
        lr=float(params.lr),
        weight_decay=float(params.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

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
        payload = torch_load_any(ckpt_path, map_location="cpu")
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
        write_metrics(metrics_path, history)

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
                tune_mode=tune_mode,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                lora_last_n_layers=lora_last_n_layers,
            )

        if val_loader is not None and np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = output_dir / "best.pt"
            save_checkpoint(
                path=best_ckpt_path,
                model=core_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                model_name=model_name,
                input_size=input_size,
                feature_dim=feature_dim,
                temperature=temperature,
                tune_mode=tune_mode,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                lora_last_n_layers=lora_last_n_layers,
            )
            if uploader is not None:
                uploader.queue_checkpoint_artifacts(
                    checkpoint_path=best_ckpt_path,
                    metrics_path=metrics_path,
                    checkpoint_label="best",
                )

    last = history[-1] if history else {"epoch": 0, "train_loss": float("nan"), "val_loss": float("nan")}
    last_ckpt_path = output_dir / "last.pt"
    save_checkpoint(
        path=last_ckpt_path,
        model=core_model,
        optimizer=optimizer,
        epoch=int(last["epoch"]),
        train_loss=float(last["train_loss"]),
        val_loss=float(last["val_loss"]),
        model_name=model_name,
        input_size=input_size,
        feature_dim=feature_dim,
        temperature=temperature,
        tune_mode=tune_mode,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_last_n_layers=lora_last_n_layers,
    )

    write_metrics(metrics_path, history)
    if uploader is not None:
        try:
            uploader.queue_checkpoint_artifacts(
                checkpoint_path=last_ckpt_path,
                metrics_path=metrics_path,
                checkpoint_label="last",
            )
            uploader.wait_for_pending_uploads()
        finally:
            uploader.shutdown()
    print(f"Training finished. Outputs saved to: {output_dir}")
