"""GR-Lite retrieval training implementation (bundle -> products).

Improvements:
  A) Cross-batch gradient accumulation for InfoNCE – accumulates embeddings
     across micro-batches so contrastive loss sees BS×grad_accum negatives.
  B) LoRA via peft – injects low-rank adapters into all attention projections.
  C) Hard Negative Mining – groups products of the same product_description
     within each batch so the model must learn fine-grained differences.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
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


# ---------------------------------------------------------------------------
# C) Hard Negative Mining – category-aware batching
# ---------------------------------------------------------------------------

def load_product_categories(data_dir: Path) -> Dict[str, str]:
    """Load product_asset_id → product_description from product_dataset.csv."""
    csv_path = data_dir / "product_dataset.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found – hard-negative mining disabled.")
        return {}
    cat_map: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = _as_str(row.get("product_asset_id"))
            desc = _as_str(row.get("product_description"))
            if pid and desc:
                cat_map[pid] = desc
    print(f"Loaded {len(cat_map)} product categories ({len(set(cat_map.values()))} unique descriptions).")
    return cat_map


class CategoryBatchSampler(BatchSampler):
    """Yields batches where products share the same product_description.

    For each batch: pick a category → sample up to *batch_size* pairs from that
    category.  If a category has fewer pairs than batch_size, top up with
    repetition from the same category.
    Categories with <2 pairs fall back to a mixed bucket.

    Used as ``batch_sampler=`` in DataLoader (yields ``List[int]`` per batch).
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        product_categories: Dict[str, str],
        batch_size: int,
        seed: int = 42,
    ) -> None:
        # We don't call super().__init__ because we manage everything ourselves.
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        # Group pair indices by product category
        self.cat_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, (_, pid) in enumerate(pairs):
            cat = product_categories.get(pid, "__other__")
            self.cat_to_indices[cat].append(i)

        # Categories with enough pairs (>=2) for meaningful hard-negatives
        self.categories = [c for c, idxs in self.cat_to_indices.items() if len(idxs) >= 2]
        # Singletons → mixed pool
        other_idxs: List[int] = []
        for c, idxs in self.cat_to_indices.items():
            if len(idxs) < 2:
                other_idxs.extend(idxs)
        if other_idxs:
            self.cat_to_indices["__mixed__"] = other_idxs
            if "__mixed__" not in self.categories:
                self.categories.append("__mixed__")

        self._total_pairs = len(pairs)
        self._num_batches = max(1, (self._total_pairs + batch_size - 1) // batch_size)
        print(
            f"CategoryBatchSampler: {len(self.categories)} categories, "
            f"{self._total_pairs} total pairs, batch_size={batch_size}, "
            f"~{self._num_batches} batches/epoch"
        )

    def __iter__(self):
        cats = list(self.categories)
        self.rng.shuffle(cats)
        cat_pools: Dict[str, List[int]] = {}
        for c in cats:
            pool = list(self.cat_to_indices[c])
            self.rng.shuffle(pool)
            cat_pools[c] = pool

        yielded = 0
        cat_idx = 0
        while yielded < self._total_pairs:
            cat = cats[cat_idx % len(cats)]
            pool = cat_pools[cat]
            if not pool:
                pool = list(self.cat_to_indices[cat])
                self.rng.shuffle(pool)
                cat_pools[cat] = pool
            take = min(self.batch_size, len(pool))
            batch = pool[:take]
            cat_pools[cat] = pool[take:]
            # Pad to batch_size from same category if needed & enough remain
            while len(batch) < self.batch_size and (self._total_pairs - yielded) >= self.batch_size:
                refill = list(self.cat_to_indices[cat])
                self.rng.shuffle(refill)
                batch.extend(refill[: self.batch_size - len(batch)])
            yield batch  # ← yield a List[int] (one complete batch)
            yielded += len(batch)
            cat_idx += 1

    def __len__(self) -> int:
        return self._num_batches


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


class ProjectionHead(nn.Module):
    """Lightweight MLP projection on top of frozen backbone features."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: Optional[int] = None) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _find_backbone_blocks(base_model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk common attribute paths to find the list of transformer blocks.

    GR-Lite models can be:
    - transformers AutoModel  → model.encoder.layer  /  model.layers
    - EomtDinov3ForUniversalSegmentation → model.layers  /  layers
    - A raw nn.Module with .blocks / .layers
    """
    candidates: List[str] = []
    # Build a list of dotted-path candidates to try
    for prefix in ("", "model.", "model.model.", "base_model.", "base_model.model."):
        for attr in ("encoder.layer", "encoder.layers", "layers", "blocks"):
            candidates.append(prefix + attr)

    for dotted in candidates:
        obj = base_model
        try:
            for part in dotted.split("."):
                if not part:
                    continue
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, nn.Sequential)) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    return None


def freeze_grlite_backbone(
    base_model: nn.Module,
    unfreeze_last_n_blocks: int = 0,
) -> Tuple[int, int]:
    """Freeze all backbone params, then unfreeze last N transformer blocks + final norms.

    Returns (frozen_count, trainable_count) as *number of parameter tensors*.
    """
    # 1) Freeze everything
    for param in base_model.parameters():
        param.requires_grad = False

    if unfreeze_last_n_blocks <= 0:
        frozen = sum(1 for p in base_model.parameters())
        return frozen, 0

    # 2) Find transformer blocks
    blocks = _find_backbone_blocks(base_model)
    if blocks is None:
        print("Warning: could not locate transformer blocks in GR-Lite – nothing unfrozen.")
        frozen = sum(1 for p in base_model.parameters())
        return frozen, 0

    n_blocks = len(blocks)
    unfreeze_n = min(unfreeze_last_n_blocks, n_blocks)

    for block in blocks[-unfreeze_n:]:
        for param in block.parameters():
            param.requires_grad = True

    # 3) Unfreeze final norms / heads that come after blocks
    for name, module in base_model.named_modules():
        lower = name.lower().split(".")[-1] if name else ""
        if lower in ("layernorm", "ln_post", "ln_final", "norm", "head", "fc_norm"):
            for param in module.parameters():
                param.requires_grad = True

    frozen = sum(1 for p in base_model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in base_model.parameters() if p.requires_grad)
    print(
        f"GR-Lite backbone: froze {frozen} params, unfroze last "
        f"{unfreeze_n}/{n_blocks} blocks → {trainable} trainable param tensors"
    )
    return frozen, trainable


# ---------------------------------------------------------------------------
# B) LoRA via peft – inject low-rank adapters into attention layers
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    base_model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Wrap *base_model* with LoRA adapters on attention projections.

    All original params are frozen; only the LoRA deltas (~1 % of params) train.
    Returns the peft-wrapped model (forward signature unchanged).
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(
        f"LoRA applied | r={r} alpha={lora_alpha} targets={target_modules}\n"
        f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    return peft_model


class GRLiteTensorEncoder(nn.Module):
    """Tensor-only GR-Lite wrapper suitable for DataParallel."""

    def __init__(self, base_model: nn.Module, proj_head: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.proj_head = proj_head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.base_model(images)
        except Exception:
            if hasattr(self.base_model, "model"):
                outputs = self.base_model.model(images)
            else:
                raise
        feats = _extract_features_from_outputs(outputs)
        if self.proj_head is not None:
            feats = self.proj_head(feats)
        return feats


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


# ---------------------------------------------------------------------------
# A) Cross-batch InfoNCE – accumulate embeddings for virtual large batch
# ---------------------------------------------------------------------------

def cross_batch_infonce_loss(
    all_query_embs: torch.Tensor,
    all_product_embs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE over pre-accumulated embeddings from multiple micro-batches.

    Args:
        all_query_embs:   [N, D] normalized bundle embeddings (N = BS * grad_accum)
        all_product_embs: [N, D] normalized product embeddings
        temperature: softmax temperature

    The positives are on the diagonal (index i ↔ i).
    """
    logits = (all_query_embs @ all_product_embs.T) / temperature  # [N, N]
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


@torch.inference_mode()
def compute_recall_at_15(
    model: nn.Module,
    val_pairs: Sequence[Pair],
    bundle_image_map: Dict[str, Path],
    product_image_map: Dict[str, Path],
    preprocess: Any,
    device: torch.device,
    amp: bool,
    batch_size: int,
    num_workers: int,
) -> float:
    """Recall@15 over val pairs: encode unique bundles & products, rank by cosine sim.

    Uses a DataLoader with multiple workers for fast parallel image loading.
    """
    if not val_pairs:
        return float("nan")
    model.eval()
    amp_on = amp and device.type == "cuda"

    gt: Dict[str, Set[str]] = defaultdict(set)
    for bid, pid in val_pairs:
        gt[bid].add(pid)

    # --- fast batch encoder using DataLoader with workers ---
    class _IdImageDataset(Dataset):
        def __init__(self, ids: List[str], img_map: Dict[str, Path], tfm: Any):
            self.ids = ids
            self.img_map = img_map
            self.tfm = tfm

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, idx: int) -> Tuple[int, Optional[torch.Tensor]]:
            aid = self.ids[idx]
            img = open_image_safe(self.img_map[aid])
            if img is None:
                return idx, None
            return idx, self.tfm(img)

    def _collate_id(batch):
        idxs, tensors = [], []
        for i, t in batch:
            if t is not None:
                idxs.append(i)
                tensors.append(t)
        if not tensors:
            return None
        return idxs, torch.stack(tensors)

    def _encode_ids(ids: List[str], img_map: Dict[str, Path]) -> Tuple[List[str], torch.Tensor]:
        ds = _IdImageDataset(ids, img_map, preprocess)
        loader = DataLoader(
            ds,
            batch_size=batch_size * 2,  # inference only → can use larger BS
            shuffle=False,
            num_workers=min(num_workers, 8),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate_id,
        )
        all_embs = torch.zeros(len(ids), 0)  # placeholder
        emb_list: List[Tuple[int, torch.Tensor]] = []
        for batch_data in loader:
            if batch_data is None:
                continue
            idxs, imgs_t = batch_data
            imgs_t = imgs_t.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                out = model(imgs_t)
                feats = _extract_features_from_outputs(out).float()
                feats = F.normalize(feats, p=2, dim=1).cpu()
            for j, orig_idx in enumerate(idxs):
                emb_list.append((orig_idx, feats[j]))
        if not emb_list:
            return [], torch.empty(0)
        # Sort by original index to preserve ordering
        emb_list.sort(key=lambda x: x[0])
        valid_ids = [ids[i] for i, _ in emb_list]
        embs = torch.stack([e for _, e in emb_list])
        return valid_ids, embs

    bundle_ids, bundle_embs = _encode_ids(sorted(gt.keys()), bundle_image_map)
    product_list = sorted({pid for _, pid in val_pairs})
    product_ids, product_embs = _encode_ids(product_list, product_image_map)

    if not bundle_ids or not product_ids:
        return float("nan")

    # Vectorized recall computation
    # sims: [num_bundles, num_products]
    sims = bundle_embs @ product_embs.T
    top15_indices = sims.topk(min(15, len(product_ids)), dim=1).indices  # [num_bundles, 15]

    recall_sum, n = 0.0, 0
    for i, bid in enumerate(bundle_ids):
        gt_pids = gt.get(bid, set())
        if not gt_pids:
            continue
        hits = sum(1 for idx in top15_indices[i].tolist() if product_ids[idx] in gt_pids)
        recall_sum += hits / len(gt_pids)
        n += 1
    return recall_sum / n if n else float("nan")


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
    """Train/fine-tune GR-Lite on bundle-product positives.

    Improvements integrated:
      A) Cross-batch InfoNCE – embeddings from grad_accum micro-batches are
         accumulated so the contrastive loss sees BS*grad_accum negatives.
      B) LoRA – when params.use_lora=True, peft LoRA adapters are injected
         into backbone attention layers instead of freeze+unfreeze.
      C) Hard-negative mining – when params.use_hard_negatives=True, a
         CategoryBatchSampler groups same-product_description pairs in each
         batch (requires data/product_dataset.csv).
    """
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
    use_lora = bool(getattr(params, "use_lora", False))
    use_hard_negatives = bool(getattr(params, "use_hard_negatives", False))

    if input_size <= 0:
        raise ValueError("params.grlite_input_size must be > 0")
    if feature_dim <= 0:
        raise ValueError("params.grlite_feature_dim must be > 0")
    if temperature <= 0:
        raise ValueError("params.grlite_temperature must be > 0")

    print(f"Loading GR-Lite model from: {model_name}")
    base_model = load_grlite_base_model(model_name=model_name, device=device, input_size=input_size)

    # Enable gradient checkpointing at model level (best-effort)
    _gc_enabled = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        try:
            base_model.gradient_checkpointing_enable()
            _gc_enabled = True
            print("Enabled HF gradient checkpointing on GR-Lite backbone.")
        except (ValueError, NotImplementedError) as exc:
            print(f"Model does not support gradient_checkpointing_enable ({exc}). Skipping.")
    if not _gc_enabled and hasattr(base_model, "set_grad_checkpointing"):
        try:
            base_model.set_grad_checkpointing(True)
            _gc_enabled = True
            print("Enabled grad checkpointing via set_grad_checkpointing.")
        except (ValueError, NotImplementedError) as exc:
            print(f"set_grad_checkpointing also failed ({exc}). Continuing without gradient checkpointing.")

    # ---------- Fine-tuning strategy ----------
    freeze_backbone = bool(getattr(params, "freeze_backbone", True))
    unfreeze_last_n = int(getattr(params, "unfreeze_last_n_blocks", 2))
    proj_hidden_dim = int(getattr(params, "proj_hidden_dim", 512))

    # Determine embedding dimension from a dummy forward
    base_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        try:
            dummy_out = base_model(dummy)
        except Exception:
            if hasattr(base_model, "model"):
                dummy_out = base_model.model(dummy)
            else:
                raise
        emb_dim = _extract_features_from_outputs(dummy_out).shape[-1]
    base_model.train()

    proj_head: Optional[nn.Module] = None

    if use_lora:
        # B) LoRA: inject adapters – all original params frozen, LoRA deltas trainable
        lora_r = int(getattr(params, "lora_r", 16))
        lora_alpha = int(getattr(params, "lora_alpha", 32))
        lora_dropout = float(getattr(params, "lora_dropout", 0.05))
        base_model = apply_lora_to_model(
            base_model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # Still add a projection head on top of LoRA-adapted features
        proj_head = ProjectionHead(input_dim=emb_dim, hidden_dim=proj_hidden_dim, output_dim=emb_dim).to(device)
        total_trainable = (
            sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            + sum(p.numel() for p in proj_head.parameters())
        )
        total_all = (
            sum(p.numel() for p in base_model.parameters())
            + sum(p.numel() for p in proj_head.parameters())
        )
        print(f"LoRA + proj_head dim={emb_dim}→{proj_hidden_dim}→{emb_dim}")
        print(f"Trainable parameters: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.1f}%)")
    elif freeze_backbone:
        # Original freeze strategy
        frozen_count, trainable_count = freeze_grlite_backbone(base_model, unfreeze_last_n_blocks=unfreeze_last_n)
        proj_head = ProjectionHead(input_dim=emb_dim, hidden_dim=proj_hidden_dim, output_dim=emb_dim).to(device)
        total_trainable = (
            sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            + sum(p.numel() for p in proj_head.parameters())
        )
        total_all = (
            sum(p.numel() for p in base_model.parameters())
            + sum(p.numel() for p in proj_head.parameters())
        )
        print(f"Freeze backbone: ON | proj_head dim={emb_dim}→{proj_hidden_dim}→{emb_dim}")
        print(f"Trainable parameters: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.1f}%)")
    else:
        total_all = sum(p.numel() for p in base_model.parameters())
        print(f"Freeze backbone: OFF | full fine-tuning ({total_all:,} params)")

    model: nn.Module = GRLiteTensorEncoder(base_model, proj_head=proj_head).to(device)
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

    # ---------- C) Hard-negative DataLoader ----------
    if use_hard_negatives:
        data_dir = Path(cfg.files.data_dir) if hasattr(cfg.files, "data_dir") else bundles_images_dir.parent
        product_categories = load_product_categories(data_dir)
        cat_sampler = CategoryBatchSampler(
            pairs=train_pairs,
            product_categories=product_categories,
            batch_size=int(params.batch_size),
            seed=int(params.seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=cat_sampler,
            num_workers=int(params.num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_skip_none,
        )
        print("Hard-negative mining: ON (CategoryBatchSampler)")
    else:
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

    # Only optimise parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Optimizer will update {len(trainable_params)} parameter tensors "
        f"({sum(p.numel() for p in trainable_params):,} scalars)"
    )
    optimizer = AdamW(
        trainable_params,
        lr=float(params.lr),
        weight_decay=float(params.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val_loss = float("inf")
    best_recall_at_15 = 0.0
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

    # Free VRAM before training loop
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # A) Cross-batch InfoNCE: accumulate embeddings across micro-batches
    # ----------------------------------------------------------------
    use_cross_batch = grad_accum > 1
    if use_cross_batch:
        print(
            f"Cross-batch InfoNCE: ON | grad_accum={grad_accum} → "
            f"virtual batch = {int(params.batch_size)} × {grad_accum} = "
            f"{int(params.batch_size) * grad_accum} negatives"
        )
    else:
        print("Cross-batch InfoNCE: OFF (grad_accum=1, standard per-batch loss)")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        running_loss = 0.0
        num_batches = 0

        # Embedding accumulators for cross-batch InfoNCE (memory-efficient)
        accum_query: List[torch.Tensor] = []
        accum_product: List[torch.Tensor] = []
        accum_step = 0  # counts micro-batches in current accumulation window

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            if batch is None:
                continue

            bundle_imgs = batch["bundle"].to(device, non_blocking=True)
            product_imgs = batch["product"].to(device, non_blocking=True)

            if use_cross_batch:
                accum_step += 1
                is_last_accum = (accum_step == grad_accum) or (step == len(train_loader))

                if not is_last_accum:
                    # Intermediate micro-batch: encode WITHOUT grad → no graph retained
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp_enabled):
                        q_emb = encode_images(model, bundle_imgs)  # [B, D] detached
                        p_emb = encode_images(model, product_imgs)
                    accum_query.append(q_emb)
                    accum_product.append(p_emb)
                else:
                    # Last micro-batch: encode WITH grad
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        q_emb = encode_images(model, bundle_imgs)  # [B, D] with grad
                        p_emb = encode_images(model, product_imgs)

                    # Build full virtual batch: detached history + current (with grad)
                    if accum_query:
                        all_q = torch.cat(accum_query + [q_emb], dim=0)  # [N, D]
                        all_p = torch.cat(accum_product + [p_emb], dim=0)
                    else:
                        all_q, all_p = q_emb, p_emb

                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        loss = cross_batch_infonce_loss(all_q, all_p, temperature)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    batch_loss = float(loss.item())
                    running_loss += batch_loss
                    num_batches += 1
                    accum_query.clear()
                    accum_product.clear()
                    accum_step = 0
            else:
                # Standard per-batch InfoNCE (grad_accum=1)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = batch_infonce_loss(
                        model=model,
                        bundle_imgs=bundle_imgs,
                        product_imgs=product_imgs,
                        temperature=temperature,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                batch_loss = float(loss.item())
                running_loss += batch_loss
                num_batches += 1

            if step % log_every == 0:
                progress.set_postfix(loss=f"{running_loss / max(1, num_batches):.4f}")

        train_loss = running_loss / max(1, num_batches)

        val_loss = float("nan")
        val_recall = float("nan")
        if val_loader is not None:
            # Skip evaluate_loss (redundant – recall@15 is the primary metric
            # and encoding images twice doubles validation time)
            val_recall = compute_recall_at_15(
                model=model,
                val_pairs=val_pairs,
                bundle_image_map=bundle_image_map,
                product_image_map=product_image_map,
                preprocess=preprocess,
                device=device,
                amp=amp_enabled,
                batch_size=int(params.batch_size),
                num_workers=int(params.num_workers),
            )

        # Free validation tensors from VRAM
        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "recall@15": float(val_recall),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": float(elapsed),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"recall@15={val_recall:.4f} "
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

        if val_loader is not None and np.isfinite(val_recall) and val_recall > best_recall_at_15:
            best_recall_at_15 = val_recall
            print(f"  ↑ New best recall@15={val_recall:.4f} — saving best.pt")
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
