"""Fashion bundle -> product retrieval with GR-Lite (and CLIP fallback).

This script solves the challenge as retrieval:
1) Encode all product images.
2) Encode bundle images.
3) Retrieve top-K products by cosine similarity.
4) Re-rank with train-bundle neighbors (simple supervised prior).
5) Export submission with max 15 products per bundle.

Notes:
- Primary encoder: `srpone/gr-lite` from Hugging Face (if available).
- Fallback encoder: `openai/clip-vit-large-patch14-336`.
- Works even when `data/product_dataset.csv` is missing by using image filenames.
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
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
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


def batched(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1)


class BaseEncoder:
    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class HFClipEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: torch.device, use_amp: bool) -> None:
        try:
            from transformers import AutoProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for CLIP fallback.") from exc

        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name

    def _encode_once(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros((0, 1), dtype=torch.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                feats = self.model.get_image_features(pixel_values=pixel_values)
        return l2_normalize(feats).cpu()

    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        emb = self._encode_once(images)
        if not tta_flip or not images:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        emb_flip = self._encode_once(flipped)
        return l2_normalize((emb + emb_flip) * 0.5)

    @property
    def name(self) -> str:
        return f"HFCLIP({self.model_name})"


class GRLiteEncoder(BaseEncoder):
    """Wrapper for https://huggingface.co/srpone/gr-lite model usage."""

    def __init__(
        self,
        repo_id: str,
        checkpoint_name: str,
        feature_dim: int,
        device: torch.device,
        local_path: Optional[Path] = None,
    ) -> None:
        if local_path is not None and local_path.exists():
            ckpt_path = local_path
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required for GR-Lite loading.") from exc
            ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename=checkpoint_name))
        try:
            # GR-Lite model card loads full checkpoint object with torch.load.
            self.model = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            self.model = torch.load(ckpt_path, map_location=device)

        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.feature_dim = feature_dim
        self.repo_id = repo_id

    def _to_tensor(self, value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().float()
        elif isinstance(value, np.ndarray):
            t = torch.from_numpy(value).float()
        elif isinstance(value, list):
            t = torch.tensor(value, dtype=torch.float32)
        else:
            raise RuntimeError(f"Unsupported GR-Lite vector type: {type(value)}")

        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    def _encode_once(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32)
        if not hasattr(self.model, "search"):
            raise RuntimeError("Loaded GR-Lite checkpoint does not expose `search`.")

        # Model card usage: scores, vectors = model.search(image_paths=[image], feature_dim=1024)
        out = self.model.search(image_paths=images, feature_dim=self.feature_dim)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            vectors = out[1]
        else:
            vectors = out
        return l2_normalize(self._to_tensor(vectors))

    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        emb = self._encode_once(images)
        if not tta_flip or not images:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        emb_flip = self._encode_once(flipped)
        return l2_normalize((emb + emb_flip) * 0.5)

    @property
    def name(self) -> str:
        return f"GRLite({self.repo_id})"


class SharedAdapter(torch.nn.Module):
    """Shared projection head used by GR-Lite full fine-tuning."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.norm = torch.nn.LayerNorm(self.in_dim)
        self.skip = (
            torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
            if self.in_dim != self.out_dim
            else torch.nn.Identity()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        base = self.skip(xn)
        delta = self.mlp(xn)
        return l2_normalize(base + delta)


class GRLiteAdapterCheckpointEncoder(BaseEncoder):
    """Encoder that applies a fine-tuned adapter on top of GR-Lite features."""

    def __init__(self, payload: Dict[str, object], device: torch.device, use_amp: bool, local_path: Optional[Path]) -> None:
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.repo_id = str(payload.get("grlite_repo", "srpone/gr-lite"))
        self.checkpoint_name = str(payload.get("grlite_checkpoint", "gr_lite.pt"))
        self.feature_dim = int(payload.get("grlite_dim", 1024))
        self.adapter_in_dim = int(payload.get("adapter_in_dim", self.feature_dim))
        self.adapter_hidden_dim = int(payload.get("adapter_hidden_dim", 1536))
        self.adapter_out_dim = int(payload.get("adapter_out_dim", self.feature_dim))
        self.adapter_dropout = float(payload.get("adapter_dropout", 0.1))

        self.base_encoder = GRLiteEncoder(
            repo_id=self.repo_id,
            checkpoint_name=self.checkpoint_name,
            feature_dim=self.feature_dim,
            device=device,
            local_path=local_path,
        )
        adapter_state = payload.get("adapter_state_dict")
        if not isinstance(adapter_state, dict):
            raise RuntimeError("Invalid GR-Lite adapter checkpoint: missing `adapter_state_dict`.")
        self.adapter = SharedAdapter(
            in_dim=self.adapter_in_dim,
            hidden_dim=self.adapter_hidden_dim,
            out_dim=self.adapter_out_dim,
            dropout=self.adapter_dropout,
        )
        missing, unexpected = self.adapter.load_state_dict(adapter_state, strict=False)
        if missing:
            print(f"Warning: GR-Lite adapter missing keys: {len(missing)}")
        if unexpected:
            print(f"Warning: GR-Lite adapter unexpected keys: {len(unexpected)}")
        self.adapter = self.adapter.to(device).eval()

    def _encode_once(self, images: List[Image.Image]) -> torch.Tensor:
        base = self.base_encoder._encode_once(images)
        if base.numel() == 0:
            return torch.zeros((0, self.adapter_out_dim), dtype=torch.float32)
        x = base.to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                proj = self.adapter(x)
        return l2_normalize(proj).cpu()

    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        emb = self._encode_once(images)
        if not tta_flip or not images:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        emb_flip = self._encode_once(flipped)
        return l2_normalize((emb + emb_flip) * 0.5)

    @property
    def name(self) -> str:
        return f"GRLiteAdapter({self.repo_id})"


class TorchvisionEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: torch.device, use_amp: bool) -> None:
        try:
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError("torchvision is required for offline fallback encoder.") from exc

        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.model_name = model_name

        # ResNet50 gives stable embeddings with minimal dependencies.
        weights = None
        if model_name == "resnet50":
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None
            try:
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(weights=None)
            model.fc = torch.nn.Identity()
            self.transform = (
                weights.transforms() if weights is not None
                else transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
            )
        else:
            raise RuntimeError(f"Unsupported torchvision fallback model: {model_name}")

        self.model = model.to(device).eval()

    def _encode_once(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros((0, 1), dtype=torch.float32)
        batch = torch.stack([self.transform(im) for im in images], dim=0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                feats = self.model(batch)
        return l2_normalize(feats).cpu()

    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        emb = self._encode_once(images)
        if not tta_flip or not images:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        emb_flip = self._encode_once(flipped)
        return l2_normalize((emb + emb_flip) * 0.5)

    @property
    def name(self) -> str:
        return f"Torchvision({self.model_name})"


class FineTunedCheckpointEncoder(BaseEncoder):
    """Load generic image-encoder checkpoints (CLIP/torchvision format)."""

    def __init__(self, checkpoint_path: Path, device: torch.device, use_amp: bool) -> None:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Invalid fine-tuned checkpoint format: {checkpoint_path}")

        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.backbone = str(ckpt.get("backbone", ""))
        self.model_name = str(ckpt.get("model_name", "unknown"))
        self.image_size = int(ckpt.get("image_size", 224))
        self.image_mean = list(ckpt.get("image_mean", [0.485, 0.456, 0.406]))
        self.image_std = list(ckpt.get("image_std", [0.229, 0.224, 0.225]))
        self.embed_dim = int(ckpt.get("embed_dim", 768))
        state_dict = ckpt.get("state_dict")
        if state_dict is None:
            raise RuntimeError("Fine-tuned checkpoint missing `state_dict`.")

        if self.backbone == "clip":
            self.model = self._build_clip_model(ckpt)
            self.mode = "clip"
        elif self.backbone == "torchvision_resnet50":
            self.model = self._build_torchvision_model()
            self.mode = "torchvision_resnet50"
        else:
            raise RuntimeError(f"Unsupported fine-tuned backbone: {self.backbone}")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading fine-tuned checkpoint: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys when loading fine-tuned checkpoint: {len(unexpected)}")
        self.model = self.model.to(device).eval()

        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        self.checkpoint_path = checkpoint_path

    def _build_clip_model(self, ckpt: Dict[str, object]) -> torch.nn.Module:
        from transformers import CLIPConfig, CLIPModel

        clip_cfg = ckpt.get("clip_config")
        if isinstance(clip_cfg, dict):
            return CLIPModel(CLIPConfig.from_dict(clip_cfg))
        # Fallback to remote/local model id.
        model_name = self.model_name or "openai/clip-vit-base-patch32"
        return CLIPModel.from_pretrained(model_name)

    def _build_torchvision_model(self) -> torch.nn.Module:
        from torchvision import models

        backbone = models.resnet50(weights=None)
        in_features = int(backbone.fc.in_features)
        backbone.fc = torch.nn.Identity()
        projection = torch.nn.Linear(in_features, self.embed_dim)
        return torch.nn.ModuleDict({"backbone": backbone, "projection": projection})

    def _encode_once(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros((0, self.embed_dim), dtype=torch.float32)
        batch = torch.stack([self.transform(im) for im in images], dim=0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.mode == "clip":
                    feats = self.model.get_image_features(pixel_values=batch)
                else:
                    backbone = self.model["backbone"]
                    projection = self.model["projection"]
                    feats = projection(backbone(batch))
        return l2_normalize(feats).cpu()

    def encode_images(self, images: List[Image.Image], tta_flip: bool) -> torch.Tensor:
        emb = self._encode_once(images)
        if not tta_flip or not images:
            return emb
        flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in images]
        emb_flip = self._encode_once(flipped)
        return l2_normalize((emb + emb_flip) * 0.5)

    @property
    def name(self) -> str:
        return f"FineTuned({self.backbone})"


def _load_payload(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return None


def create_encoder(
    finetuned_checkpoint: Optional[Path],
    grlite_local_path: Optional[Path],
    use_grlite: bool,
    grlite_repo: str,
    grlite_ckpt: str,
    grlite_dim: int,
    fallback_clip: str,
    fallback_torchvision: str,
    device: torch.device,
    use_amp: bool,
) -> BaseEncoder:
    if finetuned_checkpoint is not None and finetuned_checkpoint.exists():
        try:
            payload = _load_payload(finetuned_checkpoint)
            if payload is None:
                raise RuntimeError("Checkpoint payload is not a dictionary.")
            if str(payload.get("backbone", "")) == "grlite_adapter":
                enc = GRLiteAdapterCheckpointEncoder(
                    payload=payload,
                    device=device,
                    use_amp=use_amp,
                    local_path=grlite_local_path,
                )
            else:
                enc = FineTunedCheckpointEncoder(
                    checkpoint_path=finetuned_checkpoint,
                    device=device,
                    use_amp=use_amp,
                )
            print(f"Loaded fine-tuned encoder: {enc.name} from {finetuned_checkpoint}")
            return enc
        except Exception as exc:
            print(f"Warning: failed loading fine-tuned checkpoint ({exc}). Using fallback stack.")

    if use_grlite:
        try:
            enc = GRLiteEncoder(
                repo_id=grlite_repo,
                checkpoint_name=grlite_ckpt,
                feature_dim=grlite_dim,
                device=device,
                local_path=grlite_local_path,
            )
            print(f"Loaded primary encoder: {enc.name}")
            return enc
        except Exception as exc:
            print(f"Warning: GR-Lite load failed ({exc}). Falling back to CLIP.")
    try:
        enc = HFClipEncoder(model_name=fallback_clip, device=device, use_amp=use_amp)
        print(f"Loaded fallback encoder: {enc.name}")
        return enc
    except Exception as exc:
        print(f"Warning: CLIP fallback load failed ({exc}). Falling back to torchvision.")
    enc = TorchvisionEncoder(model_name=fallback_torchvision, device=device, use_amp=use_amp)
    print(f"Loaded final fallback encoder: {enc.name}")
    return enc


def encode_ids(
    ids: Sequence[str],
    image_map: Dict[str, Path],
    encoder: BaseEncoder,
    batch_size: int,
    tta_flip: bool,
    cache_prefix: Optional[Path] = None,
    max_items: int = 0,
) -> Tuple[List[str], torch.Tensor]:
    valid_ids = [x for x in ids if x in image_map]
    if max_items > 0:
        valid_ids = valid_ids[:max_items]

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
        desc=f"Encoding {len(valid_ids)} images",
    ):
        images: List[Image.Image] = []
        keep_ids: List[str] = []
        for asset_id in chunk_ids:
            img = open_image_safe(image_map[asset_id])
            if img is None:
                continue
            images.append(img)
            keep_ids.append(asset_id)

        if not images:
            continue
        emb = encoder.encode_images(images=images, tta_flip=tta_flip)
        if emb.ndim != 2 or emb.shape[0] != len(keep_ids):
            raise RuntimeError(
                f"Embedding shape mismatch for {encoder.name}: "
                f"got {tuple(emb.shape)}, expected ({len(keep_ids)}, dim)."
            )
        out_ids.extend(keep_ids)
        out_embs.append(emb.cpu())

    if not out_embs:
        return [], torch.zeros((0, 1), dtype=torch.float32)

    all_embs = torch.cat(out_embs, dim=0).float()

    if cache_prefix is not None:
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        ids_path = cache_prefix.with_suffix(".ids.txt")
        emb_path = cache_prefix.with_suffix(".emb.npy")
        ids_path.write_text("\n".join(out_ids) + "\n", encoding="utf-8")
        np.save(emb_path, all_embs.numpy())
        print(f"Saved cache: {emb_path}")

    return out_ids, all_embs


def split_train_val_bundles(bundle_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    uniq = list(dict.fromkeys(bundle_ids))
    if val_ratio <= 0.0:
        return set(uniq), set()
    rng = random.Random(seed)
    rng.shuffle(uniq)
    val_n = max(1, int(round(len(uniq) * val_ratio)))
    val_set = set(uniq[:val_n])
    train_set = set(uniq[val_n:])
    return train_set, val_set


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
    bundle_ids = [bid for bid in gt_map if bid in pred_map]
    if not bundle_ids:
        return {}
    metrics: Dict[str, float] = {}
    for k in ks:
        hits: List[float] = []
        recalls: List[float] = []
        for bid in bundle_ids:
            preds = pred_map[bid][:k]
            gt = gt_map[bid]
            if not gt:
                continue
            inter = len(set(preds) & gt)
            hits.append(1.0 if inter > 0 else 0.0)
            recalls.append(inter / float(len(gt)))
        if hits:
            metrics[f"hit@{k}"] = float(np.mean(hits))
        if recalls:
            metrics[f"recall@{k}"] = float(np.mean(recalls))
    return metrics


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def rank_for_queries(
    query_ids: Sequence[str],
    query_emb: torch.Tensor,
    product_ids: Sequence[str],
    product_emb: torch.Tensor,
    train_bundle_ids: Sequence[str],
    train_bundle_emb: Optional[torch.Tensor],
    train_bundle_gt_map: Dict[str, Set[str]],
    top_n: int,
    candidate_k: int,
    neighbor_k: int,
    prior_weight: float,
    use_prior: bool,
    fallback_products: Sequence[str],
) -> Dict[str, List[str]]:
    if query_emb.numel() == 0 or product_emb.numel() == 0:
        return {bid: list(fallback_products[:top_n]) for bid in query_ids}

    product_emb = l2_normalize(product_emb)
    query_emb = l2_normalize(query_emb)
    if train_bundle_emb is not None and train_bundle_emb.numel() > 0:
        train_bundle_emb = l2_normalize(train_bundle_emb)

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    results: Dict[str, List[str]] = {}

    all_scores = query_emb @ product_emb.T  # [Q, P]

    for i, bundle_id in enumerate(query_ids):
        base_scores = all_scores[i]  # [P]
        base_k = min(candidate_k, base_scores.shape[0])
        top_idx = torch.topk(base_scores, k=base_k).indices.tolist()
        candidates = {int(x) for x in top_idx}

        prior_scores: Dict[str, float] = {}
        if use_prior and train_bundle_emb is not None and train_bundle_emb.shape[0] > 0:
            neigh_scores = torch.mv(train_bundle_emb, query_emb[i])
            k_nb = min(neighbor_k, neigh_scores.shape[0])
            nb_vals, nb_idx = torch.topk(neigh_scores, k=k_nb)
            raw_prior: Dict[str, float] = defaultdict(float)
            for score_t, idx_t in zip(nb_vals.tolist(), nb_idx.tolist()):
                if score_t <= 0.0:
                    continue
                nb_bundle_id = train_bundle_ids[idx_t]
                for pid in train_bundle_gt_map.get(nb_bundle_id, ()):
                    raw_prior[pid] += score_t

            if raw_prior:
                max_val = max(raw_prior.values())
                if max_val > 0:
                    prior_scores = {pid: val / max_val for pid, val in raw_prior.items()}
                else:
                    prior_scores = dict(raw_prior)

                # Add top prior-only candidates that might not be in image top-k.
                prior_sorted = sorted(prior_scores.items(), key=lambda x: x[1], reverse=True)
                for pid, _ in prior_sorted[: candidate_k // 2]:
                    idx = pid_to_idx.get(pid)
                    if idx is not None:
                        candidates.add(idx)

        merged: List[Tuple[str, float]] = []
        for pidx in candidates:
            pid = product_ids[pidx]
            score = float(base_scores[pidx].item())
            if prior_scores:
                score += prior_weight * prior_scores.get(pid, 0.0)
            merged.append((pid, score))

        merged.sort(key=lambda x: x[1], reverse=True)
        pred_ids = unique_keep_order(pid for pid, _ in merged)[:top_n]

        if len(pred_ids) < top_n:
            for pid in fallback_products:
                if pid in pred_ids:
                    continue
                pred_ids.append(pid)
                if len(pred_ids) >= top_n:
                    break
        results[bundle_id] = pred_ids[:top_n]

    return results


def save_submission(bundle_ids: Sequence[str], pred_map: Dict[str, List[str]], output_csv: Path) -> None:
    rows = []
    for bundle_id in bundle_ids:
        preds = pred_map.get(bundle_id, [])
        for pid in preds:
            rows.append({"bundle_asset_id": bundle_id, "product_asset_id": pid})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fashion retrieval with GR-Lite + simple supervised re-ranking.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/bundles_product_match_train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/bundles_product_match_test.csv"))
    parser.add_argument("--bundle-images-dir", type=Path, default=Path("data/bundle_images"))
    parser.add_argument("--product-images-dir", type=Path, default=Path("data/product_images"))
    parser.add_argument("--submission-out", type=Path, default=Path("outputs/test_submission_grlite.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/val_metrics_grlite.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/cache"))
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint (.pt); supports GR-Lite adapter checkpoints from src/fashion_grlite_finetune.py.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--candidate-k", type=int, default=2500)
    parser.add_argument("--neighbor-k", type=int, default=40)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(use_prior=True, tta_flip=True, use_amp=True, use_grlite=True)
    parser.add_argument("--use-prior", action="store_true", dest="use_prior")
    parser.add_argument("--no-prior", action="store_false", dest="use_prior")
    parser.add_argument("--tta-flip", action="store_true", dest="tta_flip")
    parser.add_argument("--no-tta-flip", action="store_false", dest="tta_flip")
    parser.add_argument("--use-amp", action="store_true", dest="use_amp")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")

    parser.add_argument("--use-grlite", action="store_true", dest="use_grlite")
    parser.add_argument("--no-grlite", action="store_false", dest="use_grlite")
    parser.add_argument("--grlite-repo", type=str, default="srpone/gr-lite")
    parser.add_argument("--grlite-checkpoint", type=str, default="gr_lite.pt")
    parser.add_argument("--grlite-dim", type=int, default=1024)
    parser.add_argument("--grlite-local-path", type=Path, default=None, help="Optional local GR-Lite checkpoint path.")
    parser.add_argument("--fallback-clip-model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--fallback-torchvision-model", type=str, default="resnet50")

    parser.add_argument("--max-products", type=int, default=0, help="Debug only: cap product count.")
    parser.add_argument("--max-test-bundles", type=int, default=0, help="Debug only: cap test bundle count.")
    parser.add_argument("--max-train-bundles", type=int, default=0, help="Debug only: cap train bundle count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.top_n > 15:
        print("Warning: top_n > 15 is not valid for evaluation. Capping to 15.")
        args.top_n = 15

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    bundle_image_map = build_image_map(args.bundle_images_dir)
    product_image_map = build_image_map(args.product_images_dir)

    # Candidate products: all product images available (more coverage than train-only product IDs).
    product_ids = sorted(product_image_map.keys())
    train_bundle_ids_all = train_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()
    test_bundle_ids_all = test_df["bundle_asset_id"].astype(str).drop_duplicates().tolist()

    if args.max_products > 0:
        product_ids = product_ids[: args.max_products]
    if args.max_train_bundles > 0:
        train_bundle_ids_all = train_bundle_ids_all[: args.max_train_bundles]
    if args.max_test_bundles > 0:
        test_bundle_ids_all = test_bundle_ids_all[: args.max_test_bundles]

    print(f"Products candidates: {len(product_ids)}")
    print(f"Train bundles: {len(train_bundle_ids_all)}")
    print(f"Test bundles: {len(test_bundle_ids_all)}")

    encoder = create_encoder(
        finetuned_checkpoint=args.finetuned_checkpoint,
        grlite_local_path=args.grlite_local_path,
        use_grlite=args.use_grlite,
        grlite_repo=args.grlite_repo,
        grlite_ckpt=args.grlite_checkpoint,
        grlite_dim=args.grlite_dim,
        fallback_clip=args.fallback_clip_model,
        fallback_torchvision=args.fallback_torchvision_model,
        device=device,
        use_amp=args.use_amp,
    )

    model_tag = encoder.name.replace("/", "_").replace("(", "_").replace(")", "").replace(" ", "")
    product_cache = args.cache_dir / f"products_{model_tag}"
    train_bundle_cache = args.cache_dir / f"train_bundles_{model_tag}"
    test_bundle_cache = args.cache_dir / f"test_bundles_{model_tag}"

    product_ids_enc, product_emb = encode_ids(
        ids=product_ids,
        image_map=product_image_map,
        encoder=encoder,
        batch_size=args.batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=product_cache,
        max_items=0,
    )
    train_bundle_ids_enc, train_bundle_emb = encode_ids(
        ids=train_bundle_ids_all,
        image_map=bundle_image_map,
        encoder=encoder,
        batch_size=args.batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=train_bundle_cache,
        max_items=0,
    )
    test_bundle_ids_enc, test_bundle_emb = encode_ids(
        ids=test_bundle_ids_all,
        image_map=bundle_image_map,
        encoder=encoder,
        batch_size=args.batch_size,
        tta_flip=args.tta_flip,
        cache_prefix=test_bundle_cache,
        max_items=0,
    )

    gt_map = build_gt_map(train_df)
    fallback_products = [pid for pid, _ in Counter(train_df["product_asset_id"].astype(str)).most_common(args.top_n)]
    fallback_products = unique_keep_order(fallback_products + list(product_ids_enc))[: max(args.top_n, 100)]

    # Validation split on train bundles (no leakage in prior index).
    train_prior_set, val_set = split_train_val_bundles(train_bundle_ids_enc, args.val_ratio, args.seed)
    val_ids = [bid for bid in train_bundle_ids_enc if bid in val_set]
    prior_ids_for_val = [bid for bid in train_bundle_ids_enc if bid in train_prior_set]
    idx_by_train_bundle = {bid: i for i, bid in enumerate(train_bundle_ids_enc)}
    prior_idx_val = [idx_by_train_bundle[bid] for bid in prior_ids_for_val]
    val_idx = [idx_by_train_bundle[bid] for bid in val_ids]

    metrics: Dict[str, float] = {}
    if val_ids:
        val_pred = rank_for_queries(
            query_ids=val_ids,
            query_emb=train_bundle_emb[val_idx],
            product_ids=product_ids_enc,
            product_emb=product_emb,
            train_bundle_ids=prior_ids_for_val,
            train_bundle_emb=train_bundle_emb[prior_idx_val],
            train_bundle_gt_map={bid: gt_map.get(bid, set()) for bid in prior_ids_for_val},
            top_n=args.top_n,
            candidate_k=args.candidate_k,
            neighbor_k=args.neighbor_k,
            prior_weight=args.prior_weight,
            use_prior=args.use_prior,
            fallback_products=fallback_products,
        )
        metrics = evaluate_predictions(
            pred_map=val_pred,
            gt_map={bid: gt_map[bid] for bid in val_ids if bid in gt_map},
            ks=(5, 10, 15),
        )
        print("Validation metrics:", metrics)
    else:
        print("Validation skipped (val split empty).")

    # Final inference for test uses all train bundles as prior source.
    test_pred = rank_for_queries(
        query_ids=test_bundle_ids_enc,
        query_emb=test_bundle_emb,
        product_ids=product_ids_enc,
        product_emb=product_emb,
        train_bundle_ids=train_bundle_ids_enc,
        train_bundle_emb=train_bundle_emb,
        train_bundle_gt_map={bid: gt_map.get(bid, set()) for bid in train_bundle_ids_enc},
        top_n=args.top_n,
        candidate_k=args.candidate_k,
        neighbor_k=args.neighbor_k,
        prior_weight=args.prior_weight,
        use_prior=args.use_prior,
        fallback_products=fallback_products,
    )

    # Ensure all bundles from template exist in output.
    final_pred: Dict[str, List[str]] = {}
    for bid in test_bundle_ids_all:
        preds = test_pred.get(bid, [])[: args.top_n]
        if len(preds) < args.top_n:
            for pid in fallback_products:
                if pid in preds:
                    continue
                preds.append(pid)
                if len(preds) >= args.top_n:
                    break
        final_pred[bid] = preds[: args.top_n]

    save_submission(bundle_ids=test_bundle_ids_all, pred_map=final_pred, output_csv=args.submission_out)
    print(f"Saved submission: {args.submission_out}")

    payload = {
        "encoder": encoder.name,
        "num_products": len(product_ids_enc),
        "num_train_bundles_encoded": len(train_bundle_ids_enc),
        "num_test_bundles_encoded": len(test_bundle_ids_enc),
        "top_n": args.top_n,
        "candidate_k": args.candidate_k,
        "neighbor_k": args.neighbor_k,
        "prior_weight": args.prior_weight,
        "validation": metrics,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
