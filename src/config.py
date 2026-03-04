"""Typed Hydra config for retrieval training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, MISSING


@dataclass
class Params:
    """Training/model parameters."""

    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    model_name: str = "openclip_marqo_siglip"
    multi_gpu: bool = False
    gpu_ids: str = "0,1"
    amp: bool = True
    grad_accum: int = 1
    grad_checkpointing: bool = False
    log_every: int = 50
    save_every: int = 1
    max_val_k: int = 200
    recall_k: int = 15
    use_soft_targets: bool = True
    learnable_temperature: bool = True
    use_bundle_boxes: bool = True
    bbox_model_id: str = "kesimeg/yolov8n-clothing-detection"
    bbox_conf_threshold: float = 0.25
    bbox_iou_threshold: float = 0.45
    bbox_max_per_image: int = 15
    bbox_min_area_ratio: float = 0.001
    bbox_cache_path: str = ""
    # Hard negative mining
    mine_every: int = 3
    hard_neg_top_k: int = 16
    max_hard_negatives: int = 4
    max_positives: int = 8
    # Resume training
    resume_from: Optional[str] = None


@dataclass
class Files:
    """Paths used across scripts. Only paths that vary per environment are listed here.
    Derived paths (specific CSVs, output filenames) are built in code from these roots."""

    data_dir: str = MISSING
    bundles_images: str = MISSING
    products_images: str = MISSING
    yolo_detections_dir: str = MISSING


@dataclass
class Infer:
    """Inference parameters."""

    checkpoint_path: str = ""
    tta_num_augs: int = 1
    per_crop_topk: int = 50
    top_n_submit: int = 15
    val_ratio: float = 0.1
    eval_ks: str = "5,10,15"
    # Gender filtering: discard products whose known gender ≠ bundle section
    gender_filter: bool = True
    # Category diversity: max products per product_description category
    max_per_category: int = 2
    # Score threshold: discard products below this cosine similarity
    score_threshold: float = 0.0
    # Timestamp-aware rerank (score += adjustment_from_delta_ts)
    ts_rerank_enabled: bool = False
    ts_delta_weight: float = 0.0
    ts_decay_hours: float = 720.0
    ts_bonus_same_date: float = 0.0
    ts_bonus_same_month: float = 0.0
    ts_bonus_same_quarter: float = 0.0
    ts_penalty_diff_quarter: float = 0.0
    # Post-processing Rerankers
    rerank_hubness_enabled: bool = False
    rerank_hubness_max_ratio: float = 0.015
    rerank_hubness_penalty: float = 0.1
    rerank_heavy_enabled: bool = False
    rerank_heavy_model: str = "ViT-SO400M-14-SigLIP-384"
    rerank_heavy_pretrained: str = "webli"
    rerank_heavy_weight: float = 0.4
    # Lightweight trained MLP reranker on top retrieved candidates
    rerank_mlp_enabled: bool = False
    rerank_mlp_checkpoint: str = ""
    # < 0 => use blend_alpha saved in checkpoint
    rerank_mlp_blend_alpha: float = -1.0
    rerank_mlp_batch_size: int = 4096
    # Candidate pool before final top-15 truncation
    rerank_mlp_candidate_pool: int = 200
    # Optional: export one train bundle embedding per bundle for MLP training.
    # Aggregates multiple detected boxes with confidence-weighted mean.
    export_train_bundle_embeddings: bool = False
    train_bundle_embeddings_out: str = "outputs/train_bundle_embeddings.pt"


@dataclass
class HuggingFace:
    """Hugging Face Hub sync settings for training artifacts."""

    push_to_hub: bool = False
    hf_repo_id: str = ""
    hf_token: str = ""


@dataclass
class InditexConfig:
    """Root config."""

    params: Params = field(default_factory=Params)
    files: Files = field(default_factory=Files)
    infer: "Infer" = field(default_factory=lambda: Infer())
    hf: HuggingFace = field(default_factory=HuggingFace)
