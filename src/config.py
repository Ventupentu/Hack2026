"""Typed Hydra config for retrieval training."""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class Params:
    """Training/model parameters."""

    batch_size: int = MISSING
    epochs: int = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    seed: int = MISSING
    num_workers: int = MISSING
    device: str = MISSING
    model_name: str = MISSING
    multi_gpu: bool = MISSING
    gpu_ids: str = MISSING
    amp: bool = MISSING
    grad_accum: int = MISSING
    log_every: int = MISSING
    save_every: int = MISSING
    max_val_k: int = MISSING
    recall_k: int = MISSING
    use_bundle_boxes: bool = MISSING
    bbox_model_id: str = MISSING
    bbox_conf_threshold: float = MISSING
    bbox_iou_threshold: float = MISSING
    bbox_max_per_image: int = MISSING
    bbox_min_area_ratio: float = MISSING
    bbox_cache_path: str = MISSING


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
    """Parameters exclusive to the inference/evaluation pipeline."""

    val_ratio: float = MISSING
    eval_ks: str = MISSING
    top_n_submit: int = MISSING


@dataclass
class InditexConfig:
    """Root config."""

    params: Params = field(default_factory=Params)
    files: Files = field(default_factory=Files)
    infer: Infer = field(default_factory=Infer)
