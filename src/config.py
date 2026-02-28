"""Typed Hydra config for retrieval training."""

from dataclasses import dataclass, field

from omegaconf import MISSING


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
    hard_neg_margin: float = 0.2
    hard_neg_weight: float = 0.25


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
    infer: "Infer" = field(default_factory=lambda: Infer())


@dataclass
class Infer:
    """Inference parameters."""

    checkpoint_path: str = ""
    tta_num_augs: int = 1
    per_crop_topk: int = 50
    top_n_submit: int = 15
    val_ratio: float = 0.1
    eval_ks: str = "5,10,15"
