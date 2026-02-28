"""Typed Hydra config for retrieval training."""

from dataclasses import dataclass, field


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
    multi_gpu: bool = True
    gpu_ids: str = "0,1"
    amp: bool = True
    grad_accum: int = 1
    log_every: int = 50
    save_every: int = 1
    max_val_k: int = 200
    recall_k: int = 15


@dataclass
class Files:
    """Paths used by training."""

    train_manifest: str = "data/bundles_product_match_train.csv"
    val_manifest: str = "data/bundles_product_match_train.csv"
    products_manifest: str = "data/product_dataset.csv"
    bundles_images: str = "data/images/bundles"
    products_images: str = "data/images/products"
    output_dir: str = "outputs/retrieval_openclip"


@dataclass
class InditexConfig:
    """Root config."""

    params: Params = field(default_factory=Params)
    files: Files = field(default_factory=Files)
