from dataclasses import dataclass

@dataclass
class Params:
    # Parameters for the training model process
    model: str
    img_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str

@dataclass
class Files:
    # File paths for the training process
    bundles_dataset: str
    bundles_product_match_train: str
    bundles_product_match_test: str
    product_dataset: str
    bundles_images: str
    products_images: str

@dataclass
class InditexConfig:
    # Configuration for the project
    params: Params
    files: Files