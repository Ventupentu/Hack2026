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
    train_data: str
    val_data: str
    test_data: str

@dataclass
class InditexConfig:
    # Configuration for the project
    params: Params
    files: Files