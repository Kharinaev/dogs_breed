from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    num_classes: int
    model_path: str


@dataclass
class DatasetConfig:
    dataset_path: str
    csv_path: str
    batch_size: int
    image_size: list


@dataclass
class TrainConfig:
    num_epochs: int
    learning_rate: float
    use_fp16: bool
    export_onnx: Optional[str]
    experiment_name: str
    run_name: str


@dataclass
class InferConfig:
    csv_output_save_path: str
    accuracy_topk: list


@dataclass
class Params:
    model: ModelConfig
    dataset: DatasetConfig
    train: TrainConfig
    infer: InferConfig
