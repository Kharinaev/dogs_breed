from dataclasses import dataclass


@dataclass
class DatasetConfig:
    dataset_path: str
    csv_path: str
    batch_size: int


@dataclass
class TrainConfig:
    num_epochs: int
    learning_rate: float
    model_save_path: str
    experiment_name: str
    run_name: str
    tracking_uri: str
    registry_uri: str


@dataclass
class InferConfig:
    model_load_path: str
    csv_output_save_path: str


@dataclass
class Params:
    dataset: DatasetConfig
    train: TrainConfig
