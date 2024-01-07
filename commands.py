import fire
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from triton_server.client import run_client as triton_run_client

from dltoolkit.config import Params
from dltoolkit.infer import inference
from dltoolkit.run_server import run_server as run_mlflow_server
from dltoolkit.train import train_model


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


def train(
    config_name: str = "config",
    config_path: str = "configs",
    job_name: str = "train_model",
):
    hydra.initialize(version_base="1.3", config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name)
    train_model(model_cfg=cfg.model, dataset_cfg=cfg.dataset, train_cfg=cfg.train)


def infer(
    config_name: str = "config",
    config_path: str = "configs",
    job_name: str = "test_model",
):
    hydra.initialize(version_base="1.3", config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name)
    inference(model_cfg=cfg.model, dataset_cfg=cfg.dataset, infer_cfg=cfg.infer)


def run_server(
    image_path=None,
    config_name: str = "config",
    config_path: str = "configs",
    job_name: str = "run_mlflow_server",
):
    hydra.initialize(version_base="1.3", config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name)

    if image_path is None:
        image_path = "data/Stanford_Dogs_256/n02085936-Maltese_dog/n02085936_37.jpg"

    model_uri_path = cfg.train.mlflow_model_uri_file
    with open(model_uri_path, "r") as f:
        model_uri = f.read()

    class_num_dict = pd.read_csv(cfg.infer.class_num_dict_path)["class"].to_dict()

    print(f"Using model_uri: {model_uri} for MLFlow server")
    run_mlflow_server(model_uri, class_num_dict, image_path)


def triton_client(
    config_name: str = "client_config",
    config_path: str = "configs",
    job_name: str = "run_triton_client",
):
    hydra.initialize(version_base="1.3", config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name)
    triton_run_client(cfg)


if __name__ == "__main__":
    fire.Fire()
