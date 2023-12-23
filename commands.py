import fire
import hydra
from hydra.core.config_store import ConfigStore

from dltoolkit.config import Params
from dltoolkit.infer import inference
from dltoolkit.mlflow_server import run_mlflow_server
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


def run_server(model_uri, image_path=None):
    if image_path is None:
        # image_path = 'C:\code\dltoolkit\data\Stanford_Dogs_256\n02085936-Maltese_dog\n02085936_37.jpg'
        image_path = "data/Stanford_Dogs_256/n02085936-Maltese_dog/n02085936_37.jpg"
    run_mlflow_server(model_uri, image_path)


if __name__ == "__main__":
    fire.Fire()
