import fire
import hydra
from hydra.core.config_store import ConfigStore

from dltoolkit.config import Params
from dltoolkit.infer import inference
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
    train_model(train_cfg=cfg.train, dataset_cfg=cfg.dataset)


def infer(
    config_name: str = "config",
    config_path: str = "configs",
    job_name: str = "test_model",
):
    hydra.initialize(version_base="1.3", config_path=config_path, job_name=job_name)
    cfg = hydra.compose(config_name=config_name)
    inference(infer_cfg=cfg.infer, dataset_cfg=cfg.dataset)


if __name__ == "__main__":
    fire.Fire()
