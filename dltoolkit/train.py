from pathlib import Path

import hydra
import torch
from catalyst import dl

# from catalyst.loggers.mlflow import MLflowLogger
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToTensor


# from tqdm import tqdm


try:
    from dltoolkit.config import DatasetConfig, Params, TrainConfig
    from dltoolkit.dataset import StanfordDogsDataset
    from dltoolkit.model import ResNetClassifier
except ImportError:
    from config import DatasetConfig, Params, TrainConfig
    from dataset import StanfordDogsDataset
    from model import ResNetClassifier


def get_callbacks(num_classes, acc_topk):
    return {
        "criterion": dl.CriterionCallback(
            metric_key="loss", input_key="logits", target_key="targets"
        ),
        "backward": dl.BackwardCallback(metric_key="loss"),
        "optimizer": dl.OptimizerCallback(metric_key="loss"),
        "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
        "accuracy": dl.AccuracyCallback(
            input_key="logits",
            target_key="targets",
            topk=acc_topk,
            num_classes=num_classes,
        ),
        "f1": dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=num_classes
        ),
        # "checkpoint": dl.CheckpointCallback(
        #     self._logdir,
        #     loader_key="valid",
        #     metric_key="accuracy01",
        #     minimize=False,
        #     topk=1,
        # ),
        "tqdm": dl.TqdmCallback(),
    }


def train_catalyst(
    model,
    n_epochs,
    criterion,
    optimizer,
    dataset,
    batch_size,
    logger_dict,
    valid_size=0.2,
):
    indices = list(range(len(dataset)))
    train_indices, valid_indices = random_split(indices, [1 - valid_size, valid_size])
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "valid": DataLoader(valid_dataset, batch_size=batch_size, shuffle=False),
    }

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    runner = dl.SupervisedRunner(
        model=model,
        input_key="features",
        output_key="logits",
        target_key="targets",
        loss_key="loss",
    )
    runner.train(
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=n_epochs,
        valid_loader="valid",
        valid_metric="accuracy03",
        minimize_valid_metric=False,
        verbose=True,
        callbacks=get_callbacks(num_classes=dataset.n_classes, acc_topk=[1, 3, 5]),
        loggers=logger_dict,
    )


def load_model(n_classes: int, device: torch.device):
    model = ResNetClassifier(n_classes, weights=ResNet18_Weights.DEFAULT).to(device)
    model.train()

    return model


def train_model(train_cfg: TrainConfig, dataset_cfg: DatasetConfig):
    print("Preparing dataset")
    train_ds = StanfordDogsDataset(
        "train",
        abs_dvc_repo=Path.cwd(),
        dataset_path=Path(dataset_cfg.dataset_path),
        csv_path=Path(dataset_cfg.csv_path),
        transform=ToTensor(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = load_model(train_ds.n_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    print("Start training")
    logger_dict = None
    # {
    #     "mlflow": MLflowLogger(
    #         experiment=train_cfg.experiment_name,
    #         run=train_cfg.run_name,
    #         tracking_uri=train_cfg.tracking_uri,
    #         registry_uri=train_cfg.registry_uri,
    #         log_batch_metrics=True,
    #         log_epoch_metrics=True,
    #     ),
    # }
    train_catalyst(
        model,
        train_cfg.num_epochs,
        criterion,
        optimizer,
        train_ds,
        dataset_cfg.batch_size,
        logger_dict,
    )
    torch.save(model.state_dict(), train_cfg.model_save_path)
    print(f"Model saved in {train_cfg.model_save_path}")


# @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
# def main(cfg: Params) -> None:
#     train_model(cfg.train, cfg.dataset)


@hydra.main(config_path="../configs", config_name="config_subset", version_base="1.3")
def main(cfg: Params) -> None:
    train_model(cfg.train, cfg.dataset)


if __name__ == "__main__":
    main()
