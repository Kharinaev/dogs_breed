from pathlib import Path

# from torchvision.transforms import ToTensor
import albumentations as A
import git
import hydra
import mlflow
import onnx
import torch
from albumentations.pytorch import ToTensorV2
from catalyst import dl
from catalyst.loggers.mlflow import MLflowLogger

# from catalyst.loggers.tensorboard import TensorboardLogger
# from catalyst.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.models import ResNet18_Weights


try:
    from dltoolkit.config import DatasetConfig, ModelConfig, Params, TrainConfig
    from dltoolkit.dataset import StanfordDogsDataset
    from dltoolkit.model import ResNetClassifier
except ImportError:
    from config import DatasetConfig, ModelConfig, Params, TrainConfig
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
    num_classes,
    use_fp16,
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
        callbacks=get_callbacks(num_classes=num_classes, acc_topk=[1, 3, 5]),
        loggers=logger_dict,
        fp16=use_fp16,
        load_best_on_end=True,
    )


def get_git_commit_id():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def load_model(n_classes: int):
    model = ResNetClassifier(n_classes, weights=ResNet18_Weights.DEFAULT)
    model.train()

    return model


def model_export_onnx(model, train_cfg: TrainConfig, X):
    torch.onnx.export(
        model,
        X,
        train_cfg.export_onnx,
        export_params=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )
    onnx_model = onnx.load_model(train_cfg.export_onnx)

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(X.numpy(), model(X).detach().numpy())
        model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)
        print(f"MLFLow onnx_model model_uri: {model_info.model_uri}")


def train_model(
    model_cfg: ModelConfig, dataset_cfg: DatasetConfig, train_cfg: TrainConfig
):
    print("Step 1/4: Preparing dataset")
    transform = A.Compose(
        [
            A.Resize(*dataset_cfg.image_size),
            A.HorizontalFlip(p=0.25),
            A.Rotate(limit=30, p=0.1),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.1),
            A.PixelDropout(dropout_prob=0.005, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Perspective(scale=(0.01, 0.05), p=0.1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    train_ds = StanfordDogsDataset(
        "train",
        abs_dvc_repo=Path.cwd(),
        dataset_path=Path(dataset_cfg.dataset_path),
        csv_path=Path(dataset_cfg.csv_path),
        transform=transform,
        transform_type="albumentations",
    )

    print("Step 2/4: Loading model")
    model = load_model(model_cfg.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    print("Step 3/4: Start training")
    logger_dict = {
        # "tensorboard": TensorboardLogger(logdir="./logdir/tensorboard"),
        # "wandb": WandbLogger(project="dltoolkit", name=train_cfg.experiment_name),
        "mlflow": MLflowLogger(
            experiment=train_cfg.experiment_name,
            run=train_cfg.run_name,
            log_batch_metrics=True,
            log_epoch_metrics=True,
        ),
    }

    git_commit_id = get_git_commit_id()
    for _, logger in logger_dict.items():
        logger.log_hparams(dict(model_cfg))
        logger.log_hparams(dict(dataset_cfg))
        logger.log_hparams(dict(train_cfg))
        logger.log_hparams({"git_commit_id": git_commit_id})

    train_catalyst(
        model,
        train_cfg.num_epochs,
        criterion,
        optimizer,
        train_ds,
        dataset_cfg.batch_size,
        logger_dict,
        model_cfg.num_classes,
        use_fp16=train_cfg.use_fp16,
    )

    print("Step 4/4: Saving model")
    torch.save(model.state_dict(), model_cfg.model_path)
    if train_cfg.export_onnx is not None:
        h, w = dataset_cfg.image_size
        dummy_input = torch.randn(
            # dataset_cfg.batch_size,
            1,
            3,
            h,
            w,
        )
        model_export_onnx(model, train_cfg, dummy_input)

    print(f"Model saved in {model_cfg.model_path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    train_model(cfg.model, cfg.dataset, cfg.train)


if __name__ == "__main__":
    main()
