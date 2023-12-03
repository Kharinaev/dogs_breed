from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToTensor
from tqdm import tqdm


try:
    from dltoolkit.config import DatasetConfig, Params, TrainConfig
    from dltoolkit.dataset import StanfordDogsDataset
    from dltoolkit.model import ResNetClassifier
except ImportError:
    from config import DatasetConfig, Params, TrainConfig
    from dataset import StanfordDogsDataset
    from model import ResNetClassifier


def train_loop(model, n_epochs, criterion, optimizer, dl, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_num = 0
        total_num = 0
        for inputs, labels in tqdm(dl, desc=f"Epoch {epoch+1}/{n_epochs}"):

            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            preds = outputs.detach().cpu().argmax(1)
            correct_num += (preds == labels).sum()
            total_num += len(labels)

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(dl)}, Accuracy: {correct_num/total_num:0.6f}"
        )

    print("Learning finished")


def load_model(n_classes: int, device: torch.device):
    model = ResNetClassifier(n_classes, weights=ResNet18_Weights.DEFAULT).to(device)
    model.train()

    return model


def train_model(train_cfg: TrainConfig, dataset_cfg: DatasetConfig):
    print("Preparing dataset")
    train_ds = StanfordDogsDataset(
        "train",
        dvc_repo=str(Path.cwd()),
        dataset_path=dataset_cfg.dataset_path,
        csv_path=dataset_cfg.csv_path,
        transform=ToTensor(),
    )
    train_dl = DataLoader(train_ds, batch_size=dataset_cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = load_model(train_ds.n_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    print("Start training")
    train_loop(model, train_cfg.num_epochs, criterion, optimizer, train_dl, device)
    torch.save(model.state_dict(), train_cfg.model_save_path)
    print(f"Model saved in {train_cfg.model_save_path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    train_model(cfg.train, cfg.dataset)


if __name__ == "__main__":
    main()
